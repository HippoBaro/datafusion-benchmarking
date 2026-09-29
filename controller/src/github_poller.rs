//! GitHub comment poller.
//!
//! Periodically fetches new PR comments from watched repositories, detects
//! benchmark trigger phrases, and inserts corresponding jobs into SQLite.

use anyhow::Result;
use chrono::{Duration, Utc};
use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::benchmarks::{
    allowed_users_markdown, detect_benchmark, is_benchmark_trigger, is_queue_request,
    is_singular_no_names, usage_message, DetectResult,
};
use crate::config::{Config, RepoEntry, MAX_QUEUED_PER_USER};
use crate::db;
use crate::github::{self, GitHubClient};
use crate::models::{GitHubComment, JobInsert};
use crate::resources::PodResources;

/// Infinite loop that polls GitHub for new PR comments on each watched repo.
///
/// ```text
/// ┌──────────────────────────────────────────────┐
/// │  poll_loop (every POLL_INTERVAL_SECS)        │
/// │  ┌────────────────────────────────────────┐  │
/// │  │ for each repo in WATCHED_REPOS         │  │
/// │  │   fetch comments since last_scan       │  │
/// │  │   for each unseen comment              │  │
/// │  │     "show benchmark queue" → reply     │  │
/// │  │     "run benchmark X"     → insert job │  │
/// │  │   update last_scan                     │  │
/// │  └────────────────────────────────────────┘  │
/// └──────────────────────────────────────────────┘
/// ```
pub async fn poll_loop(
    config: Config,
    pool: SqlitePool,
    gh: GitHubClient,
    token: tokio_util::sync::CancellationToken,
) {
    let interval = tokio::time::Duration::from_secs(config.poll_interval_secs);
    loop {
        for repo in config.benchmark_config.repos.keys() {
            if let Err(e) = poll_repo(&pool, &gh, &config, repo).await {
                warn!(repo, error = ?e, "poll error");
            }
        }
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = token.cancelled() => {
                info!("poller shutting down");
                break;
            }
        }
    }
}

/// Fetch and process recent comments for a single repo.
#[tracing::instrument(skip(pool, gh, config))]
async fn poll_repo(
    pool: &SqlitePool,
    gh: &GitHubClient,
    config: &Config,
    repo: &str,
) -> Result<()> {
    let repo_entry = match config.benchmark_config.repos.get(repo) {
        Some(e) => e,
        None => {
            warn!(repo, "unknown repo, skipping");
            return Ok(());
        }
    };

    let since = match db::get_last_scan(pool, repo).await? {
        Some(ts) => ts,
        None => {
            let default = Utc::now() - Duration::hours(1);
            default.format("%Y-%m-%dT%H:%M:%SZ").to_string()
        }
    };

    let comments = gh.fetch_recent_comments(repo, &since).await?;
    info!(repo, count = comments.len(), "fetched comments");

    for comment in &comments {
        if let Err(e) = process_comment(pool, gh, config, repo, repo_entry, comment).await {
            warn!(comment_id = comment.id, error = ?e, "process comment error");
        }
    }

    // Store a scan timestamp that overlaps by 2 poll intervals so restarts
    // don't miss comments. The seen_comments table deduplicates processing.
    let overlap = Utc::now() - Duration::seconds((config.poll_interval_secs * 2) as i64);
    let scan_ts = overlap.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    db::set_last_scan(pool, repo, &scan_ts).await?;

    Ok(())
}

/// Build the "not allowed" reply posted when a non-whitelisted user triggers a benchmark.
fn not_allowed_message(
    login: &str,
    comment_url: &str,
    allowed_users: &std::collections::HashSet<String>,
    runner_repo_url: Option<&str>,
) -> String {
    let footer = github::issues_footer(runner_repo_url);
    format!(
        "Hi @{login}, thanks for the request ({comment_url}). \
         Only whitelisted users can trigger benchmarks. \
         Allowed users: {}.{footer}",
        allowed_users_markdown(allowed_users)
    )
}

/// Build the reply posted when a user would exceed the per-user queued-jobs cap.
fn rate_limit_message(
    login: &str,
    comment_url: &str,
    pending: i64,
    incoming: i64,
    runner_repo_url: Option<&str>,
) -> String {
    let footer = github::issues_footer(runner_repo_url);
    format!(
        "Hi @{login}, thanks for the request ({comment_url}). \
         You already have {pending} pending benchmark job(s), and this request \
         would add {incoming} more — exceeding the per-user queue limit of \
         {MAX_QUEUED_PER_USER}. Please wait for some of your current runs to \
         finish before submitting more.{footer}"
    )
}

/// Handle a single comment: skip if seen, detect triggers, insert jobs.
async fn process_comment(
    pool: &SqlitePool,
    gh: &GitHubClient,
    config: &Config,
    repo: &str,
    repo_entry: &RepoEntry,
    comment: &GitHubComment,
) -> Result<()> {
    if db::is_comment_seen(pool, comment.id).await? {
        return Ok(());
    }

    let body = comment.body_text();
    let login = comment.login();
    let comment_url = comment.url();
    let issue_url = comment.issue_url_str();
    let runner_repo_url = config.runner_repo_url.as_deref();
    let footer = github::issues_footer(runner_repo_url);

    let Some(pr_number) = pr_number_from_url(issue_url) else {
        return Ok(());
    };

    /// Helper to mark a comment as seen (used for non-trigger early returns).
    async fn mark_seen(
        pool: &SqlitePool,
        comment: &GitHubComment,
        repo: &str,
        pr_number: i64,
    ) -> Result<()> {
        db::mark_comment_seen(
            pool,
            comment.id,
            repo,
            pr_number,
            comment.login(),
            comment.created_at_str(),
        )
        .await
    }

    // Handle queue requests — mark seen only after reply succeeds.
    if is_queue_request(body) {
        info!(pr_number, login, "queue request");
        let jobs = db::get_queue_summary(pool).await?;
        let msg = format!(
            "{}{footer}",
            format_queue_message(login, comment_url, &jobs)
        );
        gh.post_comment(repo, pr_number, &msg).await?;
        mark_seen(pool, comment, repo, pr_number).await?;
        return Ok(());
    }

    // Try to detect benchmark trigger
    let request = match detect_benchmark(body, &config.resource_limits) {
        DetectResult::Parsed(req) => req,
        DetectResult::ConfigError(err) => {
            // YAML config was present but invalid — post a helpful error
            if config.benchmark_config.allowed_users.contains(login) {
                let msg = format!(
                    "Hi @{login}, your benchmark configuration could not be parsed ({comment_url}).\n\n\
                     **Error:** `{err}`\n\n{}{footer}",
                    usage_message()
                );
                gh.post_comment(repo, pr_number, &msg).await?;
            }
            mark_seen(pool, comment, repo, pr_number).await?;
            return Ok(());
        }
        DetectResult::None => {
            // With no allowlist, named/default triggers always parse, so the
            // only `None` cases that look like a trigger are `run benchmark`
            // (singular) with no names.
            if is_benchmark_trigger(body) {
                if !config.benchmark_config.allowed_users.contains(login) {
                    let msg = not_allowed_message(
                        login,
                        comment_url,
                        &config.benchmark_config.allowed_users,
                        runner_repo_url,
                    );
                    gh.post_comment(repo, pr_number, &msg).await?;
                } else {
                    let prefix = if is_singular_no_names(body) {
                        format!(
                            "Hi @{login}, `run benchmark` requires benchmark names ({comment_url}).\n\n"
                        )
                    } else {
                        format!("Hi @{login}, thanks for the request ({comment_url}).\n\n")
                    };
                    let msg = format!("{prefix}{}{footer}", usage_message());
                    gh.post_comment(repo, pr_number, &msg).await?;
                }
            }
            // Mark seen after any reply succeeds (or if not a trigger at all).
            mark_seen(pool, comment, repo, pr_number).await?;
            return Ok(());
        }
    };

    // User must be allowed — mark seen only after reply succeeds.
    if !config.benchmark_config.allowed_users.contains(login) {
        let msg = not_allowed_message(
            login,
            comment_url,
            &config.benchmark_config.allowed_users,
            runner_repo_url,
        );
        gh.post_comment(repo, pr_number, &msg).await?;
        mark_seen(pool, comment, repo, pr_number).await?;
        return Ok(());
    }

    // Reject unsupported sharding before admission, source freezing or fan-out.
    let support = repo_entry.job_type().shard_support();
    let shard_count = match support.resolve(request.shards) {
        Ok(count) => count,
        Err(error) => {
            let msg = format!(
                "Hi @{login}, your benchmark request cannot run ({comment_url}).\n\n**Error:** {error}{footer}"
            );
            gh.post_comment(repo, pr_number, &msg).await?;
            mark_seen(pool, comment, repo, pr_number).await?;
            return Ok(());
        }
    };

    info!(pr_number, login, benchmarks = ?request.benchmarks, requested_shards = ?request.shards, shard_count, "scheduling benchmark");

    // Resolve default benchmarks when "run benchmarks" is used without specific names
    let benchmarks = if request.benchmarks.is_empty() {
        repo_entry.default_standard.clone()
    } else {
        request.benchmarks.clone()
    };

    // Per-user queued-jobs cap. One comment can insert multiple jobs (one per
    // benchmark name); count them all against the cap before inserting any.
    let incoming = benchmarks.len().max(1) as i64 * i64::from(shard_count);
    let pending = db::count_user_pending(pool, login).await?;
    if pending + incoming > MAX_QUEUED_PER_USER {
        let msg = rate_limit_message(login, comment_url, pending, incoming, runner_repo_url);
        gh.post_comment(repo, pr_number, &msg).await?;
        mark_seen(pool, comment, repo, pr_number).await?;
        return Ok(());
    }

    // Only metadata is shared. Every worker independently clones, builds and
    // discovers its cases, including workers admitted in later quota waves.
    let resolved_sources = if shard_count > 1 {
        match gh
            .freeze_sources(
                repo,
                pr_number,
                request.baseline_ref.as_deref(),
                request.changed_ref.as_deref(),
            )
            .await
        {
            Ok(sources) => Some(serde_json::to_string(&sources)?),
            Err(e) => {
                gh.post_comment(repo, pr_number, &format!("Could not resolve immutable commits for sharding ({comment_url}): {e:#}{footer}")).await?;
                mark_seen(pool, comment, repo, pr_number).await?;
                return Ok(());
            }
        }
    } else {
        None
    };

    let pr_url = format!("https://github.com/{}/pull/{}", repo, pr_number);
    let env_vars_json = serde_json::to_string(&request.env_vars)?;
    let baseline_env_json = serde_json::to_string(&request.baseline_env_vars)?;
    let changed_env_json = serde_json::to_string(&request.changed_env_vars)?;
    let job_type = repo_entry.job_type().as_str();
    let targets: Vec<String> = if benchmarks.is_empty() {
        vec!["[]".to_string()]
    } else {
        benchmarks
            .iter()
            .map(|name| serde_json::to_string(&[name]))
            .collect::<Result<_, _>>()?
    };
    let mut jobs = Vec::new();
    for target in &targets {
        for index in 0..shard_count {
            jobs.push(JobInsert {
                comment_id: comment.id,
                repo,
                pr_number,
                pr_url: &pr_url,
                login,
                benchmarks: target,
                env_vars: &env_vars_json,
                baseline_env_vars: &baseline_env_json,
                changed_env_vars: &changed_env_json,
                baseline_ref: request.baseline_ref.as_deref(),
                changed_ref: request.changed_ref.as_deref(),
                job_type,
                resources: &request.resources,
                shard: crate::sharding::Shard {
                    count: shard_count,
                    index,
                },
                resolved_source_json: resolved_sources.as_deref(),
            });
        }
    }
    if shard_count > 1 {
        db::enqueue_jobs(pool, &jobs, comment.created_at_str()).await?;
    } else {
        // Preserve ordinary invocation ingestion; atomic fan-out is opt-in.
        mark_seen(pool, comment, repo, pr_number).await?;
        for job in &jobs {
            db::insert_job(pool, job).await?;
        }
    }

    // React with rocket
    if let Err(e) = gh.post_reaction(repo, comment.id, "rocket").await {
        warn!(error = %e, "failed to post reaction");
    }

    Ok(())
}

/// Extract the PR/issue number from a GitHub `issue_url` (last path segment).
/// Returns `None` if the URL doesn't end with a numeric segment.
fn pr_number_from_url(url: &str) -> Option<i64> {
    url.trim_end_matches('/')
        .rsplit('/')
        .next()
        .and_then(|s| s.parse().ok())
}

/// The pod sizing a queued job asked for, or `None` when it takes every
/// controller default.
fn job_resources(job: &crate::models::BenchmarkJob) -> Option<String> {
    PodResources {
        cpu: job.cpu_request.clone(),
        memory: job.memory_request.clone(),
        arch: job.cpu_arch.clone(),
    }
    .summary()
}

/// Build a markdown table of pending/active jobs for a "show benchmark queue" reply.
fn format_queue_message(
    login: &str,
    comment_url: &str,
    jobs: &[crate::models::BenchmarkJob],
) -> String {
    let mut lines = vec![format!(
        "Hi @{login}, you asked to view the benchmark queue ({comment_url}).\n"
    )];

    if jobs.is_empty() {
        lines.push("No pending jobs.".to_string());
    } else {
        // A Resources column is dead weight on a queue of default-sized pods,
        // so it appears only once a job in the queue asked for something.
        let show_resources = jobs.iter().any(|job| job_resources(job).is_some());
        let resources_header = if show_resources { " Resources |" } else { "" };
        let resources_divider = if show_resources { " --- |" } else { "" };

        lines.push(format!(
            "| Comment | Repo | PR | User | Benchmarks |{resources_header} Status |"
        ));
        lines.push(format!(
            "| --- | --- | --- | --- | --- |{resources_divider} --- |"
        ));
        for job in jobs {
            let comment_link = format!(
                "[#{}]({}#issuecomment-{})",
                job.comment_id, job.pr_url, job.comment_id
            );
            let resources = if show_resources {
                format!(
                    " {} |",
                    job_resources(job).unwrap_or_else(|| "default".into())
                )
            } else {
                String::new()
            };
            lines.push(format!(
                "| {} | {} | #{} | {} | {}{} |{} {} |",
                comment_link,
                job.repo,
                job.pr_number,
                job.login,
                job.benchmarks,
                job.shard_label(),
                resources,
                job.status
            ));
        }
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::BenchmarkJob;

    // ── pr_number_from_url ──────────────────────────────────────────

    #[test]
    fn pr_number_standard_url() {
        let url = "https://api.github.com/repos/apache/datafusion/issues/42";
        assert_eq!(pr_number_from_url(url), Some(42));
    }

    #[test]
    fn pr_number_trailing_slash() {
        let url = "https://api.github.com/repos/apache/datafusion/issues/42/";
        assert_eq!(pr_number_from_url(url), Some(42));
    }

    #[test]
    fn pr_number_empty() {
        assert_eq!(pr_number_from_url(""), None);
    }

    #[test]
    fn pr_number_not_a_number() {
        assert_eq!(pr_number_from_url("https://example.com/not-a-number"), None);
    }

    #[test]
    fn pr_number_trailing_slash_only() {
        assert_eq!(pr_number_from_url("https://example.com/"), None);
    }

    // ── not_allowed_message ─────────────────────────────────────────

    #[test]
    fn not_allowed_msg_contains_fields() {
        let users: std::collections::HashSet<String> =
            ["alamb"].iter().map(|s| s.to_string()).collect();
        let msg = not_allowed_message("testuser", "https://example.com/comment/1", &users, None);
        assert!(msg.contains("@testuser"));
        assert!(msg.contains("https://example.com/comment/1"));
        assert!(msg.contains("whitelisted") || msg.contains("Whitelisted"));
    }

    #[test]
    fn rate_limit_msg_contains_fields() {
        let msg = rate_limit_message("alice", "https://example.com/c/9", 12, 4, None);
        assert!(msg.contains("@alice"));
        assert!(msg.contains("https://example.com/c/9"));
        assert!(msg.contains("12 pending"));
        assert!(msg.contains("would add 4"));
        assert!(msg.contains("per-user queue limit of 15"));
    }

    #[tokio::test]
    async fn unsupported_setup_is_rejected_before_admission_freezing_and_fanout() {
        use crate::config::BenchmarkConfig;
        use crate::models::GitHubUser;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gh = GitHubClient::test_client(format!("http://{}", listener.local_addr().unwrap()));
        let server = tokio::spawn(async move {
            let (mut socket, _) =
                tokio::time::timeout(std::time::Duration::from_secs(10), listener.accept())
                    .await
                    .unwrap()
                    .unwrap();
            let mut bytes = Vec::new();
            let body_start = loop {
                let mut buffer = [0; 2048];
                let n = socket.read(&mut buffer).await.unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&buffer[..n]);
                if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    break end + 4;
                }
            };
            // Only an error comment: no PR/ref resolution, quota rejection or reaction.
            let headers = String::from_utf8_lossy(&bytes[..body_start]);
            assert!(headers.starts_with("POST /repos/test/repo/issues/42/comments HTTP/1.1\r\n"));
            let length: usize = headers
                .lines()
                .filter_map(|l| l.split_once(':'))
                .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .unwrap()
                .1
                .trim()
                .parse()
                .unwrap();
            let received = bytes.len();
            bytes.resize(body_start + length, 0);
            socket.read_exact(&mut bytes[received..]).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes[body_start..]).unwrap();
            let message = json["body"].as_str().unwrap();
            assert!(message.contains("DataFusion"));
            assert!(message.contains("shards greater than 1 are not supported"));
            assert!(message.contains("shards: 1"));
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                .await
                .unwrap();
        });
        let config = Config {
            github_token: "unused".into(),
            database_url: "sqlite::memory:".into(),
            benchmark_config: BenchmarkConfig {
                allowed_users: ["alice".into()].into(),
                repos: Default::default(),
            },
            poll_interval_secs: 2,
            reconcile_interval_secs: 3,
            k8s_namespace: "test".into(),
            runner_image: "unused".into(),
            default_cpu: "12".into(),
            default_memory: "65Gi".into(),
            ephemeral_storage: "128Gi".into(),
            default_machine_family: "c4a".into(),
            resource_limits: Default::default(),
            active_deadline_secs: 7200,
            ttl_after_finished_secs: 3600,
            storage_class: "unused".into(),
            sccache_gcs_bucket: None,
            data_cache_bucket: None,
            runner_repo_url: None,
        };
        let pool = db::connect("sqlite::memory:").await.unwrap();
        let resources = PodResources::default();
        // Only one slot remains: unsupported sharding must win over a quota error.
        for id in 1..MAX_QUEUED_PER_USER {
            db::mark_comment_seen(&pool, id, "test/repo", 42, "alice", "2024-01-01")
                .await
                .unwrap();
            db::insert_job(
                &pool,
                &JobInsert {
                    comment_id: id,
                    repo: "test/repo",
                    pr_number: 42,
                    pr_url: "https://github.com/test/repo/pull/42",
                    login: "alice",
                    benchmarks: "[\"tpch\"]",
                    env_vars: "{}",
                    baseline_env_vars: "{}",
                    changed_env_vars: "{}",
                    baseline_ref: None,
                    changed_ref: None,
                    job_type: "datafusion",
                    resources: &resources,
                    shard: Default::default(),
                    resolved_source_json: None,
                },
            )
            .await
            .unwrap();
        }
        let comment = GitHubComment {
            id: 90000,
            body: Some("run benchmark tpch\nshards: 8".into()),
            user: Some(GitHubUser {
                login: "alice".into(),
            }),
            html_url: Some("https://github.com/test/repo/pull/42#issuecomment-90000".into()),
            created_at: Some("2024-01-01".into()),
            issue_url: Some("https://api.github.com/repos/test/repo/issues/42".into()),
        };
        let entry = RepoEntry {
            kind: "datafusion".into(),
            default_standard: vec![],
        };
        process_comment(&pool, &gh, &config, "test/repo", &entry, &comment)
            .await
            .unwrap();
        process_comment(&pool, &gh, &config, "test/repo", &entry, &comment)
            .await
            .unwrap();
        server.await.unwrap();
        let rows = sqlx::query_as::<_, BenchmarkJob>(
            "SELECT * FROM benchmark_jobs WHERE comment_id = 90000",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert!(rows.is_empty());
        assert!(db::is_comment_seen(&pool, comment.id).await.unwrap());
        assert_eq!(
            db::count_user_pending(&pool, "alice").await.unwrap(),
            MAX_QUEUED_PER_USER - 1
        );
    }

    // ── format_queue_message ────────────────────────────────────────

    #[test]
    fn format_queue_empty() {
        let msg = format_queue_message("alice", "https://example.com/c/1", &[]);
        assert!(msg.contains("No pending jobs."));
    }

    fn test_job() -> BenchmarkJob {
        BenchmarkJob {
            id: 1,
            comment_id: 100,
            repo: "apache/datafusion".to_string(),
            pr_number: 42,
            pr_url: "https://github.com/apache/datafusion/pull/42".to_string(),
            login: "alice".to_string(),
            benchmarks: "[\"tpch\"]".to_string(),
            env_vars: "{}".to_string(),
            baseline_env_vars: "{}".to_string(),
            changed_env_vars: "{}".to_string(),
            baseline_ref: None,
            changed_ref: None,
            job_type: "standard".to_string(),
            cpu_request: None,
            memory_request: None,
            cpu_arch: None,
            k8s_job_name: None,
            status: "pending".to_string(),
            error_message: None,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            runner_token: None,
            effective_shards: 1,
            shard_index: 0,
            assignment_version: None,
            resolved_source_json: None,
        }
    }

    #[test]
    fn format_queue_identifies_shards() {
        let mut job = test_job();
        job.effective_shards = 4;
        job.shard_index = 1;
        let msg = format_queue_message("alice", "https://example.com/c/1", &[job]);
        assert!(msg.contains("shard 2/4"));
    }

    #[test]
    fn format_queue_with_jobs() {
        let msg = format_queue_message("bob", "https://example.com/c/2", &[test_job()]);
        assert!(msg.contains("| Comment |"));
        assert!(
            msg.contains("[#100](https://github.com/apache/datafusion/pull/42#issuecomment-100)")
        );
        assert!(msg.contains("apache/datafusion"));
        assert!(msg.contains("#42"));
    }

    /// A queue of default-sized pods has nothing to say about resources.
    #[test]
    fn format_queue_omits_the_resources_column_by_default() {
        let msg = format_queue_message("bob", "https://example.com/c/2", &[test_job()]);
        assert!(!msg.contains("Resources"));
    }

    #[test]
    fn format_queue_shows_the_requested_resources() {
        let mut custom = test_job();
        custom.id = 2;
        custom.cpu_request = Some("16".into());
        custom.memory_request = Some("128Gi".into());
        custom.cpu_arch = Some("amd64".into());

        let msg = format_queue_message("bob", "https://example.com/c/2", &[test_job(), custom]);

        assert!(msg.contains("| Benchmarks | Resources | Status |"));
        assert!(msg.contains("| 16 CPU, 128Gi, amd64 | pending |"));
        // The unconfigured job still fills the column.
        assert!(msg.contains("| default | pending |"));
    }
}
