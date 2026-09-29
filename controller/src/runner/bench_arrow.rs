//! Arrow-rs criterion benchmark runner — ports `run_arrow_criterion.sh`.

use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use tracing::{info, warn};

use crate::github;
use crate::runner::config::RunnerConfig;
use crate::runner::criterion::{self, Criterion, Run};
use crate::runner::execution::{execute, partition};
use crate::runner::git;
use crate::runner::monitor;
use crate::runner::poster::CommentPoster;
use crate::runner::shell;
use crate::runner::trigger;
use crate::sharding::ShardSupport;

pub const SHARD_SUPPORT: ShardSupport = ShardSupport::Partitioned;

/// Run an arrow-rs criterion benchmark comparing a PR branch to its merge-base.
pub async fn run(config: &RunnerConfig, poster: &CommentPoster) -> Result<()> {
    let repo_url = config.repo_url();
    let bench_name = &config.bench_name;
    config.shard.validate()?;
    let frozen = config.frozen_sources.as_ref();
    ensure!(
        config.shard.count() == 1 || frozen.is_some(),
        "distributed workers require frozen commits"
    );
    if let Some(sources) = frozen {
        sources.validate()?;
    }

    let branch_dir = PathBuf::from("/workspace/arrow-rs-branch");
    let base_dir = PathBuf::from("/workspace/arrow-rs-base");

    // Clone and checkout PR branch
    info!("=== Cloning PR branch ===");
    git::clone_shallow(&repo_url, &branch_dir, 200).await?;
    let branch_name = if let Some(sources) = frozen {
        git::checkout_frozen(&branch_dir, &sources.changed_sha).await?;
        sources.pr_head_ref.clone()
    } else {
        git::checkout_pr(&config.pr_url, &branch_dir).await?
    };
    git::submodule_update(&branch_dir).await?;
    let merge_base = if let Some(sources) = frozen {
        sources.baseline_sha.clone()
    } else {
        git::merge_base(&branch_dir).await?
    };
    let bench_branch_name = git::sanitize_branch_name(&branch_name);
    git::cargo_update(&branch_dir).await?;
    // Keep the original ref/update order for ordinary invocations.
    if frozen.is_none() {
        if let Some(ref changed_ref) = config.changed_ref {
            info!(changed_ref, "=== Checking out custom changed ref ===");
            git::fetch_pr_ref(&config.pr_url, &branch_dir).await?;
            git::fetch_origin(&branch_dir).await?;
            git::checkout(&branch_dir, changed_ref).await?;
        }
    }

    // Determine baseline: custom ref or merge-base
    let baseline_display: String;
    info!("=== Cloning merge-base ===");
    git::clone_shallow(&repo_url, &base_dir, 200).await?;
    if let Some(sources) = frozen {
        git::checkout_frozen(&base_dir, &sources.baseline_sha).await?;
        baseline_display = config
            .baseline_ref
            .clone()
            .unwrap_or_else(|| merge_base.clone());
    } else if let Some(ref baseline_ref) = config.baseline_ref {
        info!(baseline_ref, "=== Checking out custom baseline ref ===");
        git::fetch_pr_ref(&config.pr_url, &base_dir).await?;
        git::fetch_origin(&base_dir).await?;
        git::checkout(&base_dir, baseline_ref).await?;
        baseline_display = baseline_ref.clone();
    } else {
        git::checkout(&base_dir, &merge_base).await?;
        baseline_display = merge_base.clone();
    }
    git::submodule_update(&base_dir).await?;
    git::cargo_update(&base_dir).await?;

    // Pre-install stable toolchain to avoid rustup race in parallel builds
    git::rustup_stable().await?;

    // Post "running" comment
    let uname = shell::uname().await;
    let instance_type = shell::node_instance_type().await;
    let pod_resources = shell::pod_resources();
    let lscpu = shell::lscpu().await;
    let bench_command_display = format!(
        "cargo bench --features=arrow,async,test_common,experimental,object_store --bench {bench_name}"
    );
    let changed_display = config.changed_ref.as_deref().unwrap_or(&branch_name);
    let changed_sha = git::rev_parse_head(&branch_dir).await?;
    let base_sha = git::rev_parse_head(&base_dir).await?;
    let baseline_label = if config.baseline_ref.is_some() {
        baseline_display.clone()
    } else {
        format!("{} (merge-base)", &base_sha[..7.min(base_sha.len())])
    };
    let comparison = trigger::Comparison {
        repo: &config.repo,
        changed_display,
        changed_sha: &changed_sha,
        baseline_label: &baseline_label,
        base_sha: &base_sha,
    };
    let config_block = trigger::config_block(config, bench_name);

    let footer = github::issues_footer(config.runner_repo_url.as_deref());
    let running_body = format!(
        "\u{1f916} Arrow criterion benchmark running (GKE) | [trigger]({})\n\
         **Instance:** `{instance_type}` ({pod_resources}) | `{uname}`\n\
         <details><summary>CPU Details (lscpu)</summary>\n\n\
         ```\n\
         {lscpu}\n\
         ```\n\n\
         </details>\n\n\
         {comparison}\n\n\
         {config_block}\
         BENCH_COMMAND={bench_command_display}\n\
         Results will be posted here when complete{footer}",
        config.comment_url,
        comparison = comparison.line(),
    );
    let pr_number = config.pr_number()?;
    poster
        .post_comment(&config.repo, pr_number, &running_body)
        .await?;

    // Every worker builds, partitions, measures and compares in the same way.
    // The partition operation alone decides how much of the work it owns.
    info!("=== Compiling PR branch and merge-base in parallel ===");
    let mut build_args = bench_command_args(bench_name);
    build_args.push("--no-run".into());
    let branch_build = shell::spawn_command(
        "cargo",
        &str_slice(&build_args),
        &branch_dir,
        "/tmp/branch_build.log",
    );
    let base_build = shell::spawn_command(
        "cargo",
        &str_slice(&build_args),
        &base_dir,
        "/tmp/base_build.log",
    );
    branch_build
        .await
        .context("branch build task panicked")?
        .context("branch build failed")?;
    let baseline_available = match base_build.await {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            warn!("Baseline build failed (benchmark may be new): {e:#}");
            false
        }
        Err(e) => {
            warn!("Baseline build task panicked (benchmark may be new): {e:#}");
            false
        }
    };
    info!("=== Compilation complete ===");

    let harness = Criterion {
        bench_args: bench_command_args(bench_name),
    };
    let base_env = config.baseline_env_args();
    let branch_env = config.changed_env_args();
    let base = if baseline_available {
        Some(
            partition(
                &harness,
                Run::new("base", "main", &config.bench_filter, &base_dir, &base_env),
                config.shard,
                bench_name,
            )
            .await?,
        )
    } else {
        None
    };
    let branch = partition(
        &harness,
        Run::new(
            "changed",
            &bench_branch_name,
            &config.bench_filter,
            &branch_dir,
            &branch_env,
        ),
        config.shard,
        bench_name,
    )
    .await?;

    let base_stats = if let Some(base) = &base {
        info!("=== Running benchmark on merge-base ===");
        execute(&harness, base).await?
    } else {
        info!("=== Skipping merge-base benchmark (baseline build failed) ===");
        None
    };
    info!("=== Running benchmark on PR branch ===");
    let branch_stats = execute(&harness, &branch).await?;

    let report = criterion::compare(
        base.as_ref().filter(|_| base_stats.is_some()),
        &branch,
        branch_stats.is_some(),
    )
    .await?;
    let resource_section = [
        ("base (merge-base)", &base_stats),
        ("branch", &branch_stats),
    ]
    .into_iter()
    .filter_map(|(name, stats)| {
        stats
            .as_ref()
            .map(|stats| monitor::format_resource_comment(name, stats))
    })
    .collect::<Vec<_>>()
    .join("\n");
    let result_body = if base_stats.is_some() || branch_stats.is_none() {
        format_result_comment(
            &config.comment_url,
            &comparison.line(),
            &config_block,
            &report,
            &resource_section,
            &instance_type,
            &pod_resources,
            &lscpu,
            &footer,
        )
    } else {
        format_branch_only_result_comment(
            &config.comment_url,
            &comparison.line(),
            &config_block,
            &report,
            &resource_section,
            &instance_type,
            &pod_resources,
            &lscpu,
            &footer,
        )
    };
    poster
        .post_comment(&config.repo, pr_number, &result_body)
        .await?;

    Ok(())
}

/// Build cargo bench args for arrow-rs criterion.
fn bench_command_args(bench_name: &str) -> Vec<String> {
    vec![
        "bench".to_string(),
        "--features=arrow,async,test_common,experimental,object_store".to_string(),
        "--bench".to_string(),
        bench_name.to_string(),
    ]
}

/// Format the result comment body.
///
/// `comparison` and `config_block` restate what was run, so the result reads on
/// its own rather than only linking back to the trigger comment.
#[allow(clippy::too_many_arguments)]
fn format_result_comment(
    comment_url: &str,
    comparison: &str,
    config_block: &str,
    report: &str,
    resource_section: &str,
    instance_type: &str,
    pod_resources: &str,
    lscpu: &str,
    footer: &str,
) -> String {
    format!(
        "\u{1f916} Arrow criterion benchmark completed (GKE) | [trigger]({comment_url})\n\n\
         **Instance:** `{instance_type}` ({pod_resources})\n\n\
         {comparison}\n\n\
         {config_block}\
         <details><summary>CPU Details (lscpu)</summary>\n\n\
         ```\n\
         {lscpu}\n\
         ```\n\n\
         </details>\n\n\
         <details><summary>Details</summary>\n\
         <p>\n\n\
         ```\n\
         {report}\
         ```\n\n\
         </p>\n\
         </details>\n\n\
         <details><summary>Resource Usage</summary>\n\n\
         {resource_section}\
         </details>\n\
         {footer}"
    )
}

/// Format the result comment body for branch-only runs (no baseline comparison).
#[allow(clippy::too_many_arguments)]
fn format_branch_only_result_comment(
    comment_url: &str,
    comparison: &str,
    config_block: &str,
    report: &str,
    resource_section: &str,
    instance_type: &str,
    pod_resources: &str,
    lscpu: &str,
    footer: &str,
) -> String {
    format!(
        "\u{1f916} Arrow criterion benchmark completed (GKE) | [trigger]({comment_url})\n\n\
         **Instance:** `{instance_type}` ({pod_resources})\n\n\
         {comparison}\n\n\
         {config_block}\
         <details><summary>CPU Details (lscpu)</summary>\n\n\
         ```\n\
         {lscpu}\n\
         ```\n\n\
         </details>\n\n\
         **New benchmark — branch-only results (no baseline comparison)**\n\n\
         <details><summary>Details</summary>\n\
         <p>\n\n\
         ```\n\
         {report}\
         ```\n\n\
         </p>\n\
         </details>\n\n\
         <details><summary>Resource Usage</summary>\n\n\
         {resource_section}\
         </details>\n\
         {footer}"
    )
}

/// Convert Vec<String> to a slice of &str for run_command.
fn str_slice(v: &[String]) -> Vec<&str> {
    v.iter().map(|s| s.as_str()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_args_construction() {
        let args = bench_command_args("concatenate_kernel");
        assert_eq!(
            args,
            vec![
                "bench",
                "--features=arrow,async,test_common,experimental,object_store",
                "--bench",
                "concatenate_kernel"
            ]
        );
    }

    /// Stand-ins for the two "what was run" sections the runner passes in.
    const COMPARISON: &str =
        "Comparing my-branch (aaa) to bbb (merge-base) [diff](https://example.com/diff)";
    const CONFIG_BLOCK: &str = "<details><summary>Run configuration</summary>\n\n```yaml\nrun benchmark concatenate_kernel\n```\n\n</details>\n\n";

    #[test]
    fn result_comment_format() {
        let comment = format_result_comment(
            "https://example.com/comment",
            COMPARISON,
            CONFIG_BLOCK,
            "test report\n",
            "resources\n",
            "c4a-standard-48",
            "12 vCPU / 65 GiB",
            "lscpu output",
            "",
        );
        assert!(comment.contains("Arrow criterion benchmark completed"));
        assert!(comment.contains("[trigger](https://example.com/comment)"));
        assert!(comment.contains("test report"));
        assert!(comment.contains("Resource Usage"));
        assert!(comment.contains("c4a-standard-48"));
        assert!(comment.contains("12 vCPU / 65 GiB"));
        assert!(comment.contains("lscpu output"));
        assert!(comment.contains("Comparing my-branch (aaa)"));
        assert!(comment.contains("run benchmark concatenate_kernel"));
    }

    #[test]
    fn branch_only_result_comment_format() {
        let comment = format_branch_only_result_comment(
            "https://example.com/comment",
            COMPARISON,
            CONFIG_BLOCK,
            "branch report\n",
            "branch resources\n",
            "c4a-standard-48",
            "12 vCPU / 65 GiB",
            "lscpu output",
            "",
        );
        assert!(comment.contains("Arrow criterion benchmark completed"));
        assert!(comment.contains("Comparing my-branch (aaa)"));
        assert!(comment.contains("run benchmark concatenate_kernel"));
        assert!(comment.contains("New benchmark — branch-only results"));
        assert!(comment.contains("[trigger](https://example.com/comment)"));
        assert!(comment.contains("branch report"));
        assert!(comment.contains("Resource Usage"));
        assert!(comment.contains("branch resources"));
        assert!(comment.contains("c4a-standard-48"));
        assert!(comment.contains("12 vCPU / 65 GiB"));
        assert!(comment.contains("lscpu output"));
    }
}
