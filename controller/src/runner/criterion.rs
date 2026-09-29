//! Criterion discovery/execution adapter shared by all repository setups.
//! Cargo arguments, working directory and environment are supplied by callers.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::runner::{
    build_env, criterion_sharding as cases,
    execution::{Harness, Partition},
    monitor::ResourceStats,
    shell,
};

pub struct Criterion {
    pub bench_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    pub side: String,
    pub label: String,
    pub filter: Option<String>,
    pub dir: PathBuf,
    pub env: Vec<String>,
    pub target_override: Option<PathBuf>,
}

impl Run {
    pub fn new(side: &str, label: &str, filter: &str, dir: &Path, env: &[String]) -> Self {
        Self {
            side: side.into(),
            label: label.into(),
            filter: (!filter.is_empty()).then(|| filter.into()),
            dir: dir.into(),
            env: env.into(),
            target_override: None,
        }
    }

    pub fn target_dir(&self) -> PathBuf {
        self.target_override
            .clone()
            .unwrap_or_else(|| self.dir.join("target"))
    }
}

#[derive(Deserialize)]
struct Metadata {
    target_directory: PathBuf,
}

fn cargo_args(env: &[String], args: Vec<String>) -> Vec<String> {
    env.iter()
        .cloned()
        .chain(build_env::args())
        .chain(["cargo".into()])
        .chain(args)
        .collect()
}

fn refs(args: &[String]) -> Vec<&str> {
    args.iter().map(String::as_str).collect()
}

impl Criterion {
    async fn list(&self, run: &Run) -> Result<BTreeSet<String>> {
        let mut args = self.bench_args.clone();
        args.extend(["--", "--list", "--format", "terse"].map(String::from));
        args.extend(run.filter.clone());
        cases::parse_listing(
            &shell::run_command("env", &refs(&cargo_args(&run.env, args)), &run.dir).await?,
        )
    }
}

impl Harness for Criterion {
    type Plan = Run;
    type Output = ResourceStats;

    async fn discover(&self, run: &mut Run) -> Result<BTreeSet<String>> {
        let args = cargo_args(
            &run.env,
            ["metadata", "--no-deps", "--format-version=1"]
                .map(String::from)
                .into(),
        );
        let metadata: Metadata =
            serde_json::from_str(&shell::run_command("env", &refs(&args), &run.dir).await?)?;
        run.target_override = Some(metadata.target_directory);
        // Independent artifact names prevent collisions (e.g. a branch called
        // "main"). Identity partitions never alter the original baseline name.
        run.label = run.side.clone();
        self.list(run).await
    }

    async fn select(
        &self,
        run: &mut Run,
        inventory: &BTreeSet<String>,
        selected: &BTreeSet<String>,
    ) -> Result<()> {
        run.filter = cases::execution_filter(
            inventory,
            selected,
            run.filter.as_deref().unwrap_or_default(),
        )?;
        if selected != inventory {
            ensure!(
                self.list(run).await? == *selected,
                "unstable case IDs or incompatible Criterion selector"
            );
        }
        Ok(())
    }

    async fn execute(&self, run: &Run) -> Result<ResourceStats> {
        let mut args = self.bench_args.clone();
        args.extend(["--", "--save-baseline", &run.label].map(String::from));
        args.extend(run.filter.clone());
        let (command, args) = if run.env.is_empty() {
            ("cargo", args)
        } else {
            ("env", cargo_args(&run.env, args))
        };
        let (_, stats) =
            shell::run_command_monitored(command, &refs(&args), &run.dir, None).await?;
        Ok(stats)
    }

    async fn measured_cases(&self, run: &Run, _: &ResourceStats) -> Result<BTreeSet<String>> {
        let export = shell::run_command(
            "critcmp",
            &[
                "--target-dir",
                &run.target_dir().to_string_lossy(),
                "--export",
                &run.label,
            ],
            &run.dir,
        )
        .await?;
        cases::export_ids(&export, &run.label)
    }
}

fn comparison_args(
    base: Option<&Partition<Run>>,
    branch: &Partition<Run>,
    branch_measured: bool,
) -> Vec<String> {
    let labels: Vec<_> = base
        .into_iter()
        .chain(branch_measured.then_some(branch))
        .map(|p| p.plan.label.clone())
        .collect();
    if labels.is_empty() {
        return Vec::new();
    }
    let mut args = Vec::new();
    if let Some(target) = &branch.plan.target_override {
        args.extend(["--target-dir".into(), target.to_string_lossy().into_owned()]);
    }
    args.extend(labels);
    args
}

/// Compare only measured sides using native Criterion artifacts for every count.
pub async fn compare(
    base: Option<&Partition<Run>>,
    branch: &Partition<Run>,
    branch_measured: bool,
) -> Result<String> {
    let args = comparison_args(base, branch, branch_measured);
    if args.is_empty() {
        return Ok("Empty shard: no matching cases on either side; no measurements run.\n".into());
    }
    if let Some(base) = base {
        let src = base.plan.target_dir().join("criterion");
        let dst = branch.plan.target_dir().join("criterion");
        if src.exists() {
            let _ = shell::run_command(
                "cp",
                &[
                    "-r",
                    &format!("{}/.", src.to_string_lossy()),
                    &dst.to_string_lossy(),
                ],
                Path::new("/"),
            )
            .await;
        }
    }
    shell::run_command("critcmp", &refs(&args), &branch.plan.dir)
        .await
        .context("critcmp")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_build_settings_follow_side_overrides() {
        let args = cargo_args(&["CARGO_BUILD_JOBS=99".into()], vec!["bench".into()]);
        assert!(
            args.iter().position(|a| a == "CARGO_BUILD_JOBS=5").unwrap()
                > args
                    .iter()
                    .position(|a| a == "CARGO_BUILD_JOBS=99")
                    .unwrap()
        );
        assert_eq!(&args[args.len() - 2..], &["cargo", "bench"]);
    }

    fn materialized(side: &str, ids: &[&str]) -> Partition<Run> {
        Partition {
            plan: Run {
                target_override: Some("/custom-target".into()),
                ..Run::new(side, side, "", Path::new("/nonexistent"), &[])
            },
            coverage: Some(ids.iter().map(|id| (*id).into()).collect()),
        }
    }

    #[test]
    fn native_comparison_handles_identity_and_materialized_plans() {
        let base = Partition {
            plan: Run::new("base", "main", "", Path::new("/base"), &[]),
            coverage: None,
        };
        let branch = Partition {
            plan: Run::new("changed", "topic_branch", "", Path::new("/branch"), &[]),
            coverage: None,
        };
        assert_eq!(
            comparison_args(Some(&base), &branch, true),
            ["main", "topic_branch"]
        );
        assert_eq!(comparison_args(None, &branch, true), ["topic_branch"]);
        let base = materialized("base", &["common", "removed"]);
        let branch = materialized("changed", &["common", "added"]);
        assert_eq!(
            comparison_args(Some(&base), &branch, true),
            ["--target-dir", "/custom-target", "base", "changed"]
        );
        assert_eq!(
            comparison_args(Some(&base), &branch, false),
            ["--target-dir", "/custom-target", "base"]
        );
        assert_eq!(
            comparison_args(None, &branch, true),
            ["--target-dir", "/custom-target", "changed"]
        );
        assert!(comparison_args(None, &branch, false).is_empty());
    }
}
