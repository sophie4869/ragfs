//! Index sync helpers for NAS deployments.
//!
//! The indexer writes locally; this module mirrors a completed/stable snapshot
//! to a remote staging directory, switches a remote symlink, then calls the
//! running `ragfs serve` reload endpoint.

use anyhow::{Context, Result};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct SyncConfig {
    pub source_bundle: PathBuf,
    pub remote: String,
    pub remote_stage: String,
    pub remote_current: String,
    pub reload_url: String,
    pub reload_on_remote: bool,
    pub token: Option<String>,
    pub interval: Duration,
    pub settle: Duration,
    pub dry_run: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Fingerprint {
    files: u64,
    bytes: u64,
    latest_modified_ns: u128,
}

/// Return the directory that should be synced for a Lance index.
///
/// `ragfs` stores the Lance dataset as `.../index.lance` and the embedding
/// model marker beside it, so the synced bundle is the parent directory.
pub fn index_bundle_dir(index_path: &Path) -> Result<PathBuf> {
    if index_path.file_name() == Some(OsStr::new("index.lance")) {
        return index_path
            .parent()
            .map(Path::to_path_buf)
            .context("index.lance has no parent directory");
    }

    if index_path.join("index.lance").exists() {
        return Ok(index_path.to_path_buf());
    }

    index_path
        .parent()
        .map(Path::to_path_buf)
        .context("index path has no parent directory")
}

pub async fn watch(config: SyncConfig) -> Result<()> {
    if config.interval.is_zero() {
        anyhow::bail!("--interval-secs must be greater than 0");
    }

    let mut last_synced = None;
    loop {
        match stable_fingerprint(&config.source_bundle, config.settle).await {
            Ok(fingerprint) if last_synced != Some(fingerprint) => {
                sync_stable_snapshot(&config)?;
                last_synced = Some(fingerprint);
            }
            Ok(_) => {
                info!("index unchanged; skipping sync");
            }
            Err(error) => {
                warn!("index is not ready to sync: {error:#}");
            }
        }

        tokio::time::sleep(config.interval).await;
    }
}

pub async fn sync_once(config: &SyncConfig) -> Result<()> {
    if !config.source_bundle.is_dir() {
        anyhow::bail!(
            "Local index bundle does not exist or is not a directory: {}",
            config.source_bundle.display()
        );
    }

    let _ = stable_fingerprint(&config.source_bundle, config.settle).await?;
    sync_stable_snapshot(config)
}

fn sync_stable_snapshot(config: &SyncConfig) -> Result<()> {
    run_rsync(config)?;
    switch_remote_current(config)?;
    reload_server(config)?;
    Ok(())
}

async fn stable_fingerprint(path: &Path, settle: Duration) -> Result<Fingerprint> {
    let before = fingerprint_dir(path)?;
    if !settle.is_zero() {
        tokio::time::sleep(settle).await;
    }
    let after = fingerprint_dir(path)?;

    if before != after {
        anyhow::bail!(
            "local index changed during settle window ({} files/{} bytes -> {} files/{} bytes)",
            before.files,
            before.bytes,
            after.files,
            after.bytes
        );
    }

    Ok(after)
}

fn fingerprint_dir(path: &Path) -> Result<Fingerprint> {
    let mut fingerprint = Fingerprint {
        files: 0,
        bytes: 0,
        latest_modified_ns: 0,
    };
    collect_fingerprint(path, &mut fingerprint)?;
    Ok(fingerprint)
}

fn collect_fingerprint(path: &Path, fingerprint: &mut Fingerprint) -> Result<()> {
    for entry in std::fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let entry_path = entry.path();

        if file_type.is_dir() {
            collect_fingerprint(&entry_path, fingerprint)?;
        } else if file_type.is_file() {
            let metadata = entry.metadata()?;
            fingerprint.files += 1;
            fingerprint.bytes += metadata.len();
            fingerprint.latest_modified_ns = fingerprint
                .latest_modified_ns
                .max(modified_ns(metadata.modified()?));
        }
    }
    Ok(())
}

fn modified_ns(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn run_rsync(config: &SyncConfig) -> Result<()> {
    let args = rsync_args(&config.source_bundle, &config.remote, &config.remote_stage);
    run_command("rsync", &args, config.dry_run)
}

fn switch_remote_current(config: &SyncConfig) -> Result<()> {
    let script = remote_switch_script(&config.remote_stage, &config.remote_current);
    run_command("ssh", &[config.remote.clone(), script], config.dry_run)
}

fn reload_server(config: &SyncConfig) -> Result<()> {
    let args = curl_reload_args(&config.reload_url, config.token.as_deref());
    if config.dry_run {
        let display_args = curl_reload_args(
            &config.reload_url,
            config.token.as_ref().map(|_| "REDACTED"),
        );
        if config.reload_on_remote {
            println!(
                "{}",
                shell_join(
                    "ssh",
                    &[config.remote.clone(), shell_join("curl", &display_args)]
                )
            );
        } else {
            println!("{}", shell_join("curl", &display_args));
        }
        return Ok(());
    }

    if config.reload_on_remote {
        let script = shell_join("curl", &args);
        run_command("ssh", &[config.remote.clone(), script], false)
    } else {
        run_command("curl", &args, false)
    }
}

fn run_command(program: &str, args: &[String], dry_run: bool) -> Result<()> {
    if dry_run {
        println!("{}", shell_join(program, args));
        return Ok(());
    }

    let status = Command::new(program)
        .args(args)
        .status()
        .with_context(|| format!("failed to start {program}"))?;
    if !status.success() {
        anyhow::bail!("{program} failed with status {status}");
    }
    Ok(())
}

fn rsync_args(source_bundle: &Path, remote: &str, remote_stage: &str) -> Vec<String> {
    vec![
        "-a".to_string(),
        "--delete".to_string(),
        "--delay-updates".to_string(),
        ensure_trailing_slash(&source_bundle.to_string_lossy()),
        format!("{remote}:{}", ensure_trailing_slash(remote_stage)),
    ]
}

fn curl_reload_args(reload_url: &str, token: Option<&str>) -> Vec<String> {
    let mut args = vec![
        "-fsS".to_string(),
        "-X".to_string(),
        "POST".to_string(),
        reload_url.to_string(),
    ];
    if let Some(token) = token {
        args.push("-H".to_string());
        args.push(format!("x-ragfs-token: {token}"));
    }
    args
}

fn ensure_trailing_slash(path: &str) -> String {
    if path.ends_with('/') {
        path.to_string()
    } else {
        format!("{path}/")
    }
}

fn remote_switch_script(remote_stage: &str, remote_current: &str) -> String {
    let current = shell_quote(remote_current);
    let current_next = shell_quote(&format!("{remote_current}.next"));
    let stage = shell_quote(remote_stage);
    format!(
        "set -eu; \
         if ([ -e {current} ] || [ -L {current} ]) && [ ! -L {current} ]; then \
           echo 'remote current exists and is not a symlink: {remote_current}' >&2; exit 2; \
         fi; \
         ln -sfn {stage} {current_next}; \
         if mv -Tf {current_next} {current} 2>/dev/null; then exit 0; fi; \
         rm -f {current}; mv -f {current_next} {current}"
    )
}

fn shell_join(program: &str, args: &[String]) -> String {
    std::iter::once(shell_quote(program))
        .chain(args.iter().map(|arg| shell_quote(arg)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | ':' | '@'))
    {
        return value.to_string();
    }

    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_bundle_dir_uses_parent_for_lance_dataset() {
        let path = PathBuf::from("/tmp/ragfs/abc/index.lance");
        assert_eq!(
            index_bundle_dir(&path).unwrap(),
            PathBuf::from("/tmp/ragfs/abc")
        );
    }

    #[test]
    fn rsync_args_sync_bundle_contents_to_remote_stage() {
        let args = rsync_args(
            Path::new("/Users/sophie/Library/Application Support/ragfs/indices/abc"),
            "nas",
            "/volume2/docker/ragfs/index-next",
        );
        assert_eq!(
            args,
            vec![
                "-a",
                "--delete",
                "--delay-updates",
                "/Users/sophie/Library/Application Support/ragfs/indices/abc/",
                "nas:/volume2/docker/ragfs/index-next/",
            ]
        );
    }

    #[test]
    fn remote_switch_refuses_plain_directory_current() {
        let script = remote_switch_script(
            "/volume2/docker/ragfs/index-next",
            "/volume2/docker/ragfs/index",
        );
        assert!(script.contains("[ ! -L /volume2/docker/ragfs/index ]"));
        assert!(script.contains("ln -sfn /volume2/docker/ragfs/index-next"));
        assert!(script.contains("mv -Tf"));
    }

    #[test]
    fn shell_quote_handles_apostrophes() {
        assert_eq!(
            shell_quote("/tmp/HR's red flags"),
            "'/tmp/HR'\"'\"'s red flags'"
        );
    }

    #[test]
    fn curl_reload_uses_token_header_when_present() {
        let args = curl_reload_args("http://127.0.0.1:7777/api/reload", Some("secret"));
        assert_eq!(
            args,
            vec![
                "-fsS",
                "-X",
                "POST",
                "http://127.0.0.1:7777/api/reload",
                "-H",
                "x-ragfs-token: secret",
            ]
        );
    }
}
