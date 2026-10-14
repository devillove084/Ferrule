//! Private, atomic no-clobber publication for reports and sensitive trace artifacts.
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};

struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub(crate) fn publish(
    path: &Path,
    write: impl FnOnce(&mut std::fs::File) -> Result<()>,
) -> Result<()> {
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    for _ in 0..128 {
        let temp = parent.join(format!(
            ".ferrule-kvset-{}-{}.tmp",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&temp) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).context("create report staging file"),
        };
        let staging = Staging(temp);
        write(&mut file)?;
        file.sync_all().context("sync report staging file")?;
        drop(file);
        // Unlike rename, hard_link atomically refuses to replace an existing destination.
        fs::hard_link(&staging.0, path)
            .context("publish report without overwriting (requires hard-link support)")?;
        return Ok(());
    }
    bail!("unable to create report staging file")
}
