//! Conservative admission, using MemAvailable and every visible cgroup ancestor.
use super::{HostExpertCacheError, HostMemorySnapshot, error};
use ferrule_common::Result;
use std::path::{Path, PathBuf};

fn probe(reason: impl Into<String>) -> ferrule_common::Error {
    error(HostExpertCacheError::MemoryProbe {
        reason: reason.into(),
    })
}
fn read(path: impl AsRef<Path>) -> Result<String> {
    std::fs::read_to_string(path.as_ref())
        .map_err(|e| probe(format!("{}: {e}", path.as_ref().display())))
}
fn number(value: &str) -> Result<u64> {
    value
        .trim()
        .parse()
        .map_err(|_| probe("invalid memory counter"))
}
fn mem_available(text: &str) -> Result<u64> {
    proc_kib(text, "MemAvailable:")
}
fn proc_kib(text: &str, key: &str) -> Result<u64> {
    let value = text
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .ok_or_else(|| probe(format!("{key} is missing")))?;
    let fields = value.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 2 || fields[1] != "kB" {
        return Err(probe("invalid MemAvailable units"));
    }
    number(fields[0])?
        .checked_mul(1024)
        .ok_or_else(|| probe("MemAvailable overflow"))
}
fn unescape_mount(value: &str) -> String {
    value
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

/// Resolve the current process relative to the cgroup mount's root. This also
/// handles containers exposing only their subtree, not the host's / hierarchy.
fn memberships(cgroups: &str, mounts: &str) -> Result<Vec<(PathBuf, PathBuf, bool)>> {
    let mut result = Vec::new();
    for group in cgroups.lines() {
        let fields = group.splitn(3, ':').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(probe("malformed /proc/self/cgroup"));
        }
        let v2 = fields[0] == "0" && fields[1].is_empty();
        if !v2 && !fields[1].split(',').any(|c| c == "memory") {
            continue;
        }
        let membership = Path::new(fields[2]);
        let mut found = false;
        for mount in mounts.lines() {
            let Some((left, right)) = mount.split_once(" - ") else {
                continue;
            };
            let left = left.split_whitespace().collect::<Vec<_>>();
            let right = right.split_whitespace().collect::<Vec<_>>();
            if left.len() < 5 || right.len() < 3 {
                continue;
            }
            if (v2 && right[0] != "cgroup2")
                || (!v2 && (right[0] != "cgroup" || !right[2].split(',').any(|c| c == "memory")))
            {
                continue;
            }
            let root = PathBuf::from(unescape_mount(left[3]));
            let mountpoint = PathBuf::from(unescape_mount(left[4]));
            let relative = membership
                .strip_prefix(&root)
                .or_else(|_| {
                    // A cgroup namespace reports '/' for its root although mountinfo
                    // may still expose a host-rooted subtree.
                    if membership == Path::new("/") {
                        Ok(Path::new(""))
                    } else {
                        Err(())
                    }
                })
                .map_err(|_| probe("cgroup membership is outside visible mount root"))?;
            result.push((mountpoint.join(relative), mountpoint, v2));
            found = true;
            break;
        }
        if !found {
            return Err(probe("memory cgroup has no visible mount"));
        }
    }
    Ok(result)
}

pub(super) fn snapshot() -> Result<HostMemorySnapshot> {
    if !cfg!(target_os = "linux") {
        return Err(probe(
            "full prewarm memory admission currently requires Linux",
        ));
    }
    let snapshot = HostMemorySnapshot {
        system_available_bytes: mem_available(&read("/proc/meminfo")?)?,
        process_rss_bytes: proc_kib(&read("/proc/self/status")?, "VmRSS:")?,
        cgroup_limit_bytes: None,
        cgroup_current_bytes: None,
    };
    let groups = memberships(&read("/proc/self/cgroup")?, &read("/proc/self/mountinfo")?)?;
    constrain_cgroups(snapshot, groups)
}

fn constrain_cgroups(
    mut snapshot: HostMemorySnapshot,
    groups: Vec<(PathBuf, PathBuf, bool)>,
) -> Result<HostMemorySnapshot> {
    let mut tightest = u64::MAX;
    let mut apply_limit = |limit: u64, usage: u64| {
        let remaining = limit.saturating_sub(usage);
        if remaining < tightest {
            tightest = remaining;
            snapshot.cgroup_limit_bytes = Some(limit);
            snapshot.cgroup_current_bytes = Some(usage);
        }
    };
    for (mut current, root, v2) in groups {
        loop {
            let limit_path = current.join(if v2 {
                "memory.max"
            } else {
                "memory.limit_in_bytes"
            });
            // The v2 hierarchy root has no memory controller cap of its own.
            if !(v2 && current == root && !limit_path.exists()) {
                let limit = read(&limit_path)?;
                let usage = number(&read(current.join(if v2 {
                    "memory.current"
                } else {
                    "memory.usage_in_bytes"
                }))?)?;
                if limit.trim() != "max" {
                    apply_limit(number(&limit)?, usage);
                }
                if v2 {
                    let high = read(current.join("memory.high"))?;
                    if high.trim() != "max" {
                        apply_limit(number(&high)?, usage);
                    }
                } else {
                    // v1 can expose a tighter ancestor cap even when a container
                    // mount hides that ancestor's directory.
                    let stats = read(current.join("memory.stat"))?;
                    if let Some(limit) = stats
                        .lines()
                        .find_map(|line| line.strip_prefix("hierarchical_memory_limit "))
                    {
                        apply_limit(number(limit)?, usage);
                    }
                }
            }
            if current == root {
                break;
            }
            if !current.pop() || !current.starts_with(&root) {
                return Err(probe("invalid cgroup ancestry"));
            }
        }
    }
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ancestor_usage_high_limit_and_rss_are_not_double_subtracted() {
        let root = std::env::temp_dir().join(format!("host-memory-cgroups-{}", std::process::id()));
        let child = root.join("child");
        std::fs::create_dir_all(&child).unwrap();
        for (path, limit, used, high) in [
            (&root, "48000", "8000", "42000"),
            (&child, "40000", "1000", "max"),
        ] {
            for (file, value) in [
                ("memory.max", limit),
                ("memory.current", used),
                ("memory.high", high),
            ] {
                std::fs::write(path.join(file), value).unwrap();
            }
        }
        let baseline = HostMemorySnapshot {
            system_available_bytes: 100_000,
            process_rss_bytes: 700,
            cgroup_limit_bytes: None,
            cgroup_current_bytes: None,
        };
        let result = constrain_cgroups(baseline, vec![(child, root.clone(), true)]).unwrap();
        assert_eq!(result.cgroup_limit_bytes, Some(42000));
        assert_eq!(result.cgroup_current_bytes, Some(8000));
        assert_eq!(result.process_rss_bytes, 700);
        assert_eq!(result.available_bytes(), 34000);
        std::fs::write(root.join("memory.limit_in_bytes"), "48000").unwrap();
        std::fs::write(root.join("memory.usage_in_bytes"), "8000").unwrap();
        std::fs::write(
            root.join("memory.stat"),
            "hierarchical_memory_limit 32000\n",
        )
        .unwrap();
        let result =
            constrain_cgroups(baseline, vec![(root.clone(), root.clone(), false)]).unwrap();
        assert_eq!(result.available_bytes(), 24000);
        assert_eq!(proc_kib("VmRSS: 512 kB\n", "VmRSS:").unwrap(), 512 * 1024);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn available_units_and_container_hierarchy_resolution() {
        assert_eq!(
            mem_available("MemTotal: 900 kB\nMemAvailable: 123 kB\n").unwrap(),
            123 * 1024
        );
        assert!(mem_available("MemFree: 999 kB").is_err());
        assert!(mem_available("MemAvailable: 1 GB").is_err());
        let groups = memberships(
            "0::/a/b\n",
            "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        )
        .unwrap();
        assert_eq!(
            groups,
            [("/sys/fs/cgroup/a/b".into(), "/sys/fs/cgroup".into(), true)]
        );
        let groups = memberships(
            "4:memory:/docker/id\n",
            "1 0 0:1 /docker/id /sys/fs/cgroup/memory ro - cgroup cgroup rw,memory\n",
        )
        .unwrap();
        assert_eq!(
            groups,
            [(
                "/sys/fs/cgroup/memory/".into(),
                "/sys/fs/cgroup/memory".into(),
                false
            )]
        );
        assert!(memberships("4:memory:/docker/id", "").is_err());
    }
}
