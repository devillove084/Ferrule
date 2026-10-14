#[cfg(unix)]
pub(crate) fn overwrite_restoring_mtime(path: &std::path::Path) {
    use std::fs::{File, FileTimes};
    use std::io::{Seek, SeekFrom, Write};
    use std::os::unix::fs::MetadataExt;
    use std::time::{Duration, Instant};
    let before = std::fs::metadata(path).unwrap();
    let modified = before.modified().unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    bytes[0] ^= 1;
    let mut file = File::options().write(true).open(path).unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        std::thread::sleep(Duration::from_millis(2));
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_times(FileTimes::new().set_modified(modified))
            .unwrap();
        let after = file.metadata().unwrap();
        assert_eq!(before.ino(), after.ino());
        assert_eq!(before.dev(), after.dev());
        assert_eq!(before.len(), after.len());
        assert_eq!(before.mtime(), after.mtime());
        assert_eq!(before.mtime_nsec(), after.mtime_nsec());
        if (before.ctime(), before.ctime_nsec()) != (after.ctime(), after.ctime_nsec()) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "filesystem did not expose a ctime change within the finite test deadline"
        );
    }
    assert_ne!(std::fs::read(path).unwrap(), {
        bytes[0] ^= 1;
        bytes
    });
}
