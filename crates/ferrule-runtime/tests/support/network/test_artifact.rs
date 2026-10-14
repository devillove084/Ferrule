//! TEST ONLY build identity. Content, not lockfile mtime: concurrent Cargo may
//! rewrite an unchanged lockfile without needing to relink the child.
pub fn fingerprint() -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for source in [
        include_str!("test_child.rs"),
        include_str!("../../../Cargo.toml"),
        include_str!("../../../../../Cargo.lock"),
        include_str!("../../../src/parallel/network.rs"),
        include_str!("../../../src/parallel/network/tls.rs"),
        include_str!("../../../src/parallel/network/transport.rs"),
        include_str!("../../../src/parallel/process.rs"),
        include_str!("../../../src/parallel/process/owner.rs"),
        include_str!("../../../src/parallel/process/child.rs"),
        include_str!("../../../src/parallel/process/ipc.rs"),
        include_str!("../../../src/parallel/process/reaper.rs"),
    ] {
        source.hash(&mut h);
    }
    env!("CARGO_MANIFEST_DIR").hash(&mut h);
    cfg!(feature = "cuda").hash(&mut h);
    format!("TEST-ONLY-network-child-v1-{:016x}", h.finish())
}
