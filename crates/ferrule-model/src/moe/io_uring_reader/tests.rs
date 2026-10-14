use super::*;

fn temporary_shard(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "ferrule-{name}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::write(&path, vec![0u8; DIRECT_IO_ALIGNMENT]).unwrap();
    path
}

#[test]
#[allow(unsafe_code)]
fn expert_transport_controls_direct_io_file_flag() {
    for (transport, expected) in [
        (ExpertIoTransport::BufferedIoUring, false),
        (ExpertIoTransport::DirectIoUring, true),
    ] {
        let Ok(mut state) = IoUringReadState::new(1, DIRECT_IO_ALIGNMENT, transport) else {
            return;
        };
        let path = temporary_shard(transport.as_str());
        state.register_file(&path).unwrap();
        let flags = unsafe { libc::fcntl(state.files[0].as_raw_fd(), libc::F_GETFL) };
        assert_ne!(flags, -1);
        assert_eq!(flags & libc::O_DIRECT != 0, expected);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
#[allow(unsafe_code)]
fn completion_reactor_is_claimed_once_and_does_not_lose_an_early_wake() {
    let completion_hub = CompletionHub::new();
    let Ok(reader) = IoUringExpertReader::new(
        1,
        DIRECT_IO_ALIGNMENT,
        ExpertIoTransport::DirectIoUring,
        completion_hub.clone(),
    ) else {
        return;
    };
    let reader = Arc::new(reader);
    let reactor = reader
        .take_completion_reactor()
        .expect("the io_uring completion reactor is initially claimable");
    assert!(reader.take_completion_reactor().is_none());

    let listener = completion_hub.listen();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()
        .expect("build current-thread Tokio runtime");
    let local = tokio::task::LocalSet::new();
    runtime.block_on(local.run_until(async move {
        let reactor_task = tokio::task::spawn_local(reactor);
        let value = 1u64;
        let written = unsafe {
            libc::write(
                reader._completion_eventfd.as_raw_fd(),
                (&value as *const u64).cast(),
                std::mem::size_of::<u64>(),
            )
        };
        assert_eq!(written, std::mem::size_of::<u64>() as isize);
        assert!(matches!(
            listener.await,
            ferrule_common::CompletionWake::Progress(_)
        ));
        reactor_task.abort();
    }));
}

#[test]
fn dropping_an_unpolled_completion_reactor_unregisters_eventfd() {
    let completion_hub = CompletionHub::new();
    let Ok(reader) = IoUringExpertReader::new(
        1,
        DIRECT_IO_ALIGNMENT,
        ExpertIoTransport::DirectIoUring,
        completion_hub,
    ) else {
        return;
    };
    let reader = Arc::new(reader);
    let reactor = reader
        .take_completion_reactor()
        .expect("the io_uring completion reactor is initially claimable");

    drop(reactor);

    assert!(
        !reader
            .state
            .lock()
            .expect("io_uring state lock")
            .completion_eventfd_registered
    );
}
