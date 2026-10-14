//! Authenticated fault injection only; none of these hooks enter production.
use super::support::*;
use super::*;
use serde_json::json;
use std::sync::{Arc, mpsc};

fn raw_client(
    creds: &Credentials,
    address: SocketAddr,
    b: NetworkBinding,
    cfg: Value,
    opts: NetworkOptions,
) -> Channel {
    let end = Instant::now() + Duration::from_secs(3);
    let mut c = Channel::client(
        TcpStream::connect(address).unwrap(),
        &creds.client_tls(),
        end,
    )
    .unwrap();
    c.send(
        &Envelope::encode(
            Message::Hello {
                binding: b,
                limits: opts.limits,
                command_timeout_ms: opts.command_timeout.as_millis() as u64,
                config: cfg,
            },
            opts.limits.max_frame_bytes,
        )
        .unwrap(),
        end,
    )
    .unwrap();
    assert!(matches!(
        c.receive(opts.limits.max_frame_bytes, end).unwrap(),
        Message::Ready { .. }
    ));
    c
}
#[test]
fn pem_chain_order_permissions_symlink_and_redacted_errors() {
    use std::os::unix::fs::PermissionsExt;
    let creds = Credentials::new();
    let chain = creds.dir.path().join("chain.pem");
    std::fs::write(
        &chain,
        format!(
            "{}{}",
            std::fs::read_to_string(&creds.server.certificate_chain).unwrap(),
            std::fs::read_to_string(&creds.server.ca).unwrap()
        ),
    )
    .unwrap();
    let certs = tls::certificates(&chain).unwrap();
    assert_eq!(certs.len(), 2);
    assert_eq!(
        certs[0],
        tls::certificates(&creds.server.certificate_chain).unwrap()[0]
    );
    assert_eq!(certs[1], tls::certificates(&creds.server.ca).unwrap()[0]);
    let mut files = creds.client.clone();
    files.peer_leaf = chain;
    assert!(ClientTls::load(&files, "localhost").is_err());
    std::fs::set_permissions(&files.private_key, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(tls::private_key(&files.private_key).is_err());
    std::fs::set_permissions(&files.private_key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let link = creds.dir.path().join("link.key");
    std::os::unix::fs::symlink(&files.private_key, &link).unwrap();
    assert!(tls::private_key(&link).is_err());
    std::fs::write(&files.private_key, "SUPER_SECRET_invalid_PEM").unwrap();
    let e = tls::private_key(&files.private_key).unwrap_err();
    assert!(!format!("{e:?} {e}").contains("SUPER_SECRET"));
    assert!(!format!("{e:?} {e}").contains(creds.dir.path().to_str().unwrap()));
}
#[test]
fn builtin_pem_supports_pkcs1_sec1_pkcs8_labels() {
    use rustls::pki_types::PrivateKeyDer;
    let creds = Credentials::new();
    for label in ["RSA PRIVATE KEY", "EC PRIVATE KEY", "PRIVATE KEY"] {
        // TEST ONLY parser selection bytes, not usable authentication material.
        std::fs::write(
            &creds.client.private_key,
            format!("-----BEGIN {label}-----\nAQID\n-----END {label}-----\n"),
        )
        .unwrap();
        match (label, tls::private_key(&creds.client.private_key).unwrap()) {
            ("RSA PRIVATE KEY", PrivateKeyDer::Pkcs1(_))
            | ("EC PRIVATE KEY", PrivateKeyDer::Sec1(_))
            | ("PRIVATE KEY", PrivateKeyDer::Pkcs8(_)) => {}
            _ => panic!("wrong key format"),
        }
    }
}
#[test]
fn missing_client_certificate_is_rejected_before_spawn() {
    let creds = Credentials::new();
    let b = binding(0, 1, LinearMode::Dp);
    let mut cfg = config(0, 1, LinearMode::Dp);
    let started = creds.dir.path().join("started");
    cfg["started"] = json!(started);
    let r = Running::start(endpoint(&creds, b, cfg, options(400)));
    let mut roots = rustls::RootCertStore::empty();
    for cert in tls::certificates(&creds.client.ca).unwrap() {
        roots.add(cert).unwrap();
    }
    let mut cfg = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .unwrap()
    .with_root_certificates(roots)
    .with_no_client_auth();
    cfg.alpn_protocols = vec![b"ferrule-linear-rank/1".to_vec()];
    let mut client = creds.client_tls();
    client.config = Arc::new(cfg);
    let end = Instant::now() + Duration::from_secs(2);
    if let Ok(mut c) = Channel::client(TcpStream::connect(r.address).unwrap(), &client, end) {
        assert!(c.receive(1024, end).is_err());
    }
    assert!(r.finish().is_err());
    assert!(!started.exists());
}
#[test]
fn zero_oversize_truncated_and_slow_trickle_frames_are_bounded() {
    for case in ["zero", "oversize", "truncated", "trickle", "json"] {
        let creds = Credentials::new();
        let b = binding(0, 1, LinearMode::Dp);
        let cfg = config(0, 1, LinearMode::Dp);
        let opts = options(180);
        let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), opts));
        let mut c = raw_client(&creds, r.address, b, cfg, opts);
        let start = Instant::now();
        let end = start + Duration::from_secs(2);
        match case {
            "zero" => c.write_plain(&0u32.to_be_bytes(), end).unwrap(),
            "oversize" => c
                .write_plain(&(opts.limits.max_frame_bytes as u32 + 1).to_be_bytes(), end)
                .unwrap(),
            "truncated" => {
                c.write_plain(&100u32.to_be_bytes(), end).unwrap();
                c.write_plain(b"{", end).unwrap();
                c.socket.shutdown(std::net::Shutdown::Write).unwrap();
            }
            "trickle" => {
                for byte in [0, 0, 0, 100] {
                    if c.write_plain(&[byte], end).is_err() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(70));
                }
            }
            "json" => c.send(b"{invalid", end).unwrap(),
            _ => unreachable!(),
        }
        assert!(r.finish().is_err());
        assert!(start.elapsed() < Duration::from_secs(2), "{case}");
    }
}
#[test]
fn duplicate_execute_is_not_replayed_and_idle_does_not_timeout() {
    let creds = Credentials::new();
    let b = binding(0, 1, LinearMode::Dp);
    let mut cfg = config(0, 1, LinearMode::Dp);
    let opts = options(300);
    let ledger = creds.dir.path().join("ledger");
    cfg["ledger"] = json!(ledger);
    let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), opts));
    let mut c = raw_client(&creds, r.address, b.clone(), cfg, opts);
    std::thread::sleep(Duration::from_millis(450));
    let key = ProcessCommandIdentity {
        owner: b.identity,
        command: ProcessCommandId::new(1).unwrap(),
        transaction: tx(1).into(),
    };
    let frame = Envelope::encode(
        Message::Execute {
            identity: key,
            session: 5,
            payload: input(),
        },
        opts.limits.max_frame_bytes,
    )
    .unwrap();
    let end = Instant::now() + Duration::from_secs(2);
    c.send(&frame, end).unwrap();
    assert!(matches!(
        c.receive(opts.limits.max_frame_bytes, end).unwrap(),
        Message::Complete { .. }
    ));
    c.send(&frame, end).unwrap();
    assert!(c.receive(opts.limits.max_frame_bytes, end).is_err());
    assert!(r.finish().is_err());
    assert_eq!(std::fs::read_to_string(ledger).unwrap().lines().count(), 1);
}

// A real ProcessRankOwner executes first. The fault peer then corrupts or drops
// the reply; it never fabricates successful computation or fence evidence.
#[test]
fn executed_but_reply_lost_and_wrong_correlation_are_permanent_unknown() {
    for fault in [
        "lost",
        "command",
        "session",
        "transaction",
        "owner",
        "epoch",
        "rank",
        "fenced_wrong_session",
        "unknown_failure",
        "oversize_failure",
    ] {
        let creds = Credentials::new();
        let b = binding(0, 1, LinearMode::Dp);
        let mut cfg = config(0, 1, LinearMode::Dp);
        let opts = options(600);
        let ledger = creds.dir.path().join("ledger");
        cfg["ledger"] = json!(ledger);
        let endpoint = endpoint(&creds, b.clone(), cfg.clone(), opts);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (done_tx, done_rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let end = Instant::now() + Duration::from_secs(3);
            listener.set_nonblocking(true).unwrap();
            transport::wait(&listener, libc::POLLIN, Some(end)).unwrap();
            let mut c = Channel::server(listener.accept().unwrap().0, &endpoint.tls, end).unwrap();
            assert!(matches!(
                c.receive(opts.limits.max_frame_bytes, end).unwrap(),
                Message::Hello { .. }
            ));
            let mut child = ProcessRankOwner::<Value, Value, Value>::spawn(
                endpoint.launch,
                endpoint.local_identity,
                &endpoint.config,
                endpoint.local_options,
            )
            .unwrap();
            c.send(
                &Envelope::encode(
                    Message::Ready {
                        binding: endpoint.binding.clone(),
                        limits: opts.limits,
                    },
                    opts.limits.max_frame_bytes,
                )
                .unwrap(),
                end,
            )
            .unwrap();
            let Message::Execute {
                mut identity,
                mut session,
                payload,
            } = c.receive(opts.limits.max_frame_bytes, end).unwrap()
            else {
                panic!("expected execute")
            };
            let output = child
                .execute(identity.transaction.into(), session, &payload)
                .unwrap();
            let pid = output["pid"].as_u64().unwrap() as u32;
            if fault != "lost" {
                match fault {
                    "command" => identity.command = ProcessCommandId::new(2).unwrap(),
                    "session" => session += 1,
                    "transaction" => identity.transaction = tx(98).into(),
                    "epoch" => {
                        identity.owner.epoch =
                            runtime::parallel::process::ProcessGroupEpoch::new(2).unwrap()
                    }
                    "owner" => {
                        identity.owner.owner_instance =
                            runtime::parallel::process::ProcessOwnerInstanceId::new(2).unwrap()
                    }
                    "rank" => identity.owner.rank = ferrule_common::ParallelRankId::new(1),
                    _ => {}
                }
                let reply = if fault.contains("failure") || fault == "fenced_wrong_session" {
                    Message::Failure {
                        owner: identity.owner,
                        command: Some(identity),
                        session: Some(if fault == "fenced_wrong_session" {
                            session + 1
                        } else {
                            session
                        }),
                        failure: ProcessHandlerError {
                            kind: runtime::parallel::process::ProcessFailureKind::Handler,
                            quiescence: if fault == "unknown_failure" {
                                ProcessQuiescence::Unknown
                            } else {
                                ProcessQuiescence::Fenced
                            },
                            message: if fault == "oversize_failure" {
                                "x".repeat(opts.limits.max_error_bytes + 1)
                            } else {
                                "TEST ONLY fault".into()
                            },
                        },
                    }
                } else {
                    Message::Complete {
                        identity,
                        session,
                        payload: output,
                    }
                };
                c.send(
                    &Envelope::encode(reply, opts.limits.max_frame_bytes).unwrap(),
                    end,
                )
                .unwrap();
            }
            drop(c);
            child.shutdown().unwrap();
            done_tx.send(pid).unwrap();
        });
        let mut owner = Owner::connect(address, &creds.client_tls(), b, &cfg, opts).unwrap();
        assert!(
            matches!(
                owner.execute(tx(1), 3, &input()),
                Err(NetworkError::Unknown(_))
            ),
            "{fault}"
        );
        assert!(matches!(
            owner.execute(tx(2), 3, &input()),
            Err(NetworkError::Unavailable)
        ));
        let pid = done_rx.recv_timeout(Duration::from_secs(4)).unwrap();
        handle.join().unwrap();
        absent(pid);
        assert_eq!(std::fs::read_to_string(ledger).unwrap().lines().count(), 1);
    }
}
#[test]
fn tls_backpressure_obeys_original_deadline() {
    let creds = Credentials::new();
    let server = creds.server_tls();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        listener.set_nonblocking(true).unwrap();
        let end = Instant::now() + Duration::from_secs(3);
        transport::wait(&listener, libc::POLLIN, Some(end)).unwrap();
        let c = Channel::server(listener.accept().unwrap().0, &server, end).unwrap();
        ready_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        drop(c);
    });
    let mut c = Channel::client(
        TcpStream::connect(address).unwrap(),
        &creds.client_tls(),
        Instant::now() + Duration::from_secs(3),
    )
    .unwrap();
    ready_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    let bytes = vec![0; 16 * 1024 * 1024];
    let start = Instant::now();
    assert!(matches!(
        c.write_plain(&bytes, start + Duration::from_millis(150)),
        Err(NetworkError::Deadline)
    ));
    assert!(start.elapsed() < Duration::from_secs(1));
    release_tx.send(()).unwrap();
    thread.join().unwrap();
}
