use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, pem::PemObject};
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use serde::{Deserialize, Serialize};

use super::NetworkError;

const MAX_PEM_BYTES: u64 = 1024 * 1024;
const ALPN: &[u8] = b"ferrule-linear-rank/1";

/// Paths refer to files on this host, never transported or logged as credentials.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TlsFiles {
    pub ca: PathBuf,
    pub certificate_chain: PathBuf,
    pub private_key: PathBuf,
    /// Exactly one leaf, not a CA or chain. Chain validation is still mandatory.
    pub peer_leaf: PathBuf,
}
#[derive(Clone)]
pub struct ClientTls {
    pub(super) config: Arc<ClientConfig>,
    pub(super) name: ServerName<'static>,
    pub(super) pin: CertificateDer<'static>,
}
#[derive(Clone)]
pub struct ServerTls {
    pub(super) config: Arc<ServerConfig>,
    pub(super) pin: CertificateDer<'static>,
}

impl ClientTls {
    pub fn load(files: &TlsFiles, server_name: &str) -> Result<Self, NetworkError> {
        let roots = roots(&files.ca)?;
        let certs = certificates(&files.certificate_chain)?;
        let key = private_key(&files.private_key)?;
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut config = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|_| NetworkError::Authentication)?
            .with_root_certificates(roots)
            .with_client_auth_cert(certs, key)
            .map_err(|_| NetworkError::Authentication)?;
        config.enable_early_data = false;
        config.resumption = rustls::client::Resumption::disabled();
        config.alpn_protocols = vec![ALPN.to_vec()];
        Ok(Self {
            config: Arc::new(config),
            name: ServerName::try_from(server_name.to_owned())
                .map_err(|_| NetworkError::Configuration("invalid TLS server name"))?,
            pin: pin(&files.peer_leaf)?,
        })
    }
}
impl ServerTls {
    pub fn load(files: &TlsFiles) -> Result<Self, NetworkError> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            Arc::new(roots(&files.ca)?),
            provider.clone(),
        )
        .build()
        .map_err(|_| NetworkError::Authentication)?;
        let mut config = ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|_| NetworkError::Authentication)?
            .with_client_cert_verifier(verifier)
            .with_single_cert(
                certificates(&files.certificate_chain)?,
                private_key(&files.private_key)?,
            )
            .map_err(|_| NetworkError::Authentication)?;
        config.max_early_data_size = 0;
        config.send_tls13_tickets = 0;
        config.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});
        config.alpn_protocols = vec![ALPN.to_vec()];
        Ok(Self {
            config: Arc::new(config),
            pin: pin(&files.peer_leaf)?,
        })
    }
}

fn read_file(path: &Path, private: bool) -> Result<Vec<u8>, NetworkError> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    let file: File = options
        .open(path)
        .map_err(|_| NetworkError::Authentication)?;
    let metadata = file.metadata().map_err(|_| NetworkError::Authentication)?;
    // Inspect the opened descriptor, not a racy pre-open path check. Refuse
    // symlinks, nonregular files, oversized inputs and group/world key access.
    #[allow(unsafe_code)]
    let uid = unsafe { libc::geteuid() };
    if !metadata.is_file()
        || metadata.len() > MAX_PEM_BYTES
        || (private && (metadata.mode() & 0o077 != 0 || metadata.uid() != uid))
    {
        return Err(NetworkError::Authentication);
    }
    let mut bytes = Vec::new();
    file.take(MAX_PEM_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| NetworkError::Authentication)?;
    if bytes.len() as u64 > MAX_PEM_BYTES {
        return Err(NetworkError::Authentication);
    }
    Ok(bytes)
}
pub(super) fn certificates(path: &Path) -> Result<Vec<CertificateDer<'static>>, NetworkError> {
    let bytes = read_file(path, false)?;
    let certs = CertificateDer::pem_slice_iter(&bytes)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| NetworkError::Authentication)?;
    if certs.is_empty() {
        return Err(NetworkError::Authentication);
    }
    Ok(certs)
}
fn roots(path: &Path) -> Result<RootCertStore, NetworkError> {
    let mut roots = RootCertStore::empty();
    for cert in certificates(path)? {
        roots.add(cert).map_err(|_| NetworkError::Authentication)?;
    }
    Ok(roots)
}
fn pin(path: &Path) -> Result<CertificateDer<'static>, NetworkError> {
    let mut certs = certificates(path)?;
    if certs.len() != 1 {
        return Err(NetworkError::Configuration(
            "peer pin must contain exactly one leaf",
        ));
    }
    Ok(certs.remove(0))
}
pub(super) fn private_key(path: &Path) -> Result<PrivateKeyDer<'static>, NetworkError> {
    let mut bytes = read_file(path, true)?;
    let key = PrivateKeyDer::from_pem_slice(&bytes).map_err(|_| NetworkError::Authentication);
    bytes.fill(0);
    key
}
pub(super) fn verify_peer(
    connection: &rustls::Connection,
    expected: &CertificateDer<'_>,
) -> Result<(), NetworkError> {
    if connection.alpn_protocol() != Some(ALPN)
        || connection
            .peer_certificates()
            .and_then(|chain| chain.first())
            != Some(expected)
    {
        return Err(NetworkError::Authentication);
    }
    Ok(())
}
