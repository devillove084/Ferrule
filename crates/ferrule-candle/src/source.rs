use ferrule_core::{FerruleError, FerruleResult, ModelConfig};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ResolvedModelPaths {
    pub root_dir: PathBuf,
    pub tokenizer_json: PathBuf,
    pub config_json: Option<PathBuf>,
    pub weight_files: Vec<PathBuf>,
}

pub fn resolve_model_paths(cfg: &ModelConfig) -> FerruleResult<ResolvedModelPaths> {
    let root_dir = resolve_root_dir(cfg)?;
    let tokenizer_json = resolve_tokenizer_json(cfg, &root_dir)?;
    let config_json = resolve_optional_config_json(cfg, &root_dir)?;
    let weight_files = resolve_weight_files(cfg, &root_dir)?;

    if weight_files.is_empty() {
        return Err(FerruleError::Config(format!(
            "no .safetensors files found under {}",
            root_dir.display()
        )));
    }

    Ok(ResolvedModelPaths {
        root_dir,
        tokenizer_json,
        config_json,
        weight_files,
    })
}

fn resolve_root_dir(cfg: &ModelConfig) -> FerruleResult<PathBuf> {
    let model_path = PathBuf::from(&cfg.model_id);
    if model_path.exists() && model_path.is_dir() {
        return Ok(model_path);
    }

    if let Some(weights_path) = &cfg.weights_path {
        let p = PathBuf::from(weights_path);
        if p.is_dir() {
            return Ok(p);
        }
        if let Some(parent) = p.parent() {
            return Ok(parent.to_path_buf());
        }
    }

    Err(FerruleError::Config(format!(
        "model_id='{}' is not a local directory, and weights_path did not resolve to a usable local path; materialize the model locally first",
        cfg.model_id
    )))
}

fn resolve_tokenizer_json(cfg: &ModelConfig, root_dir: &Path) -> FerruleResult<PathBuf> {
    if let Some(path) = &cfg.tokenizer_path {
        let p = PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
        return Err(FerruleError::Config(format!(
            "tokenizer_path does not exist: {}",
            p.display()
        )));
    }

    let candidate = root_dir.join("tokenizer.json");
    if candidate.exists() {
        return Ok(candidate);
    }

    Err(FerruleError::Config(format!(
        "tokenizer.json not found under {}",
        root_dir.display()
    )))
}

fn resolve_optional_config_json(
    cfg: &ModelConfig,
    root_dir: &Path,
) -> FerruleResult<Option<PathBuf>> {
    if let Some(path) = &cfg.config_path {
        let p = PathBuf::from(path);
        if p.exists() {
            return Ok(Some(p));
        }
        return Err(FerruleError::Config(format!(
            "config_path does not exist: {}",
            p.display()
        )));
    }

    let candidate = root_dir.join("config.json");
    if candidate.exists() {
        return Ok(Some(candidate));
    }

    Ok(None)
}

fn resolve_weight_files(cfg: &ModelConfig, root_dir: &Path) -> FerruleResult<Vec<PathBuf>> {
    if let Some(path) = &cfg.weights_path {
        let p = PathBuf::from(path);
        if p.is_file() {
            return Ok(vec![p]);
        }
        if p.is_dir() {
            return collect_safetensors(&p);
        }
        return Err(FerruleError::Config(format!(
            "weights_path is neither file nor directory: {}",
            p.display()
        )));
    }

    collect_safetensors(root_dir)
}

fn collect_safetensors(dir: &Path) -> FerruleResult<Vec<PathBuf>> {
    let mut files = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("safetensors"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    files.sort();

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_core::ModelConfig;
    use std::fs;
    use tempfile::tempdir;

    fn cfg(model_id: String) -> ModelConfig {
        ModelConfig {
            backend: "mock".to_string(),
            model_id,
            family: "llama".to_string(),
            device: "cpu".to_string(),
            revision: None,
            tokenizer_path: None,
            weights_path: None,
            config_path: None,
            chat_template: "plain".to_string(),
            dtype: "f32".to_string(),
            use_flash_attn: false,
            use_kv_cache: true,
        }
    }

    #[test]
    fn resolves_local_model_dir_and_sorts_weights() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        fs::write(root.join("tokenizer.json"), "{}").unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();
        fs::write(root.join("model-00002-of-00002.safetensors"), "").unwrap();
        fs::write(root.join("model-00001-of-00002.safetensors"), "").unwrap();

        let resolved = resolve_model_paths(&cfg(root.display().to_string())).unwrap();

        assert_eq!(resolved.weight_files.len(), 2);
        assert!(
            resolved.weight_files[0]
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("00001")
        );
        assert_eq!(resolved.tokenizer_json, root.join("tokenizer.json"));
    }
}
