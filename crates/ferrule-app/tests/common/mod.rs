use ahash::AHashMap;

use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

pub struct MockFixture {
    _tmp: TempDir,
    pub model_dir: PathBuf,
    pub config_path: PathBuf,
}

pub fn make_mock_fixture() -> MockFixture {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    let model_dir = root.join("model");
    fs::create_dir_all(&model_dir).unwrap();

    write_test_tokenizer(&model_dir.join("tokenizer.json"));
    fs::write(model_dir.join("model.safetensors"), "").unwrap();

    let config_path = root.join("rollout.toml");
    fs::write(
        &config_path,
        format!(
            r#"[observability]
service_name = "ferrule-test"
log_level = "info"
log_format = "json"
metrics_enabled = false
metrics_bind = "127.0.0.1:0"

[model]
backend = "mock"
model_id = "{}"
family = "llama"
device = "auto"
chat_template = "plain"
dtype = "f32"
use_flash_attn = false
use_kv_cache = true

[rollout]
max_steps = 8
seed = 42
"#,
            model_dir.display()
        ),
    )
    .unwrap();

    MockFixture {
        _tmp: tmp,
        model_dir,
        config_path,
    }
}

fn write_test_tokenizer(path: &Path) {
    let vocab = AHashMap::from([
        ("[UNK]".to_string(), 0u32),
        ("Hello".to_string(), 1u32),
        ("Ferrule".to_string(), 2u32),
        (".".to_string(), 3u32),
        ("</s>".to_string(), 4u32),
    ]);

    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();

    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.save(path, false).unwrap();
}
