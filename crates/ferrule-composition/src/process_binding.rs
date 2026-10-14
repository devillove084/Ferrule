//! Fixed built-in HF binding for hidden process workers. Family selection lives
//! at the composition boundary; parallel receives bindings, never recipes.

use ferrule_common::{Error, Result};
use ferrule_model::models::qwen3::{Qwen3DenseRecipe, Qwen3MoeRecipe};
use ferrule_model::models::qwen35::{Qwen35Adapter, Qwen35Profile};
use ferrule_model::transformer::{BoundDecoderResources, DecoderLoadOptions, DecoderRecipe};
use ferrule_runtime::parallel::expert_process::NumericBoot;
use ferrule_runtime::parallel::process::decoder::{
    DecoderBindings, DecoderBoot, DecoderEndpoint, DecoderRecipeKind,
};
use ferrule_runtime::parallel::process::{ProcessBoot, ProcessHandlerError, ProcessLaunch};

/// Built-in checkpoint selection injected into the generic runtime endpoint.
pub const BUILTIN_PROCESS_BINDINGS: DecoderBindings = DecoderBindings {
    decoder: bind_decoder,
    numeric: bind_numeric,
};

/// Application entrypoint for built-in decoder and expert process workers.
/// Runtime validates boot identity and geometry before invoking the bindings.
pub fn initialize_builtin_process_endpoint(
    boot: ProcessBoot,
    child_launch: ProcessLaunch,
) -> std::result::Result<DecoderEndpoint, ProcessHandlerError> {
    DecoderEndpoint::initialize(boot, child_launch, BUILTIN_PROCESS_BINDINGS)
}

fn error(message: impl std::fmt::Display) -> Error {
    Error::Execution {
        message: format!("process checkpoint binding: {message}"),
    }
}

fn bind_decoder(boot: &DecoderBoot) -> Result<BoundDecoderResources> {
    boot.stage_boot()?;
    if (boot.recipe == DecoderRecipeKind::Qwen35DenseHybrid) != boot.hybrid.is_some() {
        return Err(error("hybrid recipe/schema mismatch"));
    }
    let file = std::fs::File::open(boot.checkpoint.join("config.json"))?;
    if file.metadata()?.len() > 1024 * 1024 {
        return Err(error("checkpoint config exceeds 1 MiB"));
    }
    #[cfg(not(feature = "cpu-reference"))]
    if boot.device == ferrule_runtime::parallel::process::decoder::DecoderDevice::Cpu {
        return Err(error(
            "CPU process execution requires cpu-reference; no fallback",
        ));
    }
    if boot.recipe == DecoderRecipeKind::Qwen35DenseHybrid {
        let (metadata, resources) = Qwen35Adapter::bind_hf_metadata(&boot.checkpoint)?;
        if metadata.config().profile() != Qwen35Profile::Dense08Bbf16 {
            return Err(error("process hybrid requires strict dense Qwen3.5-0.8B"));
        }
        return Ok(resources);
    }
    let config: serde_json::Value = serde_json::from_reader(file).map_err(error)?;
    let (recipe, family): (Box<dyn DecoderRecipe>, _) = match boot.recipe {
        DecoderRecipeKind::Qwen3Dense => (
            Box::new(Qwen3DenseRecipe::new()),
            ferrule_model::ModelFamily::Qwen3,
        ),
        DecoderRecipeKind::Qwen3Moe => (
            Box::new(Qwen3MoeRecipe::new()),
            ferrule_model::ModelFamily::QwenMoe,
        ),
        DecoderRecipeKind::Qwen35DenseHybrid => unreachable!("strict hybrid binding above"),
        DecoderRecipeKind::Synthetic => {
            #[cfg(feature = "cpu-reference")]
            {
                (
                    Box::new(ferrule_model::transformer::SyntheticDecoderRecipe::new()),
                    ferrule_model::ModelFamily::Unknown("synthetic".into()),
                )
            }
            #[cfg(not(feature = "cpu-reference"))]
            {
                return Err(error("synthetic process binding requires cpu-reference"));
            }
        }
    };
    DecoderLoadOptions::new(recipe.as_ref(), &config).open_hf(&boot.checkpoint, family)
}

fn bind_numeric(boot: &NumericBoot) -> Result<BoundDecoderResources> {
    let (metadata, resources) = Qwen35Adapter::bind_hf_metadata(&boot.checkpoint)?;
    if metadata.config().profile() != Qwen35Profile::Moe35BA3Bfp8 {
        return Err(error("numeric process requires strict Qwen3.5-35B-A3B FP8"));
    }
    Ok(resources)
}
