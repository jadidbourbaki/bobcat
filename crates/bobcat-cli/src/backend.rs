//! The model architectures the bobcat command runs, behind one interface.
//!
//! A GGUF file names its architecture. [`Model`] loads LFM2 and Qwen3.5 files, and [`Gpu`] runs
//! either on the Metal GPU with the calls the engine needs.

use std::path::Path;

use bobcat::metal::Metal;
use bobcat::qwen35::{self, Qwen35Metal};
use bobcat::{Lfm2Metal, Sampling};
use tokenizers::Tokenizer;

use crate::Error;
use crate::tokenizer;

/// A model loaded from a GGUF file.
pub(crate) enum Model {
    Lfm2(bobcat::Model),
    Qwen35(qwen35::Model),
}

impl Model {
    /// Load the model in the GGUF file at `path`, whichever architecture the file names.
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        match bobcat::Model::load(path) {
            Ok(model) => Ok(Self::Lfm2(model)),
            Err(bobcat::Error::Architecture(_)) => Ok(Self::Qwen35(qwen35::Model::load(path)?)),
            Err(error) => Err(error.into()),
        }
    }

    /// Return the tokenizer that the model file describes.
    pub(crate) fn tokenizer(&self) -> Result<Tokenizer, Error> {
        match self {
            Self::Lfm2(model) => tokenizer::from_gguf(model.gguf()),
            Self::Qwen35(model) => tokenizer::from_gguf(model.gguf()),
        }
    }

    /// Return the string metadata value `key` of the model file.
    pub(crate) fn metadata_string(&self, key: &str) -> Option<&[u8]> {
        match self {
            Self::Lfm2(model) => model.gguf().string(key),
            Self::Qwen35(model) => model.gguf().string(key),
        }
    }

    /// Return the integer metadata value `key` of the model file.
    pub(crate) fn metadata_u32(&self, key: &str) -> Option<u32> {
        match self {
            Self::Lfm2(model) => model.gguf().u32(key),
            Self::Qwen35(model) => model.gguf().u32(key),
        }
    }

    /// Return the number of tokens in the vocabulary.
    pub(crate) fn n_vocab(&self) -> u32 {
        match self {
            Self::Lfm2(model) => model.hyperparameters().n_vocab,
            Self::Qwen35(model) => model.hyperparameters().n_vocab,
        }
    }

    /// Return the sampling settings the model's file or its authors recommend.
    pub(crate) fn recommended_sampling(&self) -> Sampling {
        match self {
            Self::Lfm2(model) => model.recommended_sampling(),
            Self::Qwen35(model) => model.recommended_sampling(),
        }
    }
}

/// A model's sequence state on the Metal GPU.
pub(crate) enum Gpu<'a> {
    Lfm2(Lfm2Metal<'a>),
    Qwen35(Qwen35Metal<'a>),
}

/// The state of a sequence after some tokens, which [`Gpu::restore`] returns to.
pub(crate) enum Checkpoint {
    Lfm2(bobcat::Checkpoint),
    Qwen35(qwen35::Checkpoint),
}

impl<'a> Gpu<'a> {
    /// Prepare to run sequences of up to `n_ctx` tokens of `model` on `metal`, with a
    /// half-precision KV cache.
    pub(crate) fn new(model: &'a Model, metal: &'a mut Metal, n_ctx: u32) -> Result<Self, Error> {
        Ok(match model {
            Model::Lfm2(model) => Self::Lfm2(Lfm2Metal::new(model, metal, n_ctx, true)?),
            Model::Qwen35(model) => Self::Qwen35(Qwen35Metal::new(model, metal, n_ctx, true)?),
        })
    }

    /// Run the model on `tokens` at the next positions. `logits` receives the last token's
    /// logits when it is given.
    pub(crate) fn prefill(
        &mut self,
        tokens: &[u32],
        logits: Option<&mut [f32]>,
    ) -> Result<(), Error> {
        match self {
            Self::Lfm2(gpu) => gpu.prefill(tokens, logits, None)?,
            Self::Qwen35(gpu) => gpu.prefill(tokens, logits, None, None)?,
        }
        Ok(())
    }

    /// Run the model on `token` at the next position. `logits` receives its logits when it is
    /// given.
    pub(crate) fn step(&mut self, token: u32, logits: Option<&mut [f32]>) -> Result<(), Error> {
        match self {
            Self::Lfm2(gpu) => gpu.step(token, logits, None)?,
            Self::Qwen35(gpu) => gpu.step(token, logits, None)?,
        }
        Ok(())
    }

    /// Decode greedily from the current logits, passing each token to `emit` until it returns
    /// false or `max_tokens` tokens have gone out, and return the tokens the sequence gained.
    pub(crate) fn generate_stream(
        &mut self,
        max_tokens: u32,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, Error> {
        Ok(match self {
            Self::Lfm2(gpu) => gpu.generate_stream(max_tokens, emit)?,
            Self::Qwen35(gpu) => gpu.generate_stream(max_tokens, emit)?,
        })
    }

    /// Forget every token, so the next call starts a new sequence.
    pub(crate) fn reset(&mut self) -> Result<(), Error> {
        match self {
            Self::Lfm2(gpu) => gpu.reset()?,
            Self::Qwen35(gpu) => gpu.reset()?,
        }
        Ok(())
    }

    /// Start a new sequence of `tokens` and write each token's normalized last hidden state to
    /// `hidden`, one row per token, for a decision head.
    pub(crate) fn hidden(&mut self, tokens: &[u32], hidden: &mut [f32]) -> Result<(), Error> {
        match self {
            Self::Lfm2(_) => return Err("LFM2 models hold no decision head".into()),
            Self::Qwen35(gpu) => {
                gpu.reset()?;
                gpu.prefill(tokens, None, Some(hidden), None)?;
            }
        }
        Ok(())
    }

    /// Return the sequence state after every token run so far.
    pub(crate) fn checkpoint(&self) -> Result<Checkpoint, Error> {
        Ok(match self {
            Self::Lfm2(gpu) => Checkpoint::Lfm2(gpu.checkpoint()?),
            Self::Qwen35(gpu) => Checkpoint::Qwen35(gpu.checkpoint()?),
        })
    }

    /// Return to the sequence state of `checkpoint`, which this sequence wrote.
    pub(crate) fn restore(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        match (self, checkpoint) {
            (Self::Lfm2(gpu), Checkpoint::Lfm2(checkpoint)) => gpu.restore(checkpoint)?,
            (Self::Qwen35(gpu), Checkpoint::Qwen35(checkpoint)) => gpu.restore(checkpoint)?,
            (Self::Lfm2(_), Checkpoint::Qwen35(_)) | (Self::Qwen35(_), Checkpoint::Lfm2(_)) => {
                return Err("the checkpoint comes from another model".into());
            }
        }
        Ok(())
    }
}
