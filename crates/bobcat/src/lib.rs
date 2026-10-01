//! bobcat is a local LLM inference engine.
//!
//! [`Model`] loads an LFM2 model from a GGUF file. [`State`] runs the model on the CPU with the
//! scalar reference ops in [`scalar`], which define the correct output of every op. On macOS,
//! [`Lfm2Metal`] runs the same model on the GPU. [`qwen35`] holds the same pieces for Qwen3.5
//! models.

pub mod clef;
mod error;
mod lfm2;
#[cfg(target_os = "macos")]
mod lfm2_metal;
pub mod qwen35;
#[cfg(target_os = "macos")]
mod qwen35_metal;
pub mod scalar;
mod storage;

pub use bobcat_gguf::TensorType;
#[cfg(target_os = "macos")]
pub use bobcat_metal as metal;
pub use error::Error;
pub use lfm2::{Hyperparameters, Model, Sampling, State, Trace};
#[cfg(target_os = "macos")]
pub use lfm2_metal::{Checkpoint, Lfm2Metal};
