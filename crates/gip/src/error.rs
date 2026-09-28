//! The error type of the gip crate.

use std::io;
use std::path::PathBuf;

use gip_gguf::TensorType;

/// Why loading or running a model failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The model file could not be read.
    #[error("{}: {source}", path.display())]
    Io {
        /// The model file.
        path: PathBuf,
        /// The system's error.
        source: io::Error,
    },
    /// The model file is no well-formed GGUF file.
    #[error("{}: {source}", path.display())]
    Gguf {
        /// The model file.
        path: PathBuf,
        /// The parser's error.
        source: gip_gguf::Error,
    },
    /// A metadata entry the model needs is missing or has the wrong type.
    #[error("missing or invalid {0}")]
    Metadata(&'static str),
    /// The file holds a model of another architecture.
    #[error("architecture {0} is not lfm2")]
    Architecture(String),
    /// The hyperparameters contradict each other.
    #[error("inconsistent hyperparameters")]
    Hyperparameters,
    /// The attention heads have an odd number of elements, which rotary embeddings cannot pair.
    #[error("head size {0} is odd")]
    OddHeadDim(u32),
    /// A tensor the model needs is missing.
    #[error("missing tensor {0}")]
    MissingTensor(String),
    /// A tensor has the wrong shape.
    #[error("tensor {name} has shape {got:?}, expected {want:?}")]
    TensorShape {
        /// The tensor's name.
        name: String,
        /// The shape in the file.
        got: Vec<u64>,
        /// The shape the hyperparameters imply.
        want: Vec<u64>,
    },
    /// A tensor has a type this code path does not read.
    #[error("tensor {name} has unsupported type {data_type:?}")]
    TensorType {
        /// The tensor's name.
        name: String,
        /// The tensor's type.
        data_type: TensorType,
    },
    /// An attention layer has a KV head count other than the model's.
    #[error("layer {layer} has {n_kv_heads} KV heads")]
    KvHeads {
        /// The layer index.
        layer: u32,
        /// The layer's KV head count.
        n_kv_heads: u32,
    },
    /// A token id lies outside the vocabulary.
    #[error("token {token} lies outside the vocabulary of {n_vocab} tokens")]
    Token {
        /// The token id.
        token: u32,
        /// The vocabulary size.
        n_vocab: u32,
    },
    /// A call asks for more positions than the context has left.
    #[error("{requested} more tokens exceed the {remaining} positions left in the context")]
    ContextFull {
        /// The tokens the call would add.
        requested: u32,
        /// The positions left.
        remaining: u32,
    },
    /// An argument has the wrong size or value.
    #[error("invalid argument: {0}")]
    Argument(&'static str),
    /// The Metal path cannot run the model.
    #[error("the Metal path needs {0}")]
    MetalUnsupported(String),
    /// The Metal backend failed.
    #[cfg(target_os = "macos")]
    #[error(transparent)]
    Metal(#[from] gip_metal::Error),
}
