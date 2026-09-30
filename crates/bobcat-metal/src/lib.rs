//! bobcat's Metal GPU backend.
//!
//! [`Metal`] compiles the embedded Metal sources when it opens and records kernel launches into
//! one command buffer at a time. [`measure_bandwidth`] measures the GPU's memory read bandwidth.
//! The backend exists only on macOS. On other systems the crate is empty.

#[cfg(target_os = "macos")]
mod backend;

#[cfg(target_os = "macos")]
pub use backend::bandwidth::{Bandwidth, BandwidthSample, measure_bandwidth};
#[cfg(target_os = "macos")]
pub use backend::{
    Buffer, Element, Error, Format, MatvecOptions, Metal, Norm, ProfileEntry, Store, Ticket, View,
    attention_scratch_floats,
};
