//! gip's Metal GPU backend.
//!
//! [`Metal`] compiles the kernels in `kernels.metal` when it opens and records kernel launches into
//! one command buffer at a time. [`measure_bandwidth`] measures the GPU's memory read bandwidth.
//! The backend exists only on macOS. On other systems the crate is empty.

#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
mod bandwidth;

#[cfg(target_os = "macos")]
pub use backend::{
    Buffer, Element, Error, MatvecOptions, Metal, Norm, ProfileEntry, Store, Ticket, View,
    attention_scratch_floats,
};
#[cfg(target_os = "macos")]
pub use bandwidth::{Bandwidth, BandwidthSample, measure_bandwidth};
