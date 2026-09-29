//! The bytes of a model file, memory-mapped when the system allows it.

#![expect(unsafe_code, reason = "memmap2 maps model files")]

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

/// The bytes of a model file.
#[derive(Debug)]
pub(crate) enum Storage {
    /// A read-only memory mapping of the file.
    Mapped(Arc<Mmap>),
    /// A copy of the file in the heap, for files the system refuses to map.
    Heap(Vec<u8>),
}

impl Storage {
    /// Map the file at `path`, or read it into memory when mapping fails.
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        // SAFETY: memmap2 asks that nothing modifies or truncates the file while it is mapped.
        // bobcat only reads model files. Another process that truncates a model file during a run
        // makes later reads fault, a risk every memory-mapping model loader takes.
        let mapping = unsafe { Mmap::map(&file) };
        if let Ok(mapping) = mapping {
            return Ok(Self::Mapped(Arc::new(mapping)));
        }
        // Some file systems refuse mmap for some files. Reading the file into memory works for
        // any readable file.
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(Self::Heap(bytes))
    }
}

impl AsRef<[u8]> for Storage {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Mapped(mapping) => mapping,
            Self::Heap(bytes) => bytes,
        }
    }
}
