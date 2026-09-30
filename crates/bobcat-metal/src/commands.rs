//! Metal 4 command recording with reusable allocators and explicit resource lifetimes.

use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use block2::RcBlock;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4CommitFeedback, MTL4CommitOptions,
    MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLBuffer, MTLComputePipelineState,
    MTLDevice, MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLStages,
};

use super::{Arg, Dispatch, Error, Ticket, describe, size, to_u64};

// A decode step uses a small fraction of one arena. Extra arenas accommodate large recordings.
const CONSTANT_BYTES: usize = 1024 * 1024;
const CONSTANT_STRIDE: usize = 16;
const BUFFER_BINDINGS: usize = 31;
// Covers GENERATE_IN_FLIGHT in bobcat's lfm2_metal.rs without allocating during decode.
const INITIAL_COMMAND_SLOTS: usize = 3;

#[derive(Debug)]
struct Completion {
    seconds: f64,
    error: Option<String>,
}

type Feedback = Arc<(Mutex<Option<Completion>>, Condvar)>;
type FeedbackHandler = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTL4CommitFeedback>>)>;

#[derive(Debug)]
struct Arena {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pointer: NonNull<u8>,
    address: u64,
}

impl Arena {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let buffer = device
            .newBufferWithLength_options(CONSTANT_BYTES, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Allocation(CONSTANT_BYTES))?;
        Ok(Self {
            pointer: buffer.contents().cast::<u8>(),
            address: buffer.gpuAddress(),
            buffer,
        })
    }
}

/// Command storage that remains live until its commit feedback arrives.
#[derive(Debug)]
struct Slot {
    buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    constants: Vec<Arena>,
    scalars: HashMap<u32, u64>,
    used: usize,
    resources: Vec<(u64, Retained<ProtocolObject<dyn MTLBuffer>>)>,
    feedback: Feedback,
    callback: FeedbackHandler,
    options: Retained<MTL4CommitOptions>,
}

impl Slot {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let allocator = device.newCommandAllocator().ok_or(Error::CommandBuffer)?;
        let buffer = device.newCommandBuffer().ok_or(Error::CommandBuffer)?;
        let residency = device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|error| Error::CommandSetup(describe(&error)))?;
        let constants = Arena::new(device)?;
        let feedback: Feedback = Arc::new((Mutex::new(None), Condvar::new()));
        let result = Arc::clone(&feedback);
        let callback = RcBlock::new(
            move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                autoreleasepool(|_| {
                    // SAFETY: Metal supplies a live feedback object for the duration of the callback.
                    let feedback = unsafe { feedback.as_ref() };
                    let completion = Completion {
                        seconds: feedback.GPUEndTime() - feedback.GPUStartTime(),
                        error: feedback.error().map(|error| describe(&error)),
                    };
                    *result.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(completion);
                    result.1.notify_one();
                });
            },
        );
        let options = MTL4CommitOptions::new();
        Ok(Self {
            buffer,
            allocator,
            residency,
            constants: vec![constants],
            scalars: HashMap::with_capacity(128),
            used: 0,
            resources: Vec::with_capacity(128),
            feedback,
            callback,
            options,
        })
    }

    fn prepare(&mut self) {
        // Slots enter the free pool only after GPU completion or a discarded recording.
        self.allocator.reset();
        self.residency.removeAllAllocations();
        self.resources.clear();
        for buffer in &self.constants {
            self.residency
                .addAllocation(ProtocolObject::from_ref(&*buffer.buffer));
        }
        self.scalars.clear();
        self.used = 0;
        self.buffer.beginCommandBufferWithAllocator(&self.allocator);
        self.buffer.useResidencySet(&self.residency);
    }

    fn address(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        arg: &Arg<'_>,
    ) -> Result<u64, Error> {
        match arg {
            Arg::Buffer { view, .. } => {
                let address = view.buffer.address;
                if !self.resources.iter().any(|(old, _)| *old == address) {
                    self.residency
                        .addAllocation(ProtocolObject::from_ref(&*view.buffer.raw));
                    self.resources.push((address, view.buffer.raw.clone()));
                }
                Ok(address + to_u64(view.offset))
            }
            Arg::U32(_) | Arg::F32(_) => {
                let bits = match arg {
                    Arg::U32(value) => *value,
                    Arg::F32(value) => value.to_bits(),
                    Arg::Buffer { .. } => unreachable!("buffer arguments have a GPU address"),
                };
                // Scalars are immutable within a recording. Identical bits share storage even
                // when kernels interpret the bits with different scalar types.
                if let Some(address) = self.scalars.get(&bits) {
                    return Ok(*address);
                }
                let index = self.used / CONSTANT_BYTES;
                let offset = self.used % CONSTANT_BYTES;
                if index == self.constants.len() {
                    let buffer = Arena::new(device)?;
                    self.residency
                        .addAllocation(ProtocolObject::from_ref(&*buffer.buffer));
                    self.constants.push(buffer);
                }
                let buffer = &self.constants[index];
                // SAFETY: `offset` names a 16-byte aligned range within the arena.
                let pointer = unsafe { buffer.pointer.add(offset) }.cast::<u32>();
                // SAFETY: The arena is shared and this slot is not in flight. Each scalar gets
                // a separate aligned range, kept live through GPU completion.
                unsafe { pointer.write(bits) };
                self.used += CONSTANT_STRIDE;
                let address = buffer.address + to_u64(offset);
                self.scalars.insert(bits, address);
                Ok(address)
            }
        }
    }

    fn completion(&self) -> Completion {
        let mut result = self
            .feedback
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        while result.is_none() {
            result = self
                .feedback
                .1
                .wait(result)
                .unwrap_or_else(PoisonError::into_inner);
        }
        result
            .take()
            .expect("the feedback handler supplied a completion")
    }
}

#[derive(Debug)]
struct Recording {
    slot: Slot,
    encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
}

impl Recording {
    fn end(self) -> Slot {
        self.encoder.endEncoding();
        self.slot.buffer.endCommandBuffer();
        self.slot
    }
}

/// The Metal 4 submission path shared by prefill, decode, and profiling.
#[derive(Debug)]
pub(super) struct Commands {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    bindings: [Option<u64>; BUFFER_BINDINGS],
    recording: Option<Recording>,
    free: Vec<Slot>,
    pending: VecDeque<Slot>,
    committed: u64,
    completed: u64,
}

impl Commands {
    pub(super) fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Result<Self, Error> {
        let queue = device.newMTL4CommandQueue().ok_or(Error::Queue)?;
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(BUFFER_BINDINGS);
        let table = device
            .newArgumentTableWithDescriptor_error(&descriptor)
            .map_err(|error| Error::CommandSetup(describe(&error)))?;
        Ok(Self {
            device: device.clone(),
            queue,
            table,
            bindings: [None; BUFFER_BINDINGS],
            recording: None,
            free: (0..INITIAL_COMMAND_SLOTS)
                .map(|_| Slot::new(device))
                .collect::<Result<_, _>>()?,
            pending: VecDeque::with_capacity(INITIAL_COMMAND_SLOTS),
            committed: 0,
            completed: 0,
        })
    }

    fn recording(&mut self) -> Result<Recording, Error> {
        let mut slot = match self.free.pop() {
            Some(slot) => slot,
            None => Slot::new(&self.device)?,
        };
        slot.prepare();
        let encoder = autoreleasepool(|_| slot.buffer.computeCommandEncoder());
        let Some(encoder) = encoder else {
            slot.buffer.endCommandBuffer();
            self.free.push(slot);
            return Err(Error::CommandBuffer);
        };
        // Prefill batches and generated steps share buffers across submissions. Metal 4
        // requires an explicit dependency on prior dispatches from the same queue.
        encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        encoder.setArgumentTable(Some(&self.table));
        Ok(Recording { slot, encoder })
    }

    pub(super) fn begin(&mut self) -> Result<(), Error> {
        if self.recording.is_some() {
            return Err(Error::AlreadyRecording);
        }
        self.recording = Some(self.recording()?);
        Ok(())
    }

    pub(super) fn discard(&mut self) {
        if let Some(recording) = self.recording.take() {
            self.free.push(recording.end());
        }
    }

    pub(super) fn barrier(&self) {
        if let Some(recording) = &self.recording {
            recording
                .encoder
                .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
        }
    }

    fn submit(&mut self, recording: Recording) -> Ticket {
        let slot = recording.end();
        slot.residency.commit();
        // Metal consumes feedback handlers when submitting commit options.
        // SAFETY: The slot keeps the callback block live through this submission's completion.
        unsafe {
            slot.options
                .addFeedbackHandler(RcBlock::as_ptr(&slot.callback));
        };
        let mut pointer = NonNull::from(&*slot.buffer);
        // SAFETY: `pointer` names one live ended command buffer. The slot retains all command
        // storage, buffers, residency, and feedback options until this submission completes.
        unsafe {
            self.queue
                .commit_count_options(NonNull::from(&mut pointer), 1, &slot.options);
        };
        self.pending.push_back(slot);
        self.committed += 1;
        Ticket(self.committed)
    }

    pub(super) fn commit(&mut self) -> Result<Ticket, Error> {
        let recording = self.recording.take().ok_or(Error::NotRecording)?;
        Ok(self.submit(recording))
    }

    pub(super) fn wait(&mut self, ticket: Ticket) -> Result<f64, Error> {
        let mut seconds = 0.0;
        let mut failure = None;
        while self.completed < ticket.0
            && let Some(slot) = self.pending.pop_front()
        {
            let result = slot.completion();
            seconds += result.seconds;
            if result.error.is_some() {
                failure = result.error;
            }
            self.completed += 1;
            slot.residency.removeAllAllocations();
            slot.residency.commit();
            let mut slot = slot;
            slot.resources.clear();
            self.free.push(slot);
        }
        failure.map_or(Ok(seconds), |error| Err(Error::Execution(error)))
    }

    pub(super) fn busy(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Return the ticket that the open command buffer receives when it commits.
    pub(super) fn recording_ticket(&self) -> Result<Ticket, Error> {
        if self.recording.is_none() {
            return Err(Error::NotRecording);
        }
        Ok(Ticket(self.committed + 1))
    }

    /// Report whether a wait has seen the command buffer with `ticket` finish.
    pub(super) fn finished(&self, ticket: Ticket) -> bool {
        self.completed >= ticket.0
    }

    pub(super) fn launch(
        &mut self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        args: &[Arg<'_>],
        dispatch: &Dispatch,
        profiling: bool,
    ) -> Result<Option<f64>, Error> {
        let mut own = if profiling {
            Some(self.recording()?)
        } else {
            None
        };
        let recording = own
            .as_mut()
            .or(self.recording.as_mut())
            .ok_or(Error::NotRecording)?;
        let encoded = (|| {
            recording.encoder.setComputePipelineState(pipeline);
            for (index, arg) in args.iter().enumerate() {
                assert!(
                    index < BUFFER_BINDINGS,
                    "kernel arguments fit the argument table"
                );
                let address = recording.slot.address(&self.device, arg)?;
                if self.bindings[index] == Some(address) {
                    continue;
                }
                // SAFETY: Each index is below the table's binding count. Buffer ranges were
                // checked by the launch. The slot retains all buffers through completion.
                unsafe { self.table.setAddress_atIndex(address, index) };
                self.bindings[index] = Some(address);
            }
            match *dispatch {
                Dispatch::Threadgroups(grid, group) => recording
                    .encoder
                    .dispatchThreadgroups_threadsPerThreadgroup(size(grid), size(group)),
                Dispatch::Threads(grid, group) => recording
                    .encoder
                    .dispatchThreads_threadsPerThreadgroup(size(grid), size(group)),
            }
            Ok(())
        })();
        if let Err(error) = encoded {
            if let Some(recording) = own {
                self.free.push(recording.end());
            }
            return Err(error);
        }
        if let Some(recording) = own {
            let ticket = self.submit(recording);
            return self.wait(ticket).map(Some);
        }
        Ok(None)
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        self.discard();
        let _ = self.wait(Ticket(self.committed));
    }
}

#[cfg(test)]
mod tests {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLGPUFamily};

    use super::*;

    #[test]
    fn growing_constant_storage_keeps_earlier_addresses_live() -> Result<(), Error> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("skip: no Metal device");
            return Ok(());
        };
        if !device.supportsFamily(MTLGPUFamily::Metal4) {
            eprintln!("skip: Metal 4 is unavailable");
            return Ok(());
        }
        let mut slot = Slot::new(&device)?;
        slot.prepare();
        let first = slot.address(&device, &Arg::F32(12.5))?;
        let same = slot.address(&device, &Arg::U32(12.5_f32.to_bits()))?;
        assert_eq!(first, same);
        assert_eq!(slot.used, CONSTANT_STRIDE);
        let count = u32::try_from(CONSTANT_BYTES / CONSTANT_STRIDE)
            .expect("one constant arena holds fewer than u32::MAX scalars");
        for value in 0..count {
            slot.address(&device, &Arg::U32(value))?;
        }
        slot.buffer.endCommandBuffer();
        assert_eq!(slot.constants.len(), 2);
        assert_eq!(first, slot.constants[0].address);
        // SAFETY: Both arenas are shared, initialized, and have never been submitted to the GPU.
        let earlier = unsafe { slot.constants[0].pointer.cast::<u32>().read() };
        // SAFETY: The new arena contains the last scalar at its aligned first address.
        let later = unsafe { slot.constants[1].pointer.cast::<u32>().read() };
        assert_eq!(earlier, 12.5_f32.to_bits());
        assert_eq!(later, count - 1);
        slot.prepare();
        let changed = slot.address(&device, &Arg::U32(7))?;
        let restored = slot.address(&device, &Arg::F32(12.5))?;
        slot.buffer.endCommandBuffer();
        assert_eq!(changed, first);
        assert_eq!(restored, first + to_u64(CONSTANT_STRIDE));
        // SAFETY: The arena was never submitted and now stores the new recording's first scalar.
        let overwritten = unsafe { slot.constants[0].pointer.cast::<u32>().read() };
        assert_eq!(overwritten, 7);
        Ok(())
    }
}
