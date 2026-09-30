//! Metal 4 submission, constant storage, and resource lifetime regressions.

#![cfg(target_os = "macos")]

use std::error::Error;
use std::sync::Barrier;
use std::thread;

use bobcat::metal::{Error as MetalError, Metal};

#[test]
fn concurrent_backends_compile_and_submit_independently() -> Result<(), Box<dyn Error>> {
    let probe = match Metal::open() {
        Ok(metal) => metal,
        Err(error @ (MetalError::NoDevice | MetalError::Metal4Unsupported)) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    drop(probe);
    // Eight workers repeatedly open both pipeline sets to exercise the macOS 26 compiler race.
    let barrier = Barrier::new(8);
    thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|worker| {
                let barrier = &barrier;
                scope.spawn(move || -> Result<(), MetalError> {
                    barrier.wait();
                    for iteration in 0..4 {
                        let mut metal = if (worker + iteration) % 2 == 0 {
                            Metal::open_simd_matmul()?
                        } else {
                            Metal::open_tensor_matmul()?
                        };
                        let input = metal.new_buffer(32)?;
                        let output = metal.new_buffer(32)?;
                        let expected = [worker as f32 + iteration as f32; 8];
                        metal.write(input.at(0), &expected)?;
                        metal.begin()?;
                        metal.copy(input.at(0), output.at(0), 8)?;
                        metal.end()?;
                        let mut got = [0.0_f32; 8];
                        metal.read(output.at(0), &mut got)?;
                        assert_eq!(got.map(f32::to_bits), expected.map(f32::to_bits));
                    }
                    Ok(())
                })
            })
            .collect();
        for worker in workers {
            worker
                .join()
                .expect("the Metal initialization worker must finish")?;
        }
        Ok::<_, MetalError>(())
    })?;
    Ok(())
}

#[test]
fn queued_commands_preserve_constants_and_dropped_buffers() -> Result<(), Box<dyn Error>> {
    let mut metal = match Metal::open() {
        Ok(metal) => metal,
        Err(error @ (MetalError::NoDevice | MetalError::Metal4Unsupported)) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let input = metal.new_buffer(32)?;
    let temporary = metal.new_buffer(32)?;
    let output = metal.new_buffer(32)?;
    metal.write(input.at(0), &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    assert!(matches!(
        metal.new_readback(usize::MAX),
        Err(MetalError::Allocation(_))
    ));
    let mut readback = metal.new_readback(1)?;
    for slot in [0, 1, usize::MAX] {
        assert!(matches!(
            metal.read_readback(&readback, slot),
            Err(MetalError::Access { .. })
        ));
    }
    metal.begin()?;
    assert!(matches!(
        metal.argmax_readback(input.at(0), temporary.at(0), &mut readback, usize::MAX, 8),
        Err(MetalError::Access { .. })
    ));
    metal.discard();
    metal.begin()?;
    metal.argmax_readback(input.at(0), temporary.at(0), &mut readback, 0, 8)?;
    metal.end()?;
    assert_eq!(metal.read_readback(&readback, 0)?, 7);
    metal.begin()?;
    metal.copy(input.at(0), temporary.at(0), 8)?;
    metal.commit()?;
    metal.begin()?;
    metal.copy(temporary.at(0), output.at(0), 3)?;
    let last = metal.commit()?;
    drop(input);
    drop(temporary);
    assert!(matches!(
        metal.read(output.at(0), &mut [0.0_f32; 8]),
        Err(MetalError::Busy)
    ));
    metal.wait(last)?;
    let mut got = [0.0_f32; 8];
    metal.read(output.at(0), &mut got)?;
    assert_eq!(
        got.map(f32::to_bits),
        [1.0_f32, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0].map(f32::to_bits)
    );

    // Both reusable command slots must deliver feedback on every subsequent submission.
    for count in 1..=8 {
        metal.begin()?;
        metal.copy(output.at(0), output.at(0), count)?;
        metal.end()?;
    }
    metal.begin()?;
    metal.copy(output.at(0), output.at(0), 8)?;
    metal.discard();
    metal.set_profiling(true);
    metal.begin()?;
    metal.copy(output.at(0), output.at(0), 8)?;
    metal.end()?;
    assert_eq!(metal.profile().len(), 1);
    assert_eq!(metal.profile()[0].calls, 1);
    assert!(metal.profile()[0].seconds.is_finite());
    metal.set_profiling(false);
    Ok(())
}
