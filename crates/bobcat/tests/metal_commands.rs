//! Metal 4 submission, constant storage, and resource lifetime regressions.

#![cfg(target_os = "macos")]

use std::error::Error;

use bobcat::metal::{Error as MetalError, Metal};

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
