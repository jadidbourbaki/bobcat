//! Metal 4 bandwidth dispatch and completion feedback regressions.

#![cfg(target_os = "macos")]

use std::error::Error;

use bobcat::metal::{Error as MetalError, measure_bandwidth};

#[test]
fn bandwidth_reports_completed_repeated_and_partial_grids() -> Result<(), Box<dyn Error>> {
    let grids = [17, 257, 1024];
    let measured = match measure_bandwidth(1024 * 1024, &grids, 3) {
        Ok(measured) => measured,
        Err(error @ (MetalError::NoDevice | MetalError::Metal4Unsupported)) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    assert_eq!(measured.samples.len(), grids.len());
    for (sample, threads) in measured.samples.iter().zip(grids) {
        assert_eq!(sample.threads, threads);
        assert!(sample.gigabytes_per_second.is_finite());
        assert!(sample.gigabytes_per_second > 0.0);
    }
    for (bytes, grids, reps) in [(0, &[17][..], 1), (16, &[0][..], 1), (16, &[17][..], 0)] {
        assert!(matches!(
            measure_bandwidth(bytes, grids, reps),
            Err(MetalError::BandwidthInput(_))
        ));
    }
    Ok(())
}
