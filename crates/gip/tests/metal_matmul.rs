//! Checks full and partial Metal matrix tiles against the scalar reference.

#![cfg(target_os = "macos")]

use std::error::Error;

use gip::TensorType;
use gip::metal::{Format, Metal, Store};
use gip::scalar;

fn check_shape(
    metal: &mut Metal,
    tensor_type: TensorType,
    format: Format,
    n_rows: u32,
    n_cols: u32,
    n_tokens: u32,
    store: Store,
) -> Result<(), Box<dyn Error>> {
    let rows = n_rows as usize;
    let cols = n_cols as usize;
    let tokens = n_tokens as usize;
    let (block_weights, block_bytes) = tensor_type.block();
    let mut weights = Vec::with_capacity(scalar::row_bytes(tensor_type, cols) * rows);
    for block in 0..rows * cols / block_weights {
        let mut bytes = vec![0; block_bytes];
        bytes[..2].copy_from_slice(&0x3000_u16.to_le_bytes());
        match tensor_type {
            TensorType::Q8_0 => {
                for (i, quant) in bytes[2..].iter_mut().enumerate() {
                    *quant = u8::try_from((block * 17 + i * 13) % 127)?;
                }
            }
            TensorType::Q4K => {
                bytes[2..4].copy_from_slice(&0x3000_u16.to_le_bytes());
                bytes[4..16].fill(0x11);
                for (i, quant) in bytes[16..].iter_mut().enumerate() {
                    *quant = u8::try_from((block * 17 + i * 13) % 251)?;
                }
            }
            _ => unreachable!("the tile test uses Q8_0 and Q4_K weights"),
        }
        weights.extend_from_slice(&bytes);
    }
    let input: Vec<f32> = (0..tokens * cols)
        .map(|i| ((i * 19 % 97) as f32 - 48.0) / 64.0)
        .collect();
    let prior: Vec<f32> = (0..tokens * rows)
        .map(|i| ((i * 11 % 53) as f32 - 26.0) / 32.0)
        .collect();
    let mut want = prior.clone();
    let mut product = vec![0.0; rows];
    for (token, output) in input.chunks_exact(cols).zip(want.chunks_exact_mut(rows)) {
        scalar::matvec(tensor_type, &weights, token, &mut product);
        for (value, &product) in output.iter_mut().zip(&product) {
            *value = match store {
                Store::Overwrite => product,
                Store::Accumulate => *value + product,
                Store::Swiglu => scalar::silu(*value) * product,
            };
        }
    }

    let weight_buffer = metal.new_buffer(weights.len())?;
    let input_buffer = metal.new_buffer(input.len() * 4)?;
    let output_buffer = metal.new_buffer(prior.len() * 4)?;
    metal.write(weight_buffer.at(0), &weights)?;
    metal.write(input_buffer.at(0), &input)?;
    metal.write(output_buffer.at(0), &prior)?;
    metal.begin()?;
    metal.matmul(
        format,
        weight_buffer.at(0),
        n_rows,
        n_cols,
        input_buffer.at(0),
        output_buffer.at(0),
        n_tokens,
        store,
    )?;
    metal.end()?;

    let mut got = vec![0.0; want.len()];
    metal.read(output_buffer.at(0), &mut got)?;
    let scale = want.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let error = got
        .iter()
        .zip(&want)
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0, f32::max);
    assert!(
        error / scale < 1e-4,
        "{format:?} {store:?} {n_rows}x{n_cols} by {n_tokens}: {error} / {scale}"
    );
    Ok(())
}

#[test]
fn q8_0_matrix_tiles_match_scalar() -> Result<(), Box<dyn Error>> {
    let mut metal = match Metal::open() {
        Ok(metal) => metal,
        Err(error) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
    };
    for (rows, cols, tokens) in [(64, 64, 32), (70, 96, 33)] {
        for store in [Store::Overwrite, Store::Accumulate, Store::Swiglu] {
            check_shape(
                &mut metal,
                TensorType::Q8_0,
                Format::Q8_0,
                rows,
                cols,
                tokens,
                store,
            )?;
        }
    }
    Ok(())
}

#[test]
fn q4k_matrix_tiles_match_scalar() -> Result<(), Box<dyn Error>> {
    let mut metal = match Metal::open() {
        Ok(metal) => metal,
        Err(error) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
    };
    for (rows, cols, tokens) in [(64, 256, 32), (70, 256, 33)] {
        for store in [Store::Overwrite, Store::Accumulate, Store::Swiglu] {
            check_shape(
                &mut metal,
                TensorType::Q4K,
                Format::Q4K,
                rows,
                cols,
                tokens,
                store,
            )?;
        }
    }
    Ok(())
}
