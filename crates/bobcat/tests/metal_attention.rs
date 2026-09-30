//! Checks bounded-query attention against individual GPU queries and a double-precision reference.

#![cfg(target_os = "macos")]

use std::error::Error;

use bobcat::metal::{Metal, attention_scratch_floats};
use half::f16;

#[test]
fn attention_query_batches_preserve_causal_tails() -> Result<(), Box<dyn Error>> {
    let mut metal = match Metal::open() {
        Ok(metal) => metal,
        Err(error) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
    };
    for (first, queries, heads, kv_heads, dim) in [
        (0_u32, 7_u32, 4_u32, 1_u32, 32_u32),
        (63, 7, 4, 2, 64),
        (64, 73, 4, 1, 128),
        (63, 70, 4, 2, 64),
        (129, 5, 2, 2, 32),
    ] {
        for half_cache in [false, true] {
            let keys = first + queries;
            let context = keys + 67;
            let q_dim = (heads * dim) as usize;
            let kv_dim = (kv_heads * dim) as usize;
            let q: Vec<f32> = (0..queries as usize * q_dim)
                .map(|i| ((i * 19 % 97) as f32 - 48.0) / 67.0)
                .collect();
            let values = |factor| {
                (0..keys as usize * kv_dim)
                    .map(|i| {
                        let value = ((i * factor % 83) as f32 - 41.0) / 53.0;
                        if half_cache {
                            f16::from_f32(value).to_f32()
                        } else {
                            value
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let k = values(29);
            let v = values(31);
            let q_buffer = metal.new_buffer(q.len() * 4)?;
            metal.write(q_buffer.at(0), &q)?;
            let cache_bytes = k.len() * if half_cache { 2 } else { 4 };
            let k_buffer = metal.new_buffer(cache_bytes)?;
            let v_buffer = metal.new_buffer(cache_bytes)?;
            if half_cache {
                let bytes = |values: &[f32]| {
                    values
                        .iter()
                        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                        .collect::<Vec<_>>()
                };
                metal.write(k_buffer.at(0), &bytes(&k))?;
                metal.write(v_buffer.at(0), &bytes(&v))?;
            } else {
                metal.write(k_buffer.at(0), &k)?;
                metal.write(v_buffer.at(0), &v)?;
            }
            let scratch =
                metal.new_buffer(attention_scratch_floats(heads, dim, context, queries) * 4)?;
            let tiled = metal.new_buffer(q.len() * 4)?;
            let separate = metal.new_buffer(q.len() * 4)?;
            metal.begin()?;
            metal.attention(
                q_buffer.at(0),
                k_buffer.at(0),
                v_buffer.at(0),
                half_cache,
                scratch.at(0),
                tiled.at(0),
                heads,
                kv_heads,
                dim,
                first,
                queries,
                context,
            )?;
            metal.end()?;
            for qi in 0..queries {
                metal.begin()?;
                metal.attention(
                    q_buffer.floats(qi as usize * q_dim),
                    k_buffer.at(0),
                    v_buffer.at(0),
                    half_cache,
                    scratch.at(0),
                    separate.floats(qi as usize * q_dim),
                    heads,
                    kv_heads,
                    dim,
                    first + qi,
                    1,
                    context,
                )?;
                metal.end()?;
            }
            let mut got = vec![0.0_f32; q.len()];
            let mut single = got.clone();
            metal.read(tiled.at(0), &mut got)?;
            metal.read(separate.at(0), &mut single)?;
            for (&a, &b) in got.iter().zip(&single) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "tiled query changes the result: {a} / {b}"
                );
            }
            // The scale matches the float32 value passed to the GPU. Dot products and softmax
            // use double precision to check the GPU reduction order within an absolute 1e-6.
            let scale = f64::from(1.0 / (dim as f32).sqrt());
            for qi in 0..queries as usize {
                for head in 0..heads as usize {
                    let kv_head = head / (heads / kv_heads) as usize;
                    let query = &q[qi * q_dim + head * dim as usize..][..dim as usize];
                    let mut scores: Vec<f64> = (0..=first as usize + qi)
                        .map(|t| {
                            query
                                .iter()
                                .enumerate()
                                .map(|(d, &x)| {
                                    f64::from(x)
                                        * f64::from(k[t * kv_dim + kv_head * dim as usize + d])
                                })
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for score in &mut scores {
                        *score = (*score - best).exp();
                    }
                    let total = scores.iter().sum::<f64>();
                    for d in 0..dim as usize {
                        let want = scores
                            .iter()
                            .enumerate()
                            .map(|(t, &weight)| {
                                weight * f64::from(v[t * kv_dim + kv_head * dim as usize + d])
                            })
                            .sum::<f64>()
                            / total;
                        let actual = f64::from(got[qi * q_dim + head * dim as usize + d]);
                        assert!(
                            (actual - want).abs() < 1e-6,
                            "scalar attention: {actual} / {want}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
