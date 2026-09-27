gguf_malformed = executable(
  'gguf_malformed',
  'gguf_malformed.c',
  include_directories: gip_inc,
  link_with: gip_core,
)
test('gguf_malformed', gguf_malformed)

test_deps = [m_dep]
if have_metal
  test_deps += metal_dep
  metal_matvec = executable(
    'metal_matvec',
    'metal_matvec.c',
    include_directories: gip_inc,
    link_with: gip_core,
    dependencies: test_deps,
  )
  test('metal_matvec', metal_matvec)
endif

lfm2_reference = executable(
  'lfm2_reference',
  'lfm2_reference.c',
  include_directories: gip_inc,
  link_with: gip_core,
  dependencies: test_deps,
  c_args: ['-DGIP_HAVE_METAL=@0@'.format(have_metal ? 1 : 0)],
)

# The BF16 file holds the checkpoint's own weights.  The Q8_0 file's
# reference comes from transformers loading the same GGUF, so both sides
# see identical dequantized weights.  The Metal pass runs Q8_0 only.
foreach variant : [
  ['bf16', 'LFM2.5-350M-BF16.gguf', 'LFM2.5-350M', 'scalar'],
  ['q8_0', 'LFM2.5-350M-Q8_0.gguf', 'LFM2.5-350M-Q8_0', 'scalar'],
  ['metal_q8_0', 'LFM2.5-350M-Q8_0.gguf', 'LFM2.5-350M-Q8_0', 'metal'],
]
  test(
    'lfm2_reference_' + variant[0],
    lfm2_reference,
    args: [
      meson.project_source_root() / 'models' / variant[1],
      meson.project_source_root() / 'models/ref' / variant[2],
      variant[3],
    ],
    timeout: 900,
  )
endforeach
