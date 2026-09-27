gguf_malformed = executable(
  'gguf_malformed',
  'gguf_malformed.c',
  include_directories: gip_inc,
  link_with: gip_core,
)
test('gguf_malformed', gguf_malformed)

if have_metal
  metal_matvec = executable(
    'metal_matvec',
    'metal_matvec.c',
    include_directories: gip_inc,
    link_with: gip_core,
    dependencies: [m_dep, metal_dep],
  )
  test('metal_matvec', metal_matvec)
endif

lfm2_reference = executable(
  'lfm2_reference',
  'lfm2_reference.c',
  include_directories: gip_inc,
  link_with: gip_core,
  dependencies: m_dep,
)
# The BF16 file holds the checkpoint's own weights.  The Q8_0 file's
# reference comes from transformers loading the same GGUF, so both sides
# see identical dequantized weights.
foreach variant : [
  ['bf16', 'LFM2.5-350M-BF16.gguf', 'LFM2.5-350M'],
  ['q8_0', 'LFM2.5-350M-Q8_0.gguf', 'LFM2.5-350M-Q8_0'],
]
  test(
    'lfm2_reference_' + variant[0],
    lfm2_reference,
    args: [
      meson.project_source_root() / 'models' / variant[1],
      meson.project_source_root() / 'models/ref' / variant[2],
    ],
    timeout: 900,
  )
endforeach
