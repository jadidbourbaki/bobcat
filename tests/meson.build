gguf_malformed = executable(
  'gguf_malformed',
  'gguf_malformed.c',
  include_directories: gip_inc,
  link_with: gip_core,
)
test('gguf_malformed', gguf_malformed)

lfm2_reference = executable(
  'lfm2_reference',
  'lfm2_reference.c',
  include_directories: gip_inc,
  link_with: gip_core,
  dependencies: m_dep,
)
test(
  'lfm2_reference',
  lfm2_reference,
  args: [
    meson.project_source_root() / 'models/LFM2.5-350M-BF16.gguf',
    meson.project_source_root() / 'models/ref/LFM2.5-350M',
  ],
  timeout: 900,
)
