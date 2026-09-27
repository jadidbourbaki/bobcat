embed = executable(
  'embed',
  meson.project_source_root() / 'tools/embed.c',
  native: true,
)

metal_source = custom_target(
  'metal_source',
  input: 'kernels.metal',
  output: 'metal_source.c',
  command: [embed, 'gip_metal_source', '@INPUT@', '@OUTPUT@'],
)

metal_sources = [files('backend.m'), metal_source]
metal_dep = dependency('appleframeworks', modules: ['Foundation', 'Metal'])
