project(
  'gip',
  'c',
  version: '0.1.0',
  meson_version: '>=1.3.0',
  default_options: [
    'c_std=gnu17',
    'warning_level=2',
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
  ],
)

cc = meson.get_compiler('c')

add_project_arguments(
  '-D_GNU_SOURCE',
  '-DGIP_VERSION="@0@"'.format(meson.project_version()),
  cc.get_supported_arguments(
    '-Wshadow',
    '-Wformat=2',
    '-Wimplicit-fallthrough',
    '-Wstrict-prototypes',
    '-Wmissing-prototypes',
    '-Werror=implicit',
    '-Werror=incompatible-pointer-types',
    '-Werror=int-conversion',
    '-Werror=format-security',
  ),
  language: 'c',
)

threads = dependency('threads')

subdir('tools')
