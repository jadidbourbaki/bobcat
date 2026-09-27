# bw reads memory with NEON loads, so it builds only on 64-bit ARM.
if host_machine.cpu_family() == 'aarch64'
  executable('bw', 'bw.c', dependencies: threads)
endif

if host_machine.system() == 'darwin'
  metal = dependency('appleframeworks', modules: ['Foundation', 'Metal'])
  executable(
    'gpu_bw',
    'gpu_bw.m',
    dependencies: metal,
    objc_args: ['-fobjc-arc'],
  )
endif
