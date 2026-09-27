# bw reads memory with NEON loads, so it builds only on 64-bit ARM.
if host_machine.cpu_family() == 'aarch64'
  executable('bw', 'bw.c', dependencies: threads)
endif
