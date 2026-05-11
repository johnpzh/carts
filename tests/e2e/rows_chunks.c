// RUN: %carts compile %samples_dir/rows/chunks/rowchunk.c -O3 --arts-config
// %arts_config -o %t_arts RUN: env artsConfig=%arts_config %t_arts | %FileCheck
// %s CHECK: [CARTS] {{.*}}: PASS
