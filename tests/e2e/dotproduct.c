// XFAIL: *
// RUN: %carts compile %samples_dir/dotproduct/dotproduct.c -O3 --arts-config
// %arts_config -o %t_arts RUN: env artsConfig=%arts_config %t_arts | %FileCheck
// %s CHECK: [CARTS] {{.*}}: PASS
