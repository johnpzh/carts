// XFAIL: *
// RUN: %carts compile %samples_dir/mixed_access/mixed_access.c -O3
// --arts-config %arts_config -o %t_arts RUN: env artsConfig=%arts_config
// %t_arts | %FileCheck %s CHECK: [CARTS] {{.*}}: PASS
