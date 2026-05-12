// RUN: %carts compile %samples_dir/matrixmul/matrixmul.c -O3 --arts-config %arts_config -o %t_arts
// RUN: env artsConfig=%arts_config %t_arts | %FileCheck %s
// CHECK: [CARTS] {{.*}}: PASS
