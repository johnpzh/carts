// RUN: rm -rf %t.pipes
// RUN: %carts compile %s --all-pipelines -O3 --arts-config %arts_config -o %t.pipes
// RUN: %FileCheck %s --input-file=%t.pipes/5_rt/stages/12_pre-lowering.mlir

// CHECK: arts_rt.rec_dep
// CHECK-SAME: acquire_modes = array<i32: 3>

extern void *malloc(unsigned long);
extern void free(void *);
extern int printf(const char *, ...);

int main() {
  int n = 64;
  float *A = (float *)malloc((unsigned long)n * n * sizeof(float));
  float *B = (float *)malloc((unsigned long)n * n * sizeof(float));

  for (int i = 0; i < n * n; ++i)
    A[i] = (float)i;

#pragma omp parallel for
  for (int i = 1; i < n - 1; ++i)
    for (int j = 1; j < n - 1; ++j)
      B[i * n + j] =
          A[(i - 1) * n + j] + A[i * n + j] + A[(i + 1) * n + j];

  float sum = 0.0f;
  for (int i = 1; i < n - 1; ++i)
    for (int j = 1; j < n - 1; ++j)
      sum += B[i * n + j];

  printf("%f\n", sum);
  free(A);
  free(B);
  return sum == 0.0f;
}
