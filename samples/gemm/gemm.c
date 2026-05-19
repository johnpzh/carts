#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"

#define TOLERANCE 1e-8

static double checksum(double **C, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      sum += C[i][j];
  return sum;
}

int main(int argc, char **argv) {
  CARTS_TIMER_START();
  const int n = (argc > 1) ? atoi(argv[1]) : 128;
  if (n <= 0) {
    CARTS_TEST_FAIL("GEMM size must be positive");
  }

  double **A = (double **)malloc((size_t)n * sizeof(double *));
  double **B = (double **)malloc((size_t)n * sizeof(double *));
  double **C = (double **)malloc((size_t)n * sizeof(double *));
  double **Ref = (double **)malloc((size_t)n * sizeof(double *));
  if (!A || !B || !C || !Ref) {
    CARTS_TEST_FAIL("allocation failed");
  }
  for (int i = 0; i < n; ++i) {
    A[i] = (double *)malloc((size_t)n * sizeof(double));
    B[i] = (double *)malloc((size_t)n * sizeof(double));
    C[i] = (double *)malloc((size_t)n * sizeof(double));
    Ref[i] = (double *)malloc((size_t)n * sizeof(double));
    if (!A[i] || !B[i] || !C[i] || !Ref[i]) {
      CARTS_TEST_FAIL("row allocation failed");
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i][j] = (double)((i + j) % 17) / 17.0;
      B[i][j] = (double)((i - j + 31) % 19) / 19.0;
      C[i][j] = 0.0;
      Ref[i][j] = 0.0;
    }
  }

  double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = C[i][j];
      for (int k = 0; k < n; ++k)
        acc += A[i][k] * B[k][j];
      C[i][j] = acc;
    }
  }
  double stop = omp_get_wtime();

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = Ref[i][j];
      for (int k = 0; k < n; ++k)
        acc += A[i][k] * B[k][j];
      Ref[i][j] = acc;
    }
  }

  int verified = 1;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (fabs(C[i][j] - Ref[i][j]) > TOLERANCE) {
        verified = 0;
        break;
      }
    }
  }

  printf("gemm n=%d kernel=%f checksum=%0.12f\n", n, stop - start,
         checksum(C, n));

  for (int i = 0; i < n; ++i) {
    free(A[i]);
    free(B[i]);
    free(C[i]);
    free(Ref[i]);
  }
  free(A);
  free(B);
  free(C);
  free(Ref);

  if (!verified) {
    CARTS_TEST_FAIL("GEMM mismatch");
  }
  CARTS_TEST_PASS();
}
