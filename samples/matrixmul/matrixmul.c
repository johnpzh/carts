#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"

#define TOLERANCE 1e-6

int main(int argc, char *argv[]) {
  CARTS_TIMER_START();
  const int N = (argc >= 2) ? atoi(argv[1]) : 64;
  const int BS = (argc >= 3) ? atoi(argv[2]) : 16;
  if (N <= 0 || BS <= 0) {
    printf("Matrix size and block size must be positive integers.\n");
    return EXIT_FAILURE;
  }
  if (N % BS != 0) {
    printf("Block size (BS=%d) must evenly divide matrix size (N=%d).\n", BS,
           N);
    return EXIT_FAILURE;
  }

  /// Allocate matrices
  float **A = (float **)malloc(N * sizeof(float *));
  float **B = (float **)malloc(N * sizeof(float *));
  float **C_parallel = (float **)malloc(N * sizeof(float *));
  float **C_sequential = (float **)malloc(N * sizeof(float *));
  for (int i = 0; i < N; i++) {
    A[i] = (float *)malloc(N * sizeof(float));
    B[i] = (float *)malloc(N * sizeof(float));
    C_parallel[i] = (float *)malloc(N * sizeof(float));
    C_sequential[i] = (float *)malloc(N * sizeof(float));
  }

  /// Initialize matrices
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = (float)(i + j);
      B[i][j] = (float)(i - j);
      C_parallel[i][j] = 0.0f;
      C_sequential[i][j] = 0.0f;
    }
  }

  /// Print A and B (debug only)
  CARTS_DEBUG_PRINT("Matrix A:\n");
  for (int i = 0; i < N; i++) {
    CARTS_DEBUG_PRINT("  ");
    for (int j = 0; j < N; j++)
      CARTS_DEBUG_PRINT("%6.2f ", A[i][j]);
    CARTS_DEBUG_PRINT("\n");
  }

  CARTS_DEBUG_PRINT("\nMatrix B:\n");
  for (int i = 0; i < N; i++) {
    CARTS_DEBUG_PRINT("  ");
    for (int j = 0; j < N; j++)
      CARTS_DEBUG_PRINT("%6.2f ", B[i][j]);
    CARTS_DEBUG_PRINT("\n");
  }

  /// Parallel computation
  double t_start = omp_get_wtime();
#pragma omp parallel
#pragma omp single
  {
    for (int i = 0; i < N; i += BS) {
      for (int j = 0; j < N; j += BS) {
        for (int k = 0; k < N; k += BS) {
#pragma omp task depend(in : A[i][k], B[k][j]) depend(inout : C_parallel[i][j])
          {
            for (int ii = i; ii < i + BS; ii++)
              for (int jj = j; jj < j + BS; jj++)
                for (int kk = k; kk < k + BS; kk++)
                  C_parallel[ii][jj] += A[ii][kk] * B[kk][jj];
          }
        }
      }
    }
  }
  double t_end = omp_get_wtime();
  printf("Finished in %f seconds\n", t_end - t_start);

  /// Sequential computation
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++)
        sum += A[i][k] * B[k][j];
      C_sequential[i][j] = sum;
    }
  }

  /// Print C parallel (debug only)
  CARTS_DEBUG_PRINT("\nMatrix C (Parallel Result):\n");
  for (int i = 0; i < N; i++) {
    CARTS_DEBUG_PRINT("  ");
    for (int j = 0; j < N; j++)
      CARTS_DEBUG_PRINT("%8.2f ", C_parallel[i][j]);
    CARTS_DEBUG_PRINT("\n");
  }

  /// Verification
  int verified = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(C_parallel[i][j] - C_sequential[i][j]) > TOLERANCE) {
        printf("Mismatch at (%d,%d): Parallel=%.2f, Sequential=%.2f\n", i, j,
               C_parallel[i][j], C_sequential[i][j]);
        verified = 0;
      }
    }
  }

  /// Clean up
  for (int i = 0; i < N; i++) {
    free(A[i]);
    free(B[i]);
    free(C_parallel[i]);
    free(C_sequential[i]);
  }
  free(A);
  free(B);
  free(C_parallel);
  free(C_sequential);

  if (verified) {
    CARTS_TEST_PASS();
  } else {
    CARTS_TEST_FAIL("matrix multiplication mismatch");
  }
}