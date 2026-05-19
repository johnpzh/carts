///==========================================================================///
/// File: array.c
///
/// Test for chunk-style OpenMP dependencies.
///==========================================================================///

#include "carts/utils/testing/CartsTest.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 4

int main(int argc, char *argv[]) {
  CARTS_TIMER_START();
  const unsigned N = (argc >= 2) ? atoi(argv[1]) : 64;
  if (N % BLOCK_SIZE != 0) {
    printf("Error: N=%d must be a multiple of BLOCK_SIZE=%d\n", N, BLOCK_SIZE);
    exit(EXIT_FAILURE);
  }

  double *A = (double *)malloc(N * sizeof(double));
  double *B = (double *)malloc(N * sizeof(double));

  /// Initialize A
  for (unsigned i = 0; i < N; i++)
    A[i] = i * 1.0;

#pragma omp parallel
  {
#pragma omp single
    {
      /// Process in chunks - each task handles BLOCK_SIZE elements
      for (unsigned i = 0; i < N; i += BLOCK_SIZE) {
        /// Chunk dependency: depend on A[i:BLOCK_SIZE] not A[i]
#pragma omp task firstprivate(i) depend(in : A[i : BLOCK_SIZE])                \
    depend(out : B[i : BLOCK_SIZE])
        {
          for (unsigned j = 0; j < BLOCK_SIZE; j++)
            B[i + j] = A[i + j] * 2.0;
        }
      }
    }
  }

  /// Verify
  int correct = 1;
  for (unsigned i = 0; i < N; i++) {
    if (B[i] != A[i] * 2.0) {
      printf("Mismatch at B[%u]: expected %f, got %f\n", i, A[i] * 2.0, B[i]);
      correct = 0;
      break;
    }
  }

  free(A);
  free(B);

  if (correct)
    CARTS_TEST_PASS();
  else
    CARTS_TEST_FAIL("chunked array verification failed");
}
