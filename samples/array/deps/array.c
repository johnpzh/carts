#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "carts/utils/testing/CartsTest.h"
int main(int argc, char *argv[]) {
  CARTS_TIMER_START();
  const int N = (argc >= 2) ? atoi(argv[1]) : 100;
  int *A = (int *)malloc(N * sizeof(int));
  int *B = (int *)malloc(N * sizeof(int));
  srand(time(NULL));

  CARTS_DEBUG_PRINT("Initializing arrays A and B with size %d\n", N);

#pragma omp parallel
  {
#pragma omp single
    {
      for (int i = 0; i < N; i++) {
#pragma omp task depend(inout : A[i])
        {
          A[i] = i;
          CARTS_DEBUG_PRINT("Task %d - 0: Initializing A[%d] = %d\n", i, i,
                            A[i]);
        }

        if (i == 0) {
#pragma omp task depend(in : A[i]) depend(inout : B[i])
          {
            CARTS_DEBUG_PRINT("Task %d - 1 -> Input: A[%d] = %d\n", i, i, A[i]);
            B[i] = A[i] + 5;
            CARTS_DEBUG_PRINT("Task %d - 1: Computing B[%d] = %d\n", i, i,
                              B[i]);
          }
        } else {
#pragma omp task depend(in : A[i]) depend(in : B[i - 1]) depend(inout : B[i])
          {
            CARTS_DEBUG_PRINT("Task %d - 2 -> Input: A[%d] = %d, B[%d] = %d\n",
                              i, i, A[i], i - 1, B[i - 1]);
            B[i] = A[i] + B[i - 1] + 5;
            CARTS_DEBUG_PRINT("Task %d - 2: Computing B[%d] = %d\n", i, i,
                              B[i]);
          }
        }
      }
    }
  }

  CARTS_DEBUG_PRINT("Final arrays:\n");
  for (int i = 0; i < N; i++)
    CARTS_DEBUG_PRINT("A[%d] = %d, B[%d] = %d\n", i, A[i], i, B[i]);

  /// Verify: A[i] = i, B[i] = sum(A[0..i]) + 5*(i+1)
  int correct = 1;
  for (int i = 0; i < N; i++) {
    int expected_B = (i == 0) ? A[0] + 5 : A[i] + B[i - 1] + 5;
    /// Check iteratively: B[i] = i*(i+1)/2 + 5*(i+1)
    int sum_A = i * (i + 1) / 2;
    expected_B = sum_A + 5 * (i + 1);
    if (A[i] != i || B[i] != expected_B) {
      correct = 0;
      break;
    }
  }

  free(A);
  free(B);

  if (correct)
    CARTS_TEST_PASS();
  else
    CARTS_TEST_FAIL("array computation mismatch");
}
