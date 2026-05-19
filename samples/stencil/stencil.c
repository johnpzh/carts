#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "carts/utils/testing/CartsTest.h"

/// #define DEBUG 0
/// 1D stencil computation: output[i] = input[i-1] + input[i] + input[i+1]
int main(int argc, char *argv[]) {
  CARTS_TIMER_START();
  int N = (argc >= 2) ? atoi(argv[1]) : 100;
  if (N < 3) {
    printf("N must be at least 3\n");
    return 1;
  }

  int *input = (int *)malloc(N * sizeof(int));
  int *output = (int *)malloc(N * sizeof(int));

  /// Initialize
  for (int i = 0; i < N; i++) {
    input[i] = i + 1;
    output[i] = 0;
  }

  CARTS_DEBUG_PRINT("1D Stencil: N=%d\n", N);

#pragma omp parallel
  {
#pragma omp single
    {
      /// Each task computes output[i] from input[i-1:i+2] (3-point stencil)
      /// Tasks can run in parallel as long as their read ranges don't overlap
      /// with write locations
      for (int i = 1; i < N - 1; i++) {
        /// This task reads input[i-1], input[i], input[i+1]
        /// and writes to output[i]
#pragma omp task depend(in : input[i - 1], input[i], input[i + 1])             \
    depend(out : output[i])
        {
          CARTS_DEBUG_PRINT("Task %d: Computing output[%d] from input[%d:%d]\n",
                            i, i, i - 1, i + 2);
          output[i] = input[i - 1] + input[i] + input[i + 1];
          CARTS_DEBUG_PRINT("Task %d: output[%d] = %d + %d + %d = %d\n", i, i,
                            input[i - 1], input[i], input[i + 1], output[i]);
        }
      }
    }
  }

  /// Verify results
  int correct = 1;
  for (int i = 1; i < N - 1; i++) {
    /// = 3*i + 3
    int expected = (i - 1 + 1) + (i + 1) + (i + 1 + 1);
    if (output[i] != expected) {
      correct = 0;
      break;
    }
  }

  free(input);
  free(output);

  if (correct)
    CARTS_TEST_PASS();
  else
    CARTS_TEST_FAIL("stencil computation mismatch");
}
