#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"

// #define DEBUG 0

int main(int argc, char **argv) {
  CARTS_TIMER_START();
  const int N = (argc >= 2) ? atoi(argv[1]) : 100;
  int *a = (int *)malloc(N * sizeof(int));
  int *b = (int *)malloc(N * sizeof(int));
  int *c = (int *)malloc(N * sizeof(int));

  /// Initialize arrays
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }
  /// Print initial values (debug only)
  CARTS_DEBUG_PRINT("Initial values:\n");
  for (int i = 0; i < N; i++)
    CARTS_DEBUG_PRINT("a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);

  /// Parallel region with worksharing loop - vector addition
#pragma omp parallel for
  for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];

  /// Results
  bool success = true;
  for (int i = 0; i < N; i++) {
    if (c[i] != a[i] + b[i]) {
      success = false;
      break;
    }
  }

  CARTS_DEBUG_PRINT("Results:\n");
  for (int i = 0; i < N; i++)
    CARTS_DEBUG_PRINT("c[%d] = %d, a[%d] = %d, b[%d] = %d\n", i, c[i], i, a[i],
                      i, b[i]);

  free(a);
  free(b);
  free(c);

  if (success)
    CARTS_TEST_PASS();
  else
    CARTS_TEST_FAIL("vector addition mismatch");
}
