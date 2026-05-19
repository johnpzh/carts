#include <omp.h>
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
  int *d = (int *)malloc(N * sizeof(int));

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
    c[i] = 0;
    d[i] = 0;
  }

  CARTS_DEBUG_PRINT("Initial values:\n");
  for (int i = 0; i < N; i++)
    CARTS_DEBUG_PRINT("a[%d] = %d, b[%d] = %d, c[%d] = %d, d[%d] = %d\n", i,
                      a[i], i, b[i], i, c[i], i, d[i]);

  /// Single parallel region with back-to-back for loops
#pragma omp parallel
  {
#pragma omp for schedule(static, 4)
    for (int i = 0; i < N; i++)
      c[i] = a[i] + b[i];

#pragma omp for schedule(static, 4)
    for (int i = 0; i < N; i++)
      d[i] = c[i] * 2;
  }

  bool success = true;
  for (int i = 0; i < N; i++) {
    if (c[i] != a[i] + b[i] || d[i] != 2 * (a[i] + b[i])) {
      success = false;
      break;
    }
  }

  free(a);
  free(b);
  free(c);
  free(d);

  if (success) {
    CARTS_TEST_PASS();
  } else {
    CARTS_TEST_FAIL("computation mismatch");
  }
}
