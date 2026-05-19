#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"
// #define DEBUG 0

int main(int argc, char **argv) {
  CARTS_TIMER_START();
  const int N = (argc >= 2) ? atoi(argv[1]) : 100;
  if (N <= 0) {
    printf("Error: N must be a positive integer.\n");
    return 1;
  }
  int *a = (int *)malloc(N * sizeof(int));
  int *b = (int *)malloc(N * sizeof(int));
  long long dot_product = 0;
  int i;

  /// Initialize arrays
  for (i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  CARTS_DEBUG_PRINT("Performing parallel dot product...\n");

  double t_start = omp_get_wtime();
/// Parallel loop for dot product calculation
#pragma omp parallel for reduction(+ : dot_product)
  for (i = 0; i < N; i++)
    dot_product += (long long)a[i] * b[i];

  double t_end = omp_get_wtime();
  printf("Finished in %f seconds\n", t_end - t_start);
  printf("Parallel dot product finished.\n");
  printf("Dot product: %lld\n", dot_product);

  /// Verify results
  long long expected_dot_product = 0;
  for (i = 0; i < N; i++)
    expected_dot_product += (long long)a[i] * b[i];

  free(a);
  free(b);

  if (dot_product == expected_dot_product) {
    CARTS_TEST_PASS();
  } else {
    CARTS_TEST_FAIL("dot product mismatch");
  }
}
