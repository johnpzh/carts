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

  int randomNumber = rand() % 100;
  /// Initialize arrays
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = (i * 2) + randomNumber;
  }

  /// Print initial values (debug only)
  CARTS_DEBUG_PRINT("Initial values:\n");
  for (int i = 0; i < N; i++)
    CARTS_DEBUG_PRINT("a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);

  /// Parallel region with worksharing loop - vector addition
  printf("Parallel region:\n");
#pragma omp parallel for schedule(static, 4)
  for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];

  printf("Parallel region completed\n");

  /// Verify results
  int sum = 0;
#pragma omp parallel
  {
#pragma omp for reduction(+ : sum)
    for (int i = 0; i < N; i++)
      sum += c[i];
  }

  printf("Sum of results: %d\n", sum);
  int expectedSum = N * (N - 1) / 2 + N * (N - 1) + N * randomNumber;
  printf("Expected sum: %d\n", expectedSum);

  free(a);
  free(b);
  free(c);

  if (sum == expectedSum)
    CARTS_TEST_PASS();
  else
    CARTS_TEST_FAIL("sum mismatch");
}
