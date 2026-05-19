#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"

int main() {
  CARTS_TIMER_START();
  int sum = 0;
  int data[100];

  /// Initialize array
  for (int i = 0; i < 100; i++)
    data[i] = i;

  printf("Starting parallel region with single and for constructs.\n");

#pragma omp parallel
  {
#pragma omp single
    {
      int tid = omp_get_thread_num();
      printf("Single region executed by thread %d\n", tid);
      sum += 1000;
    }
#pragma omp for
    for (int i = 0; i < 100; i++)
      data[i] = data[i] * 2;
  }

  printf("Parallel region finished.\n");
  printf("Sum value: %d\n", sum);
  printf("First element: %d, Last element: %d\n", data[0], data[99]);

  /// Verify: sum should be 1000, data[i] = i*2
  if (sum == 1000 && data[0] == 0 && data[99] == 198)
    CARTS_TEST_PASS();
  else
    CARTS_TEST_FAIL("verification failed");
}
