#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"

int main() {
  CARTS_TIMER_START();
  printf("Parallel region execution - Number of threads: %d\n",
         omp_get_num_threads());
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    printf("Hello from thread %d of %d\n", tid, num_threads);
  }

  printf("Parallel region finished.\n");

  CARTS_TEST_PASS();
}
