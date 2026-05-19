
#include <cstdlib>
#include <stdio.h>

#include <omp.h>
#include <stdlib.h>
#include <time.h>

#include "carts/utils/testing/CartsTest.h"

int main() {
  CARTS_TIMER_START();
  srand(time(NULL));
  printf("Main EDT: Starting...\n");

  int shared_number = rand() % 100 + 1;
  int random_number = rand() % 10 + 1;
  printf("EDT 0: The initial number is %d/%d\n", shared_number, random_number);

#pragma omp parallel
  {
#pragma omp single
    {
      printf("EDT 1: The number is %d/%d\n", shared_number, random_number);
#pragma omp task firstprivate(random_number)
      {
        shared_number++;
        random_number++;
        printf("EDT 2: The number is %d/%d\n", shared_number, random_number);
      }
    }
  }

  shared_number++;
  printf("EDT 3: The final number is %d - %d.\n", shared_number, random_number);
  CARTS_TEST_PASS();
}
