#include <cstdio>
#include <cstdlib>

#include <omp.h>
#include <time.h>

#include "carts/utils/testing/CartsTest.h"

int main() {
  CARTS_TIMER_START();
  srand(time(NULL));
  printf("Main EDT: Starting concurrent dynamic memory example...\n");

  const int data_size = 10;
  const double base_value = (rand() % 50) + 1.5;

  printf("Main EDT: Base value = %.2f, Data size = %d\n", base_value,
         data_size);

  /// Allocate separate data arrays for each task outside parallel region
  double *task1_data = (double *)malloc(data_size * sizeof(double));
  double *task2_data = (double *)malloc(data_size * sizeof(double));
  printf("Main EDT: Allocated separate data arrays for each task\n");

  /// Initialize task1 data
  for (int i = 0; i < data_size; ++i) {
    task1_data[i] = base_value + (double)i;
    task2_data[i] = base_value + 100.0 + (double)i;
  }

  /// Print initial values
  printf("\nMain EDT: Initial Array Values:\n");
  printf("=====================================\n");
  printf("%-8s %-12s %-12s\n", "Index", "Task1_Data", "Task2_Data");
  printf("-------------------------------------\n");
  for (int i = 0; i < data_size; ++i)
    printf("%-8d %-12.2f %-12.2f\n", i, task1_data[i], task2_data[i]);

  printf("=====================================\n");

#pragma omp parallel
  {
#pragma omp single
    {
      printf("Creating Task 1...\n");
#pragma omp task
      {
        printf("Task 1: Thread ID = %d\n", omp_get_thread_num());

        /// Process task1 data
        double sum = 0.0;
        for (int i = 0; i < data_size; ++i) {
          task1_data[i] *= 2.0;
          sum += task1_data[i];
          printf("Task 1: task1_data[%d] = %.2f\n", i, task1_data[i]);
        }

        printf("Task 1: Sum = %.2f\n", sum);
      }

      printf("Creating Task 2...\n");
#pragma omp task
      {
        printf("Task 2: Thread ID = %d\n", omp_get_thread_num());

        /// Process task2 data
        double product = 1.0;
        for (int i = 0; i < data_size; ++i) {
          task2_data[i] += 5.0;
          double mod_val = task2_data[i] - (int)(task2_data[i] / 10.0) * 10.0;
          product *= (mod_val + 1.0);
          printf("Task 2: task2_data[%d] = %.2f\n", i, task2_data[i]);
        }

        printf("Task 2: Product = %.2f\n", product);
      }

      printf("Both tasks created asynchronously\n");
    }
  }

  /// Print final values
  printf("\nMain EDT: Final Array Values:\n");
  printf("=====================================\n");
  printf("%-8s %-12s %-12s\n", "Index", "Task1_Data", "Task2_Data");
  printf("-------------------------------------\n");

  for (int i = 0; i < data_size; ++i)
    printf("%-8d %-12.2f %-12.2f\n", i, task1_data[i], task2_data[i]);

  printf("=====================================\n");
  printf("Main EDT: All concurrent tasks completed. Program finished.\n");

  free(task1_data);
  free(task2_data);
  CARTS_TEST_PASS();
}
