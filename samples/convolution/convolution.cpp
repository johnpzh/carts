
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <omp.h>

#include "carts/utils/testing/CartsTest.h"

void print_usage(const char *program_name) {
  printf("Usage: %s [options]\n", program_name);
  printf("Options:\n");
  printf("  -n <size>     Matrix size (NxN, default: 200)\n");
  printf("  -k <size>     Kernel size (KxK, default: 5)\n");
  printf("  -t <tasks>    Number of tasks (default: 4)\n");
  printf("  -h            Show this help\n");
}

int main(int argc, char **argv) {
  CARTS_TIMER_START();
  int matrix_size = 200;
  int kernel_size = 5;
  int num_tasks = 4;

  /// Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      matrix_size = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
      kernel_size = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
      num_tasks = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      printf("Unknown option: %s\n", argv[i]);
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  /// Validate parameters
  if (matrix_size <= 0 || kernel_size <= 0 || num_tasks <= 0) {
    printf("Error: All sizes must be positive\n");
    return EXIT_FAILURE;
  }

  if (kernel_size > matrix_size) {
    printf("Error: Kernel size cannot be larger than matrix size\n");
    return EXIT_FAILURE;
  }

  const int block_size = matrix_size / num_tasks;
  const int output_size = matrix_size - kernel_size + 1;
  const int A_size = matrix_size;

  printf("Configuration:\n");
  printf("  Matrix size: %dx%d\n", matrix_size, matrix_size);
  printf("  Kernel size: %dx%d\n", kernel_size, kernel_size);
  printf("  Number of tasks: %d\n", num_tasks);
  printf("  Block size: %d\n", block_size);
  printf("  Output size: %dx%d\n", output_size, output_size);
  printf("  Initializing with random numbers\n\n");

  /// Dynamic memory allocation
  double **A = new double *[matrix_size];
  double **Kernel = new double *[kernel_size];
  double **C_parallel = new double *[output_size];
  double **C_serial = new double *[output_size];

  for (int i = 0; i < matrix_size; ++i)
    A[i] = new double[matrix_size];
  for (int i = 0; i < kernel_size; ++i)
    Kernel[i] = new double[kernel_size];
  for (int i = 0; i < output_size; ++i) {
    C_parallel[i] = new double[output_size];
    C_serial[i] = new double[output_size];
  }

  /// Initialize matrices with random numbers
  printf("Initializing matrices with random numbers...\n");

  /// Seed random number generator
  srand(time(NULL));

  /// Initialize matrix A with random values between 0.0 and 1.0
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j)
      A[i][j] = (double)rand() / RAND_MAX;
  }

  /// Initialize kernel with random values between -1.0 and 1.0
  for (int i = 0; i < kernel_size; ++i) {
    for (int j = 0; j < kernel_size; ++j)
      Kernel[i][j] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
  }

  /// Initialize output matrices to zero
  for (int i = 0; i < output_size; ++i) {
    for (int j = 0; j < output_size; ++j) {
      C_parallel[i][j] = 0.0;
      C_serial[i][j] = 0.0;
    }
  }

  double t_start = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp single
    {
      printf("Starting tasks\n");
      for (int task_id = 0; task_id < num_tasks; ++task_id) {
        const int start_row = task_id * block_size;
#pragma omp task firstprivate(start_row, task_id)
        {
          printf("Task %d running\n", task_id + 1);
          for (int ii = 0; ii < block_size; ++ii) {
            const int i = start_row + ii;
            if (i < output_size) {
              for (int j = 0; j < output_size; ++j) {
                double tmp = 0.0;
                for (int k = 0; k < kernel_size; ++k) {
                  for (int l = 0; l < kernel_size; ++l) {
                    const int A_row = i + k;
                    const int A_col = j + l;
                    if (A_row < A_size && A_col < A_size)
                      tmp += A[A_row][A_col] * Kernel[k][l];
                  }
                }
                C_parallel[i][j] = tmp;
              }
            }
          }
        }
      }
    }
  }
  double t_end = omp_get_wtime();
  printf("Finished in %f seconds\n", t_end - t_start);

  /// Serial computation for verification
  for (int i = 0; i < output_size; ++i) {
    for (int j = 0; j < output_size; ++j) {
      double tmp = 0.0;
      for (int k = 0; k < kernel_size; ++k) {
        for (int l = 0; l < kernel_size; ++l) {
          const int A_row = i + k;
          const int A_col = j + l;
          if (A_row < A_size && A_col < A_size)
            tmp += A[A_row][A_col] * Kernel[k][l];
        }
      }
      C_serial[i][j] = tmp;
    }
  }

  /// Verification
  int errors = 0;
  const double tolerance = 1e-6;
  for (int i = 0; i < output_size; ++i) {
    for (int j = 0; j < output_size; ++j) {
      if (fabs(C_parallel[i][j] - C_serial[i][j]) > tolerance) {
        errors++;
        printf("Mismatch at (%d,%d): Parallel=%.2f vs Serial=%.2f\n", i, j,
               C_parallel[i][j], C_serial[i][j]);
      }
    }
  }

  /// Memory cleanup
  for (int i = 0; i < matrix_size; ++i)
    delete[] A[i];

  for (int i = 0; i < kernel_size; ++i)
    delete[] Kernel[i];

  for (int i = 0; i < output_size; ++i) {
    delete[] C_parallel[i];
    delete[] C_serial[i];
  }
  delete[] A;
  delete[] Kernel;
  delete[] C_parallel;
  delete[] C_serial;

  if (errors == 0)
    CARTS_TEST_PASS();
  else
    CARTS_TEST_FAIL("convolution verification failed");
}
