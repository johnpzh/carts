/*
 * CARTS Jacobi-For Test Example
 * Demonstrates two distinct data access patterns:
 * 1. Loop 1: Uniform access (copy) - each worker writes only its assigned rows
 * 2. Loop 2: Stencil computation - neighbor access (reads u[i-1], u[i], u[i+1])
 *
 * Original: https://github.com/viroulep/kastors
 * License: GNU LGPL
 * Author: John Burkardt (modified by KaStORS team)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"

/// Sequential version for verification
static void sweep_seq(int nx, int ny, double dx, double dy, double **f,
                      int itold, int itnew, double **u, double **unew) {
  for (int it = itold + 1; it <= itnew; it++) {
    /// Save the current estimate
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++)
        u[i][j] = unew[i][j];
    }

    /// Compute a new estimate
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
          unew[i][j] = f[i][j];
        } else {
          unew[i][j] = 0.25 * (u[i - 1][j] + u[i][j + 1] + u[i][j - 1] +
                               u[i + 1][j] + f[i][j] * dx * dy);
        }
      }
    }
  }
}

/// Parallel version with #pragma omp parallel for
static void sweep(int nx, int ny, double dx, double dy, double **f, int itold,
                  int itnew, double **u, double **unew, int block_size) {
  int i, j, it;

  for (it = itold + 1; it <= itnew; it++) {
    /// LOOP 1: Save the current estimate - UNIFORM access pattern
    /// Each worker writes ONLY its assigned rows (no halo needed)
#pragma omp parallel for private(j)
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++)
        u[i][j] = unew[i][j];
    }

    /// LOOP 2: Compute a new estimate - STENCIL access pattern
    /// Each worker reads its rows PLUS neighboring rows (halo needed)
#pragma omp parallel for private(j)
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
          unew[i][j] = f[i][j];
        } else {
          /// 5-point stencil: reads u[i-1], u[i], u[i+1]
          unew[i][j] = 0.25 * (u[i - 1][j] + u[i][j + 1] + u[i][j - 1] +
                               u[i + 1][j] + f[i][j] * dx * dy);
        }
      }
    }
  }
}

int main(void) {
  CARTS_TIMER_START();

#ifdef SIZE
  int nx = SIZE, ny = SIZE;
#else
  int nx = 100, ny = 100;
#endif
  int itold = 0, itnew = 10;
  int block_size = 10;
  double dx = 1.0 / (nx - 1);
  double dy = 1.0 / (ny - 1);

  printf("Jacobi-For Test: %d x %d grid, %d iterations\n", nx, ny, itnew);
  printf(
      "Demonstrating uniform (Loop 1) vs stencil (Loop 2) access patterns\n");

  /// Allocate 2D arrays
  double **f = (double **)malloc(nx * sizeof(double *));
  double **u = (double **)malloc(nx * sizeof(double *));
  double **unew = (double **)malloc(nx * sizeof(double *));
  double **unew_seq = (double **)malloc(nx * sizeof(double *));

  for (int i = 0; i < nx; i++) {
    f[i] = (double *)malloc(ny * sizeof(double));
    u[i] = (double *)malloc(ny * sizeof(double));
    unew[i] = (double *)malloc(ny * sizeof(double));
    unew_seq[i] = (double *)malloc(ny * sizeof(double));
  }

  /// Initialize arrays
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      f[i][j] = 0.0;
      u[i][j] = 0.0;
      unew[i][j] = 0.0;
      unew_seq[i][j] = 0.0;
    }
  }

  /// Run sequential version for verification
  printf("Running sequential version for verification...\n");
  sweep_seq(nx, ny, dx, dy, f, itold, itnew, u, unew_seq);

  /// Save sequential result
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++)
      unew_seq[i][j] = unew[i][j];
  }

  /// Re-initialize for parallel version
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      u[i][j] = 0.0;
      unew[i][j] = 0.0;
    }
  }

  /// Run parallel version
  printf("Running parallel version with #pragma omp parallel for...\n");
  sweep(nx, ny, dx, dy, f, itold, itnew, u, unew, block_size);

  /// Compare results
  double error = 0.0;
  double max_error = 0.0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      double diff = unew_seq[i][j] - unew[i][j];
      error += diff * diff;
      if (fabs(diff) > max_error)
        max_error = fabs(diff);
    }
  }
  error = sqrt(error / (nx * ny));

  printf("RMS error: %e\n", error);
  printf("Max error: %e\n", max_error);

  /// Free 2D arrays
  for (int i = 0; i < nx; i++) {
    free(f[i]);
    free(u[i]);
    free(unew[i]);
    free(unew_seq[i]);
  }
  free(f);
  free(u);
  free(unew);
  free(unew_seq);

  if (error < 1e-6)
    CARTS_TEST_PASS();
  else
    CARTS_TEST_FAIL("jacobi-for verification failed");
}
