#include <stdlib.h>

#include "arts/utils/testing/CartsTest.h"

#define NX 64
#define NY 64

static void rhs(int nx, int ny, double **f) {
  int i, j;

#pragma omp parallel for private(i)
  for (j = 0; j < ny; ++j) {
    for (i = 0; i < nx; ++i) {
      f[i][j] = (double)(i + j);
    }
  }
}

static void sweep(int nx, int ny, double **f, double **u, double **unew) {
  int i, j;

#pragma omp parallel for private(j)
  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      u[i][j] = unew[i][j];
    }
  }

#pragma omp parallel for private(j)
  for (i = 0; i < nx; ++i) {
    for (j = 0; j < ny; ++j) {
      if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        unew[i][j] = f[i][j];
      } else {
        unew[i][j] = 0.25 * (u[i - 1][j] + u[i][j + 1] + u[i][j - 1] +
                             u[i + 1][j] + f[i][j]);
      }
    }
  }
}

static int verify(int nx, int ny, double **unew) {
  const double eps = 1.0e-9;

  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      double expected = (double)(i + j);
      if (!(i == 0 || j == 0 || i == nx - 1 || j == ny - 1))
        expected *= 0.25;

      double diff = unew[i][j] - expected;
      if (diff < -eps || diff > eps)
        return 0;
    }
  }

  return 1;
}

int main(void) {
  CARTS_TIMER_START();

  double **f = (double **)malloc(NX * sizeof(double *));
  double **u = (double **)malloc(NX * sizeof(double *));
  double **unew = (double **)malloc(NX * sizeof(double *));

  for (int i = 0; i < NX; ++i) {
    f[i] = (double *)malloc(NY * sizeof(double));
    u[i] = (double *)malloc(NY * sizeof(double));
    unew[i] = (double *)malloc(NY * sizeof(double));
    for (int j = 0; j < NY; ++j)
      unew[i][j] = 0.0;
  }

  rhs(NX, NY, f);
  sweep(NX, NY, f, u, unew);
  int ok = verify(NX, NY, unew);

  for (int i = 0; i < NX; ++i) {
    free(f[i]);
    free(u[i]);
    free(unew[i]);
  }
  free(f);
  free(u);
  free(unew);

  if (ok)
    CARTS_TEST_PASS();
  CARTS_TEST_FAIL("mixed orientation mismatch");
}
