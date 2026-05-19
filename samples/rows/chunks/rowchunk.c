#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"

#define ROWS 1000
#define COLS 1000

/// Test example demonstrating 2D row-chunking optimization.
/// Uses array-of-arrays pattern (float **A) which is the common
/// allocation pattern in carts examples.
/// With N workers, the 2D array is chunked by rows: ROWS/N rows per chunk.
int main(int argc, char *argv[]) {
  CARTS_TIMER_START();

  /// Allocate using array-of-arrays pattern
  float **A = (float **)malloc(ROWS * sizeof(float *));
  float **B = (float **)malloc(ROWS * sizeof(float *));
  for (int i = 0; i < ROWS; i++) {
    A[i] = (float *)malloc(COLS * sizeof(float));
    B[i] = (float *)malloc(COLS * sizeof(float));
  }

  /// Initialize array A
  for (int i = 0; i < ROWS; i++)
    for (int j = 0; j < COLS; j++)
      A[i][j] = (float)(i * COLS + j);

  /// Row-parallel operation - should trigger row-chunking
  /// Each iteration processes one row independently
#pragma omp parallel for
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      B[i][j] = A[i][j] * 2.0f + 1.0f;
    }
  }

  /// Verify results
  int correct = 1;
  for (int i = 0; i < ROWS && correct; i++) {
    for (int j = 0; j < COLS && correct; j++) {
      float expected = A[i][j] * 2.0f + 1.0f;
      if (B[i][j] != expected) {
        printf("Mismatch at [%d][%d]: expected %f, got %f\n", i, j, expected,
               B[i][j]);
        correct = 0;
      }
    }
  }

  /// Free using array-of-arrays pattern
  for (int i = 0; i < ROWS; i++) {
    free(A[i]);
    free(B[i]);
  }
  free(A);
  free(B);

  if (correct) {
    CARTS_TEST_PASS();
  } else {
    CARTS_TEST_FAIL("matrix computation mismatch");
  }
}
