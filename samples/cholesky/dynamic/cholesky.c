#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "carts/utils/testing/CartsTest.h"

#define BLOCK_SIZE 16
#define TOLERANCE 10e-6

int main(int argc, char *argv[]) {
  CARTS_TIMER_START();
  const unsigned N = (argc >= 2) ? atoi(argv[1]) : 64;
  srand(time(NULL));

  double **A = (double **)malloc(N * sizeof(double *));
  double **L_parallel = (double **)malloc(N * sizeof(double *));
  double **L_sequential = (double **)malloc(N * sizeof(double *));
  double **L_orig = (double **)malloc(N * sizeof(double *));

  for (unsigned i = 0; i < N; i++) {
    A[i] = (double *)malloc(N * sizeof(double));
    L_parallel[i] = (double *)malloc(N * sizeof(double));
    L_sequential[i] = (double *)malloc(N * sizeof(double));
    L_orig[i] = (double *)malloc(N * sizeof(double));
  }

  /// Generate a positive-definite matrix A = L_orig * L_orig^T
  for (unsigned i = 0; i < N; i++) {
    for (unsigned j = 0; j < N; j++) {
      /// Avoid zeros on off-diagonal
      if (j <= i)
        L_orig[i][j] = (double)rand() / RAND_MAX + 0.01;
      /// L_orig is lower triangular
      else
        L_orig[i][j] = 0.0;
    }
    /// Strengthen diagonal for positive definiteness
    L_orig[i][i] += 1.0;
  }

  for (unsigned i = 0; i < N; i++) {
    /// A is symmetric, compute its lower triangle
    for (unsigned j = 0; j <= i; j++) {
      A[i][j] = 0.0;
      for (unsigned k_sum = 0; k_sum <= j; k_sum++)
        A[i][j] += L_orig[i][k_sum] * L_orig[j][k_sum];
    }
  }

  /// Fill upper triangle of A by symmetry
  for (unsigned i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++)
      A[i][j] = A[j][i];
  }

  /// Initialize matrices for Cholesky decomposition
  for (unsigned i = 0; i < N; i++) {
    for (unsigned j = 0; j < N; j++) {
      L_parallel[i][j] = A[i][j];
      L_sequential[i][j] = A[i][j];
    }
  }

  /// Parallel Cholesky Decomposition
  double start_time = omp_get_wtime();
#pragma omp parallel
#pragma omp single
  {
    for (unsigned k_block = 0; k_block < N; k_block += BLOCK_SIZE) {
      unsigned kb = (k_block + BLOCK_SIZE > N) ? N - k_block : BLOCK_SIZE;

      /// Task 1: Factorize diagonal block (POTRF-like operation)
#pragma omp task depend(inout : L_parallel[k_block][k_block])
      {
        for (unsigned i = k_block; i < k_block + kb; i++) {
          for (unsigned j = k_block; j < i; j++)
            L_parallel[i][i] -= L_parallel[i][j] * L_parallel[i][j];

          /// Avoid sqrt of negative due to floating point inaccuracies
          if (L_parallel[i][i] < 0.0)
            L_parallel[i][i] = 0.0;
          L_parallel[i][i] = sqrt(L_parallel[i][i]);

          for (unsigned j = i + 1; j < k_block + kb; j++) {
            for (unsigned m = k_block; m < i; m++)
              L_parallel[j][i] -= L_parallel[j][m] * L_parallel[i][m];

            /// Handle division by zero if diagonal element L_ii is zero
            if (L_parallel[i][i] != 0.0)
              L_parallel[j][i] /= L_parallel[i][i];
            /// Set to 0 to prevent NaN; indicates numerical issues
            else
              L_parallel[j][i] = 0.0;
          }
        }
      }

      /// Task 2: Update panel of blocks below diagonal (TRSM-like operation)
      for (unsigned j_block = k_block + kb; j_block < N;
           j_block += BLOCK_SIZE) {
        unsigned jb = (j_block + BLOCK_SIZE > N) ? N - j_block : BLOCK_SIZE;
#pragma omp task depend(in : L_parallel[k_block][k_block])                     \
    depend(inout : L_parallel[j_block][k_block])
        {
          /// Column index in L_kk and panel L_jk
          for (unsigned col_k = k_block; col_k < k_block + kb; col_k++) {
            /// Row index in panel L_jk
            for (unsigned row_j = j_block; row_j < j_block + jb; row_j++) {
              for (unsigned m = k_block; m < col_k; m++) {
                L_parallel[row_j][col_k] -=
                    L_parallel[row_j][m] * L_parallel[col_k][m];
              }
              /// Avoid division by zero
              if (L_parallel[col_k][col_k] != 0.0)
                L_parallel[row_j][col_k] /= L_parallel[col_k][col_k];
              /// Avoid NaN; indicates numerical issues
              else
                L_parallel[row_j][col_k] = 0.0;
            }
          }
        }
      }

      /// Task 3: Update trailing submatrix (SYRK/GEMM-like operation)
      for (unsigned j_block_trail = k_block + kb; j_block_trail < N;
           j_block_trail += BLOCK_SIZE) {
        int jb_trail =
            (j_block_trail + BLOCK_SIZE > N) ? N - j_block_trail : BLOCK_SIZE;
        for (unsigned i_block_trail = j_block_trail; i_block_trail < N;
             i_block_trail += BLOCK_SIZE) {
          int ib_trail =
              (i_block_trail + BLOCK_SIZE > N) ? N - i_block_trail : BLOCK_SIZE;
#pragma omp task depend(in : L_parallel[i_block_trail][k_block],               \
                            L_parallel[j_block_trail][k_block])                \
    depend(inout : L_parallel[i_block_trail][j_block_trail])
          {
            for (unsigned row_target = i_block_trail;
                 row_target < i_block_trail + ib_trail; row_target++) {
              for (unsigned col_target = j_block_trail;
                   col_target < j_block_trail + jb_trail; col_target++) {
                /// Update only lower triangular part of the target block
                if (row_target >= col_target) {
                  double sum_update = 0.0;
                  for (unsigned m_sum = k_block; m_sum < k_block + kb;
                       m_sum++) {
                    sum_update += L_parallel[row_target][m_sum] *
                                  L_parallel[col_target][m_sum];
                  }
                  L_parallel[row_target][col_target] -= sum_update;
                }
              }
            }
          }
        }
      }
    } /// End k_block loop
  } /// End OpenMP single / parallel region
  double parallel_time = omp_get_wtime() - start_time;
  printf("Finished in %f seconds\n", parallel_time);

  /// Sequential Cholesky Decomposition
  start_time = omp_get_wtime();
  for (unsigned k = 0; k < N; k++) {
    /// Avoid sqrt of negative
    if (L_sequential[k][k] < 0.0)
      L_sequential[k][k] = 0.0;
    L_sequential[k][k] = sqrt(L_sequential[k][k]);

    if (L_sequential[k][k] != 0.0) {
      for (unsigned i = k + 1; i < N; i++)
        L_sequential[i][k] /= L_sequential[k][k];
    } else {
      /// L_kk is zero, avoid division by zero for elements L_ik
      for (unsigned i = k + 1; i < N; i++)
        L_sequential[i][k] = 0.0;
    }

    for (unsigned j = k + 1; j < N; j++) {
      for (unsigned i = j; i < N; i++) {
        L_sequential[i][j] -= L_sequential[i][k] * L_sequential[j][k];
      }
    }
  }
  double sequential_time = omp_get_wtime() - start_time;

  /// Verify results: Check if A = L * L^T
  int parallel_correct = 1;
  int sequential_correct = 1;
  double max_par_err = 0.0, max_seq_err = 0.0;

  for (unsigned i = 0; i < N; i++) {
    /// Check lower triangle
    for (unsigned j = 0; j <= i; j++) {
      double sum = 0.0;
      for (unsigned k_sum = 0; k_sum <= j; k_sum++)
        sum += L_parallel[i][k_sum] * L_parallel[j][k_sum];
      double err = fabs(A[i][j] - sum);
      if (err > max_par_err)
        max_par_err = err;
      if (err > TOLERANCE * fabs(A[i][j]) && err > TOLERANCE)
        parallel_correct = 0;
    }
  }

  for (unsigned i = 0; i < N; i++) {
    /// Check lower triangle
    for (unsigned j = 0; j <= i; j++) {
      double sum = 0.0;
      for (unsigned k_sum = 0; k_sum <= j; k_sum++)
        sum += L_sequential[i][k_sum] * L_sequential[j][k_sum];
      double err = fabs(A[i][j] - sum);
      if (err > max_seq_err)
        max_seq_err = err;
      if (err > TOLERANCE * fabs(A[i][j]) && err > TOLERANCE)
        sequential_correct = 0;
    }
  }

  for (unsigned i = 0; i < N; i++) {
    free(A[i]);
    free(L_parallel[i]);
    free(L_sequential[i]);
    free(L_orig[i]);
  }
  free(A);
  free(L_parallel);
  free(L_sequential);
  free(L_orig);

  if (parallel_correct) {
    CARTS_TEST_PASS();
  } else {
    CARTS_TEST_FAIL("cholesky verification failed");
  }
}