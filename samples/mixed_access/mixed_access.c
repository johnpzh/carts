/*
 * CARTS Mixed Access Pattern Test
 * Demonstrates LULESH-like access patterns with full-range chunked acquires:
 *
 * 1. Node-parallel loop: Direct write nodeData[i] = ... (CHUNKED partitioning)
 *    - Workers write contiguous chunks of node data
 *    - Each worker acquires only its chunk for writing
 *
 * 2. Element-parallel loop: Indirect read nodeData[nodelist[e][j]] (FULL-RANGE)
 *    - Each element reads scattered nodes via indirection array
 *    - Workers acquire ALL chunks (full-range) for indirect reads
 *    - This avoids DbCopy/DbSync overhead while maintaining correctness
 *
 * The H1.2 heuristic detects this pattern and enables:
 * - Chunked allocation for the data array
 * - Full-range acquires for indirect read operations
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"

#define NODES_PER_ELEM 4

static int **AllocateInt2D(int rows, int cols) {
  int **arr = (int **)malloc(rows * sizeof(int *));
  for (int i = 0; i < rows; i++)
    arr[i] = (int *)malloc(cols * sizeof(int));
  return arr;
}

static void FreeInt2D(int **arr, int rows) {
  for (int i = 0; i < rows; i++)
    free(arr[i]);
  free(arr);
}

// Sequential version for verification
static void compute_seq(int numNodes, int numElems, double *nodeData,
                        double *elemData, int **nodelist) {
  /// Phase 1: Initialize node data (direct access)
  for (int i = 0; i < numNodes; i++)
    nodeData[i] = (double)(i * i) * 0.01;

  /// Phase 2: Compute element values from nodes (indirect access)
  for (int e = 0; e < numElems; e++) {
    double sum = 0.0;
    for (int j = 0; j < NODES_PER_ELEM; j++) {
      int nodeIdx = nodelist[e][j];
      sum += nodeData[nodeIdx];
    }
    elemData[e] = sum / NODES_PER_ELEM;
  }
}

// Parallel version with #pragma omp parallel for
static void compute_parallel(int numNodes, int numElems, double *nodeData,
                             double *elemData, int **nodelist) {
  /// Phase 1: Node-parallel initialization - CHUNKED access
#pragma omp parallel for
  for (int i = 0; i < numNodes; i++)
    nodeData[i] = (double)(i * i) * 0.01;

  /// Phase 2: Element-parallel computation - INDIRECT access
  /// Each element reads scattered nodes via nodelist indirection
  /// Workers acquire full-range (all chunks) for indirect access
#pragma omp parallel for
  for (int e = 0; e < numElems; e++) {
    double sum = 0.0;
    for (int j = 0; j < NODES_PER_ELEM; j++) {
      int nodeIdx = nodelist[e][j];
      sum += nodeData[nodeIdx];
    }
    elemData[e] = sum / NODES_PER_ELEM;
  }
}

// Build a simple 2D quad mesh connectivity
// Elements are quads with 4 nodes each
static void build_mesh(int nx, int ny, int **nodelist) {
  int numNodesX = nx + 1;
  for (int ey = 0; ey < ny; ey++) {
    for (int ex = 0; ex < nx; ex++) {
      int elem = ey * nx + ex;
      /// Counter-clockwise node ordering for quad element
      nodelist[elem][0] = ey * numNodesX + ex;           /// bottom-left
      nodelist[elem][1] = ey * numNodesX + ex + 1;       /// bottom-right
      nodelist[elem][2] = (ey + 1) * numNodesX + ex + 1; /// top-right
      nodelist[elem][3] = (ey + 1) * numNodesX + ex;     /// top-left
    }
  }
}

int main(void) {
  CARTS_TIMER_START();

#ifdef SIZE
  int nx = SIZE, ny = SIZE;
#else
  int nx = 50, ny = 50;
#endif

  int numNodesX = nx + 1;
  int numNodesY = ny + 1;
  int numNodes = numNodesX * numNodesY;
  int numElems = nx * ny;

  printf("Mixed Access Test: %d nodes, %d elements\n", numNodes, numElems);
  printf("Pattern: chunked node writes + full-range indirect reads\n");

  /// Allocate arrays
  double *nodeData = (double *)malloc(numNodes * sizeof(double));
  double *nodeData_seq = (double *)malloc(numNodes * sizeof(double));
  double *elemData = (double *)malloc(numElems * sizeof(double));
  double *elemData_seq = (double *)malloc(numElems * sizeof(double));
  int **nodelist = AllocateInt2D(numElems, NODES_PER_ELEM);

  /// Initialize
  for (int i = 0; i < numNodes; i++) {
    nodeData[i] = 0.0;
    nodeData_seq[i] = 0.0;
  }
  for (int e = 0; e < numElems; e++) {
    elemData[e] = 0.0;
    elemData_seq[e] = 0.0;
  }

  /// Build mesh connectivity
  build_mesh(nx, ny, nodelist);

  /// Run sequential version for verification
  printf("Running sequential version...\n");
  compute_seq(numNodes, numElems, nodeData_seq, elemData_seq, nodelist);

  /// Run parallel version
  printf("Running parallel version...\n");
  compute_parallel(numNodes, numElems, nodeData, elemData, nodelist);

  /// Compare results
  double error = 0.0;
  double max_error = 0.0;
  for (int e = 0; e < numElems; e++) {
    double diff = elemData_seq[e] - elemData[e];
    error += diff * diff;
    if (fabs(diff) > max_error) {
      max_error = fabs(diff);
    }
  }
  error = sqrt(error / numElems);

  printf("RMS error: %e\n", error);
  printf("Max error: %e\n", max_error);

  /// Free arrays
  free(nodeData);
  free(nodeData_seq);
  free(elemData);
  free(elemData_seq);
  FreeInt2D(nodelist, numElems);

  if (error < 1e-10)
    CARTS_TEST_PASS();
  else
    CARTS_TEST_FAIL("mixed_access verification failed");
}
