#include <stdio.h>
#include <stdlib.h>

#include "carts/utils/testing/CartsTest.h"

int main(int argc, char **argv) {
  CARTS_TIMER_START();
  int N = (argc >= 2) ? atoi(argv[1]) : 100;
  if (N < 3) {
    printf("Size must be >= 3\n");
    return 1;
  }

  int *in = (int *)malloc(N * sizeof(int));
  int *out = (int *)malloc(N * sizeof(int));

  for (int i = 0; i < N; ++i)
    in[i] = i;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; ++i) {
    int left = in[i];
    if (i > 0)
      left = in[i - 1];
    int center = in[i];
    int right = in[i];
    if (i + 1 < N)
      right = in[i + 1];
    out[i] = left + center + right;
  }

  int checksum = 0;
  for (int i = 0; i < N; ++i)
    checksum += out[i];

  int expected_checksum = 3 * N * (N - 1) / 2;
  printf("Checksum: %d\n", checksum);
  printf("Expected checksum: %d\n", expected_checksum);

  free(in);
  free(out);

  if (checksum == expected_checksum) {
    CARTS_TEST_PASS();
  } else {
    CARTS_TEST_FAIL("checksum mismatch");
  }
}
