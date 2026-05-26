# CTest registration for CARTS lit test suites.
#
# Usage:
#   ctest -L pass     # Run co-located pass tests only
#   ctest             # Run all registered test suites
#
include(carts_lit_test)

# Co-located pass tests (IREE-style: test/ dirs adjacent to pass source).
carts_lit_test(NAME carts-sde-tests
  TEST_DIR ${CMAKE_SOURCE_DIR}/lib/carts/dialect/sde/test
  LABELS pass
  DEPENDS carts-compile)

carts_lit_test(NAME carts-core-tests
  TEST_DIR ${CMAKE_SOURCE_DIR}/lib/carts/dialect/arts/test
  LABELS pass
  DEPENDS carts-compile)

carts_lit_test(NAME carts-rt-tests
  TEST_DIR ${CMAKE_SOURCE_DIR}/lib/carts/dialect/arts-rt/test
  LABELS pass
  DEPENDS carts-compile)

# Cross-cutting tests.
carts_lit_test(NAME carts-cli-tests
  TEST_DIR ${CMAKE_SOURCE_DIR}/tests/cli
  LABELS pass
  DEPENDS carts-compile)

carts_lit_test(NAME carts-verify-tests
  TEST_DIR ${CMAKE_SOURCE_DIR}/tests/verify
  LABELS pass
  DEPENDS carts-compile)

carts_lit_test(NAME carts-attrs-tests
  TEST_DIR ${CMAKE_SOURCE_DIR}/tests/attrs
  LABELS pass
  DEPENDS carts-compile)

# E2E tests: compile + run samples.
carts_lit_test(NAME carts-e2e-tests
  TEST_DIR ${CMAKE_SOURCE_DIR}/tests/e2e
  LABELS e2e
  DEPENDS carts-compile)
