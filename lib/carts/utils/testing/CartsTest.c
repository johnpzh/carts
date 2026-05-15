///===----------------------------------------------------------------------===///
/// File: CartsTest.c
///
/// Implementation of CARTS test utilities.
///===----------------------------------------------------------------------===///

#include <stdio.h>
#include <string.h>

/// Extract test name from __FILE__ path.
/// For "/path/to/examples/foo/bar/test.c" returns "foo/bar"
/// For "/path/to/examples/foo/test.c" returns "foo"
const char *carts_extract_test_name(const char *path) {
  static char buf[256];
  const char *last_slash = strrchr(path, '/');
  if (!last_slash)
    return path;

  /// Find the "examples" directory marker
  const char *examples = strstr(path, "/examples/");
  if (examples) {
    const char *start = examples + 10;
    size_t len = (size_t)(last_slash - start);
    if (len > 0 && len < sizeof(buf) - 1) {
      strncpy(buf, start, len);
      buf[len] = '\0';
      return buf;
    }
  }

  /// Fallback: return just the parent directory name
  const char *p = last_slash - 1;
  while (p > path && *p != '/')
    p--;
  const char *start = (*p == '/') ? p + 1 : p;
  size_t len = (size_t)(last_slash - start);
  if (len >= sizeof(buf))
    len = sizeof(buf) - 1;
  strncpy(buf, start, len);
  buf[len] = '\0';
  return buf;
}

/// Report test pass with optional timing.
int carts_test_pass_impl(const char *file, double elapsed) {
  const char *name = carts_extract_test_name(file);
  if (elapsed >= 0)
    printf("[CARTS] %s: PASS (%.4fs)\n", name, elapsed);
  else
    printf("[CARTS] %s: PASS\n", name);
  return 0;
}

/// Report test failure with reason.
int carts_test_fail_impl(const char *file, const char *reason) {
  const char *name = carts_extract_test_name(file);
  if (reason)
    printf("[CARTS] %s: FAIL - %s\n", name, reason);
  else
    printf("[CARTS] %s: FAIL\n", name);
  return 1;
}
