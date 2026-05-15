add_library(CartsBenchmarks STATIC
  CartsBenchmarks.cpp
)

# Keep benchmark timing helpers as ordinary out-of-line calls.
# -O0 prevents local optimization of the warmup/timer bodies.
# -fno-lto and IPO=FALSE prevent whole-program/LTO rewriting of these calls.
# -stdlib=libc++ matches the libc++ used at final link time.
target_compile_options(CartsBenchmarks PRIVATE -O0 -fno-lto -stdlib=libc++)

# Enable OpenMP for thread pool warmup (required for single-node ARTS vs OMP comparisons)
find_package(OpenMP REQUIRED)
target_link_libraries(CartsBenchmarks PRIVATE OpenMP::OpenMP_CXX)

# dlsym (Linux needs ${CMAKE_DL_LIBS}; macOS resolves in libSystem)
target_link_libraries(CartsBenchmarks PRIVATE ${CMAKE_DL_LIBS})

set_target_properties(CartsBenchmarks PROPERTIES
  OUTPUT_NAME cartsbenchmarks
  INTERPROCEDURAL_OPTIMIZATION FALSE
)

install(TARGETS CartsBenchmarks
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${CMAKE_SOURCE_DIR}/include/carts/utils/benchmarks/CartsBenchmarks.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/carts/utils/benchmarks
)
