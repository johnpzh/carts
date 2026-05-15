add_library(CartsTest STATIC
  CartsTest.c
)

set_target_properties(CartsTest PROPERTIES
  OUTPUT_NAME cartstest
)

install(TARGETS CartsTest
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install the header file
install(FILES ${CMAKE_SOURCE_DIR}/include/carts/utils/testing/CartsTest.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/carts/utils/testing
)