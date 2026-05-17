add_library(MLIRCartsSdeUtils INTERFACE)

target_link_libraries(MLIRCartsSdeUtils
  INTERFACE
    MLIRIR
    MLIRCartsSde
)
