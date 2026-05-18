include("./_cmake/CMakeColors.txt")
cmake_minimum_required(VERSION 3.13.4)

include(CheckCXXSourceCompiles)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

project(CARTS LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

message_title("Project CARTS")

# Building the project against a build area.
message_blue("Searching for MLIRConfig.cmake in: ${MLIR_DIR}")
find_package(MLIR REQUIRED CONFIG)

set(Clang_DIR ${CLANG_DIR})
message_blue("Searching for ClangConfig.cmake in: ${Clang_DIR}")

if(NOT LLVM_BUILD_MAIN_SRC_DIR)
    set(LLVM_BUILD_MAIN_SRC_DIR ${CMAKE_SOURCE_DIR}/llvm-project/llvm)
  endif()
set(LLVM_SOURCE_DIR ${LLVM_BUILD_MAIN_SRC_DIR} CACHE STRING "Location of LLVM source")

# Status
message_info("Using MLIRConfig.cmake in: ${MLIR_DIR}")
message_info("Using LLVMConfig.cmake in: ${LLVM_DIR}")
message_info("Using ClangConfig.cmake in: ${CLANG_DIR}")

# Set up the build and install paths.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddClang)
include(HandleLLVMOptions)
include(GNUInstallDirs)

# Clang and MLIR
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


# Linker configuration
option(CARTS_USE_LINKER "Specify linker to use for CARTS builds" "")
if(CARTS_USE_LINKER)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --ld-path=${CARTS_USE_LINKER} -w")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} --ld-path=${CARTS_USE_LINKER} -w")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --ld-path=${CARTS_USE_LINKER} -w")
  message(STATUS "Using custom linker: ${CARTS_USE_LINKER}")
endif()

# ARTS introspection options
option(ENABLE_ARTS_INTROSPECTION "Enable ARTS introspection and counter system" OFF)
if(ENABLE_ARTS_INTROSPECTION)
  message(STATUS "ARTS introspection enabled")
  add_definitions(-DCOUNTERS)
  add_definitions(-DMETRICS)
else()
  message(STATUS "ARTS introspection disabled")
endif()


# CARTS testing options
option(CARTS_ENABLE_TESTS "Enable CARTS test suite" ON)
option(CARTS_TESTS_REQUIRE_INSTALLED_RUNNER
       "Keep the active CARTS install runner in sync when running tests" ON)
if(CARTS_ENABLE_TESTS)
  message(STATUS "CARTS tests enabled")
  set(CARTS_TOOLS_DIR bin)
  get_filename_component(_CARTS_INSTALL_ROOT "${CMAKE_INSTALL_PREFIX}/.." ABSOLUTE)

  # Locate LLVM test tools (llvm-lit and FileCheck) from common LLVM build/install dirs
  set(_LLVM_TOOL_HINT_DIRS
      ${LLVM_TOOLS_BINARY_DIR}
      ${LLVM_RUNTIME_OUTPUT_INTDIR}
      ${LLVM_BINARY_DIR}
      ${LLVM_DIR}/../../bin
      ${LLVM_DIR}/../bin
      ${POLYGEIST_DIR}/llvm-project/build/bin
      ${POLYGEIST_DIR}/llvm-project/build/NATIVE/bin
      ${_CARTS_INSTALL_ROOT}/llvm/bin
      ${CMAKE_SOURCE_DIR}/.install/llvm/bin)

  find_program(LLVM_LIT_EXECUTABLE NAMES llvm-lit lit.py
               HINTS ${_LLVM_TOOL_HINT_DIRS}
               DOC "Path to llvm-lit test runner")
  find_program(FILECHECK_EXECUTABLE NAMES FileCheck
               HINTS ${_LLVM_TOOL_HINT_DIRS}
               DOC "Path to FileCheck tool")

  # Export a conventional var for downstream use if needed
  if(LLVM_LIT_EXECUTABLE)
    get_filename_component(LLVM_LIT_TOOLS_DIR ${LLVM_LIT_EXECUTABLE} DIRECTORY)
  else()
    set(LLVM_LIT_TOOLS_DIR "")
  endif()
  find_program(DEKK_EXECUTABLE dekk)

  set(MISSING_TOOLS "")
  if(NOT DEKK_EXECUTABLE)
    list(APPEND MISSING_TOOLS dekk)
  endif()
  if(NOT LLVM_LIT_EXECUTABLE)
    list(APPEND MISSING_TOOLS llvm-lit)
  endif()
  if(NOT FILECHECK_EXECUTABLE)
    list(APPEND MISSING_TOOLS FileCheck)
  endif()

  if(MISSING_TOOLS)
    message(WARNING "Missing required test tools: ${MISSING_TOOLS}")
    message(WARNING "Please ensure your LLVM build includes testing tools (llvm-lit, FileCheck) or install them via your LLVM build.")
  else()
    message(STATUS "All required test tools found")
  endif()
  
  # Create test targets with proper dependencies
  add_custom_target(check
    COMMAND ${DEKK_EXECUTABLE} carts test --suite pass
    COMMENT "Running CARTS pass test suite"
    USES_TERMINAL
    VERBATIM
  )
  
  # Add convenience target for samples (end-to-end examples).
  add_custom_target(check-examples
    COMMAND ${DEKK_EXECUTABLE} carts examples run --all
    COMMENT "Running CARTS examples (samples/) test suite"
    USES_TERMINAL
    VERBATIM
  )
  
  # Add verbose test targets
  add_custom_target(check-verbose
    COMMAND ${DEKK_EXECUTABLE} carts test -v
    COMMENT "Running CARTS test suite (verbose)"
    USES_TERMINAL
    VERBATIM
  )

  # E2E tests: compile + run samples
  add_custom_target(check-e2e
    COMMAND ${DEKK_EXECUTABLE} carts test --suite e2e
    COMMENT "Running CARTS E2E test suite (compile + run samples)"
    USES_TERMINAL
    VERBATIM
  )

  # Run both pass and E2E suites
  add_custom_target(check-all
    COMMAND ${DEKK_EXECUTABLE} carts test --suite all
    COMMENT "Running all CARTS test suites"
    USES_TERMINAL
    VERBATIM
  )

  # Generate lit.site.cfg.py from CMake variables for out-of-source test execution.
  configure_file(
    ${CMAKE_SOURCE_DIR}/tests/lit.site.cfg.py.in
    ${CMAKE_BINARY_DIR}/tests/lit.site.cfg.py
    @ONLY)

  # CTest integration: register co-located pass test suites.
  enable_testing()
  add_subdirectory(tests)

else()
  message(STATUS "CARTS tests disabled")
endif()



# Polygeist
include_directories(
  ${POLYGEIST_BUILD_DIR}/include
  ${POLYGEIST_DIR}/include
)
link_directories(${POLYGEIST_BUILD_DIR}/lib)

find_library(MLIR_POLYGEIST_LIB NAMES MLIRPolygeist PATHS ${POLYGEIST_BUILD_DIR}/lib)
if(MLIR_POLYGEIST_LIB)
  message(STATUS "Found Polygeist library: ${MLIR_POLYGEIST_LIB}")
else()
  message(FATAL_ERROR "Polygeist library not found")
endif()

# Create imported targets so CMake tracks the .a files for re-linking
find_library(MLIR_POLYGEIST_TRANSFORMS_LIB NAMES MLIRPolygeistTransforms PATHS ${POLYGEIST_BUILD_DIR}/lib)
if(NOT MLIR_POLYGEIST_TRANSFORMS_LIB)
  message(FATAL_ERROR "MLIRPolygeistTransforms library not found")
endif()

add_library(MLIRPolygeist STATIC IMPORTED)
set_target_properties(MLIRPolygeist PROPERTIES IMPORTED_LOCATION ${MLIR_POLYGEIST_LIB})

add_library(MLIRPolygeistTransforms STATIC IMPORTED)
set_target_properties(MLIRPolygeistTransforms PROPERTIES IMPORTED_LOCATION ${MLIR_POLYGEIST_TRANSFORMS_LIB})

# set(CARTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CARTS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CARTS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )
set(CARTS_TOOLS_DIR bin)
set(CARTS_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)

# Add the subdirectories.
message_blue("Adding subdirectories")
message_blue(" -- Adding include")
add_subdirectory(include)
message_blue(" -- Adding lib")
add_subdirectory(lib)
message_blue(" -- Adding tools")
add_subdirectory(tools)

if(CARTS_ENABLE_TESTS AND CARTS_TESTS_REQUIRE_INSTALLED_RUNNER)
  foreach(_carts_test_target check check-examples check-verbose check-e2e check-all)
    if(TARGET ${_carts_test_target} AND TARGET carts-compile)
      add_dependencies(${_carts_test_target} carts-compile)
    endif()
  endforeach()
endif()
