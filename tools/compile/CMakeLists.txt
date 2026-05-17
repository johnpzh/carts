set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  IPO
  MCParser
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
)

add_clang_executable(carts-compile
  Compile.cpp
)

install(
  TARGETS carts-compile
  EXPORT CartsTargets
  RUNTIME DESTINATION ${CARTS_TOOLS_DIR}
  COMPONENT carts-compile
)

target_include_directories(carts-compile PRIVATE
  "${LLVM_SOURCE_DIR}/../clang/include"
  "${CMAKE_BINARY_DIR}/tools/clang/include"
)

target_compile_definitions(carts-compile PUBLIC -DLLVM_OBJ_ROOT="${LLVM_BINARY_DIR}")

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRCartsArts
  MLIRCartsArtsRt
  MLIRCartsSde
  MLIRCartsSdeTransforms
  MLIRCartsArtsTransforms
  MLIRCartsArtsRtTransforms
  MLIRCartsCodir
  MLIRCartsSdeToCodirConversion
  MLIRCartsCodirToArtsConversion
  MLIRCartsCodirTransforms
  MLIRFuncAllExtensions
  MLIRPolygeist
  MLIRPolygeistTransforms
  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenMPDialect
  MLIRGPUTransforms
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRFunctionInterfaces
  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRTargetLLVMIRImport
  MLIRSCFTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation
  MLIRGPUToNVVMTransforms
  MLIRGPUToGPURuntimeTransforms
  MLIRFuncAllExtensions
  MLIRLLVMIRToLLVMTranslation
  MLIRFromLLVMIRTranslationRegistration
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRCAPIRegisterEverything
  MLIRBuiltinToLLVMIRTranslation
)

target_link_libraries(carts-compile PRIVATE ${LIBS})
add_dependencies(carts-compile MLIROpsIncGen MLIRRtOpsIncGen
                 MLIRSdeOpsIncGen MLIRCodirOpsIncGen MLIRCartsArtsPassIncGen
                 MLIRCartsCodirConversionPassIncGen
                 MLIRCartsCodirPassIncGen)

if(CARTS_ENABLE_TESTS AND CARTS_TESTS_REQUIRE_INSTALLED_RUNNER)
  set(_carts_install_bin_dir ${CMAKE_INSTALL_PREFIX}/${CARTS_TOOLS_DIR})
  add_custom_command(
    TARGET carts-compile
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_carts_install_bin_dir}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:carts-compile>
            ${_carts_install_bin_dir}/$<TARGET_FILE_NAME:carts-compile>
    COMMENT "Syncing carts-compile into ${_carts_install_bin_dir} for tests"
    VERBATIM
  )
endif()
