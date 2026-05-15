///==========================================================================///
/// File: Umbrella.cpp
///
/// Stub translation unit so the MLIRCartsTransforms umbrella library has at
/// least one source file. The umbrella exists only to aggregate the
/// per-dialect pass libraries (MLIRCartsSdeTransforms,
/// MLIRCartsArtsTransforms, MLIRCartsArtsRtTransforms) for backwards
/// compatibility with consumers that still link MLIRCartsTransforms by name.
///==========================================================================///

namespace mlir::carts {

namespace {

[[maybe_unused]] inline void cartsTransformsUmbrellaAnchor() {}

} // namespace

} // namespace mlir::carts
