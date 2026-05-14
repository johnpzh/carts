set(CARTS_SDE_TRANSFORMS_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/state/PatternAnalysis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/state/codelet/ScalarForwarding.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dep/loop/IterationSpaceDecomposition.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dep/loop/Vectorization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dep/tensor/Interchange.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dep/tensor/Tiling.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dep/fusion/ElementwiseFusion.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/effect/scheduling/ChunkOpt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/effect/scheduling/ReductionStrategy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/effect/scheduling/ScheduleRefinement.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/effect/distribution/BarrierElimination.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/effect/distribution/DistributionPlanning.cpp"
)
set(CARTS_SDE_TRANSFORMS_SOURCE_FILES ${CARTS_SDE_TRANSFORMS_SOURCE_FILES} PARENT_SCOPE)
