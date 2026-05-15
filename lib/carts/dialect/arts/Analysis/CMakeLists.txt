# Add the main analysis source files
file(GLOB CARTS_ANALYSIS_SOURCE_FILES CONFIGURE_DEPENDS "*.cpp")

# Add the Db subdirectory
add_subdirectory(db)

# Add the EDT subdirectory
add_subdirectory(edt)

# Add the Graphs subdirectory
add_subdirectory(graphs)

# Add the Loop subdirectory
add_subdirectory(loop)

# Add the Heuristics subdirectory
add_subdirectory(heuristics)

# Combine all analysis source files
list(APPEND CARTS_ANALYSIS_SOURCE_FILES ${CARTS_DB_ANALYSIS_SOURCE_FILES})
list(APPEND CARTS_ANALYSIS_SOURCE_FILES ${CARTS_EDT_ANALYSIS_SOURCE_FILES})
list(APPEND CARTS_ANALYSIS_SOURCE_FILES ${CARTS_GRAPHS_SOURCE_FILES})
list(APPEND CARTS_ANALYSIS_SOURCE_FILES ${LOOP_ANALYSIS_SOURCE_FILES})
list(APPEND CARTS_ANALYSIS_SOURCE_FILES ${CARTS_HEURISTICS_SOURCE_FILES})

set(CARTS_ANALYSIS_SOURCE_FILES ${CARTS_ANALYSIS_SOURCE_FILES} PARENT_SCOPE)
