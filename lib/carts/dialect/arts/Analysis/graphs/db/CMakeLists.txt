# Db Graphs Implementation
# Source files for the Db graph components

file(GLOB CARTS_DB_GRAPHS_SOURCE_FILES CONFIGURE_DEPENDS "*.cpp")

# Set the source files for parent scope
set(CARTS_DB_GRAPHS_SOURCE_FILES ${CARTS_DB_GRAPHS_SOURCE_FILES} PARENT_SCOPE) 
