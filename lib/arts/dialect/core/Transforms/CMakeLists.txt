file(GLOB CARTS_TRANSFORMS_SOURCE_FILES CONFIGURE_DEPENDS
    "*.cpp"
    "db/*.cpp"
    "db/block/*.cpp"
    "db/elementwise/*.cpp"
    "db/strategy/*.cpp"
    "db/transforms/*.cpp"
    "edt/*.cpp"
    "edt/tiling2d/*.cpp"
    "epoch/*.cpp"
    "loop/*.cpp"
    "verify/*.cpp"
)

set(CARTS_TRANSFORMS_SOURCE_FILES ${CARTS_TRANSFORMS_SOURCE_FILES} PARENT_SCOPE)
