file(GLOB CARTS_TRANSFORMS_SOURCE_FILES CONFIGURE_DEPENDS
    "*.cpp"
    "db/*.cpp"
    "edt/*.cpp"
    "epoch/*.cpp"
    "verify/*.cpp"
)

set(CARTS_TRANSFORMS_SOURCE_FILES ${CARTS_TRANSFORMS_SOURCE_FILES} PARENT_SCOPE)
