file(GLOB SOURCES "*.cpp")

add_library(functionality_image STATIC 
    ${SOURCES}
)

target_link_libraries(functionality_image INTERFACE 
    CUDA::cusparse 
    CUDA::cudart
)