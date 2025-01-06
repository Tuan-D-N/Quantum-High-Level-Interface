file(GLOB SOURCES "*.cpp")

add_library(radon_functionality STATIC 
${SOURCES}
)

target_link_libraries(radon_functionality PRIVATE 
CUDA::cusparse 
CUDA::cudart
)