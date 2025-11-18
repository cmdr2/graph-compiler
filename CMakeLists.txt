cmake_minimum_required(VERSION 3.12)
project(ggml-test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set(GGML_CUDA ON)

add_subdirectory(ggml)

link_libraries(ggml)

file(GLOB_RECURSE SRC_FILES "src/*.cpp" "src/*.h" "src/*.hpp")

add_executable(tinycnn generated/tinycnn.cpp generated/tinycnn_graph.h ${SRC_FILES})
add_executable(sd_vae generated/sd_vae.cpp generated/sd_vae_graph.h ${SRC_FILES})

target_include_directories(tinycnn PRIVATE src)
target_include_directories(sd_vae PRIVATE src)
