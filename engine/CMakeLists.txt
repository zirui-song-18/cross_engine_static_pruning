cmake_minimum_required(VERSION 3.15)

project(sparse_engine VERSION 0.1 LANGUAGES CXX)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ENABLE_FLOAT "Enable float value" OFF)
option(NO_RERANK "Disable two-stage re-ranking (single-stage mode)" OFF)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(pybind11)

# Source files
file(GLOB SOURCES "src/*.cpp")

# Create Python module
pybind11_add_module(sparse_engine ${SOURCES})

# Include directories
target_include_directories(sparse_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cereal/include
)

# Compile definitions
if(ENABLE_FLOAT)
    target_compile_definitions(sparse_engine PRIVATE USE_FLOAT)
    message(STATUS "Using FLOAT values")
else()
    message(STATUS "Using INT8 values")
endif()

if(NO_RERANK)
    target_compile_definitions(sparse_engine PRIVATE NO_RERANK)
    message(STATUS "Re-ranking DISABLED (single-stage mode)")
endif()

# Compile options
target_compile_options(sparse_engine PRIVATE
    -O3
    -march=native
    -Wno-format
    -ffast-math
)

# Link libraries
target_link_libraries(sparse_engine PRIVATE
    Threads::Threads
)

# Find Threads
find_package(Threads REQUIRED)
