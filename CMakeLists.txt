cmake_minimum_required(VERSION 3.16)
project(aegis_zero_gc_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(aegis_benchmark benchmark.cpp)

if(MSVC)
    target_compile_options(aegis_benchmark PRIVATE /O2 /Oi /Ot /arch:AVX2)
else()
    target_compile_options(aegis_benchmark PRIVATE -O3 -march=native -Wall -Wextra -pthread)
endif()
