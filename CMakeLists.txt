cmake_minimum_required(VERSION 3.16)
project(aegis_zero_gc_benchmark C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
else()
    add_compile_options(-O3 -mavx2 -Wall -Wextra -pthread)
endif()

# Standalone 100M/1B operation hardware benchmark
add_executable(bench_1b bench_1b.c)

# Native shared library for PyTorch zero-GC tensor feeding
add_library(aegis_feeder SHARED aegis_feeder.c)
set_target_properties(aegis_feeder PROPERTIES PREFIX "")
