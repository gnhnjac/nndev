cmake_minimum_required(VERSION 3.20)

set ( CMAKE_CXX_COMPILER gcc )

project(NNDev C)

include_directories(include)

file(GLOB_RECURSE SOURCES src/*.c)
add_executable(main ${SOURCES})
target_link_libraries(main m)