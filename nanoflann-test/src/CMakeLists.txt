cmake_minimum_required(VERSION 3.8)

project(nanoflann-test)

include_directories(${PROJECT_SOURCE_DIR}/include/)

add_executable(nanoflann nanoflann_dynamic.cpp)



