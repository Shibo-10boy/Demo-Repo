cmake_minimum_required(VERSION 3.8.0)

project(osqp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORTS_COMPILE_COMMANDS TRUE)

find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)
include_directories(
    include()
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable( st_liner_qp
    src/st_liner_qp.cpp
)

target_link_libraries(
    st_liner_qp
    ${EIGEN3_INCLUDE_LIB}
    OsqpEigen::OsqpEigen
)