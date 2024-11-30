cmake_minimum_required(VERSION 3.8.0)

project(least_square)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

find_package(Ceres REQUIRED)
find_package(Python3 COMPONENTS Development NumPy)

include_directories(
    include
    ${CERES_INCLUDE_DIRS}
    /home/b212/下载/第三方库/matplotlib-cpp
)


add_executable(least_square_node
    src/least-square.cpp
)

target_link_libraries(least_square_node
Python3::Python
Python3::NumPy
${CERES_LIBRARIES}

)