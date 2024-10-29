cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)
project(Sobel)


set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O0")

find_package(OpenCV REQUIRED)

include_directories("/usr/include/opencv4")

add_executable(Sobel Sobel.cpp)

target_link_libraries(Sobel ${OpenCV_LIBS})
