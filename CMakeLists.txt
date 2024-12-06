cmake_minimum_required(VERSION 3.10)
project(Sobel)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories("/usr/include/opencv4")

add_compile_options(-Wall -Wextra -g -O0)

add_executable(Sobel Sobel.cpp)

target_link_libraries(Sobel ${OpenCV_LIBS})
