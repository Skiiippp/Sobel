cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)
project(NaiveSobel)


set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -O0")

find_package(OpenCV REQUIRED)

include_directories("/usr/include/opencv4")

add_executable(NaiveSobel NaiveSobel.cpp)

target_link_libraries(NaiveSobel ${OpenCV_LIBS})
