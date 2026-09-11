cmake_minimum_required(VERSION 3.20)
project(parce_cert_exp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(software_gate src/software_gate.cpp)
target_compile_options(software_gate PRIVATE -Wall -Wextra)

add_executable(sender src/sender.cpp)
target_compile_options(sender PRIVATE -Wall -Wextra)

add_executable(receiver src/receiver.cpp)
target_compile_options(receiver PRIVATE -Wall -Wextra)
