cmake_minimum_required(VERSION 3.16)
project(varpro_installed_consumer LANGUAGES CXX)

find_package(varpro CONFIG REQUIRED)
add_executable(header_consumer main.cpp)
target_link_libraries(header_consumer PRIVATE varpro::varpro)
