cmake_minimum_required(VERSION 3.16)
project(varpro_cpp VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(varpro INTERFACE)
add_library(varpro::varpro ALIAS varpro)
target_compile_features(varpro INTERFACE cxx_std_17)
target_include_directories(varpro INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS varpro EXPORT varproTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT varproTargets
    FILE varproTargets.cmake
    NAMESPACE varpro::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/varpro
)

configure_package_config_file(
    cmake/varproConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/varproConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/varpro
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/varproConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/varproConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/varproConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/varpro
)

option(VARPRO_BUILD_EXAMPLE "Build the double-exponential example" ON)
if(VARPRO_BUILD_EXAMPLE)
    add_executable(double_exponential examples/double_exponential.cpp)
    target_link_libraries(double_exponential PRIVATE varpro::varpro)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(varpro_tests tests/cpp/test_varpro.cpp tests/cpp/header_link_probe.cpp)
    target_link_libraries(varpro_tests PRIVATE varpro::varpro)
    add_test(NAME varpro_tests COMMAND varpro_tests)
endif()
