cmake_minimum_required(VERSION 3.16)
project(varpro_cpp VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# An installed Eigen package or a plain header checkout is sufficient.
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Directory containing Eigen/ and unsupported/")
if(EIGEN3_INCLUDE_DIR)
    if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core" OR
       NOT EXISTS "${EIGEN3_INCLUDE_DIR}/unsupported/Eigen/LevenbergMarquardt")
        message(FATAL_ERROR "EIGEN3_INCLUDE_DIR must contain Eigen and unsupported headers")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
    )
else()
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()

add_library(varpro cpp/varpro.cpp)
add_library(varpro::varpro ALIAS varpro)
target_compile_features(varpro PUBLIC cxx_std_17)
target_include_directories(varpro PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(varpro PUBLIC Eigen3::Eigen)

install(TARGETS varpro
    EXPORT varproTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
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
    add_executable(varpro_tests tests/cpp/test_varpro.cpp)
    target_link_libraries(varpro_tests PRIVATE varpro::varpro)
    add_test(NAME varpro_tests COMMAND varpro_tests)
endif()
