cmake_minimum_required(VERSION 3.1)
project(VehicleDC)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
message(STATUS "PythonLibs: ${PYTHON_INCLUDE_DIRS} ${PYTHON_LIBRARIES}")
find_package(NumPy REQUIRED)
if(NOT PYTHON_NUMPY_FOUND)
    message(FATAL_ERROR "Numpy Python library is required.")
endif()

include(ExternalProject)
ExternalProject_Add(eigen
        PREFIX ${CMAKE_BINARY_DIR}/3rd_party/eigen/
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/3rd_party/eigen-git-mirror/
        )

set(DEFAULT_PY_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/share/${CMAKE_PROJECT_NAME}/py_module/ CACHE PATH "Path to default python module path")
set(DEFAULT_MODEL_PATH ${CMAKE_INSTALL_PREFIX}/share/${CMAKE_PROJECT_NAME}/model CACHE PATH "Path to default model path")

option(BUILD_SHARED_LIBS "Build as shared library" ON)
if(BUILD_SHARED_LIBS)
    set(BUILDING_TYPE SHARED)
else()
    set(BUILDING_TYPE STATIC)
endif()
configure_file(src/config.h.in src/config.h ESCAPE_QUOTES)
add_library(VehicleDC ${BUILDING_TYPE} src/vehicle_dc_impl.cpp src/vehicle_dc.cpp)
set_target_properties(VehicleDC PROPERTIES C_VISIBILITY_PRESET hidden)
set_target_properties(VehicleDC PROPERTIES CXX_VISIBILITY_PRESET hidden)
add_dependencies(VehicleDC eigen)
target_include_directories(VehicleDC PUBLIC include)
target_include_directories(VehicleDC PRIVATE ${CMAKE_BINARY_DIR}/src ${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR}
        ${CMAKE_BINARY_DIR}/contrib/include/eigen3 3rd_party/pybind11/include)
target_link_libraries(VehicleDC ${PYTHON_LIBRARIES})
target_compile_definitions(VehicleDC PRIVATE BUILDING_SHARED_LIB)

install(TARGETS VehicleDC
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY share/ DESTINATION share/${CMAKE_PROJECT_NAME})

option(BUILD_TEST "Build the tests" ON)
if(BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            IMMEDIATE @ONLY)

    add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
