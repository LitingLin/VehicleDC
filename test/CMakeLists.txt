cmake_minimum_required(VERSION 3.1)
project(basic_test)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")
find_package(TurboJPEG)
if(TURBOJPEG_FOUND)
    add_executable(basic_test test.cpp jpeg_decoder.cpp)

    set(TESTING_PY_MODULE_PATH ${CMAKE_SOURCE_DIR}/share/py_module)
    set(TESTING_MODEL_PATH ${CMAKE_SOURCE_DIR}/share/model)
    set(TESTING_SOURCE_PATH ${CMAKE_CURRENT_LIST_DIR})
    configure_file(test.h.in test.h)

    target_include_directories(basic_test PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/Catch2/single_include/
            ${TURBOJPEG_INCLUDE_DIR})
    target_link_libraries(basic_test VehicleDC ${TURBOJPEG_LIBRARY})

    add_test(NAME test1
            COMMAND basic_test)
else()
    message(STATUS "libjpeg-turbo is not found. Testing is disabled.")
endif()
