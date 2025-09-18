cmake_minimum_required(VERSION 3.14)
project(column-io VERSION 1.0 LANGUAGES CXX CUDA)

option(INTERNAL_VERSION "Option for internal version" ON)
message(STATUS "internal version: " ${INTERNAL_VERSION})
if (INTERNAL_VERSION)
  set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
  add_definitions(-DINTERNAL_VERSION=1)
else()
  set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
  add_definitions(-DINTERNAL_VERSION=0)
endif(INTERNAL_VERSION)

set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

find_package(CUDA REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;86")
endif()

add_compile_options(-ggdb -O3 "-fvisibility=hidden")

#FLAG: NEED_ODPS_COLUMN 表示启用ODPS storage接口. 副作用为使用Cxx11abi=1, 禁用直读接口和新ailake接口等不兼容新abi的功能
if (INTERNAL_VERSION)
    set(NEED_ODPS_COLUMN $ENV{NEED_ODPS_COLUMN})
    if (NOT DEFINED NEED_ODPS_COLUMN OR NEED_ODPS_COLUMN STREQUAL "0")
        set(_GLIBCXX_USE_CXX11_ABI 0 CACHE INTERNAL "Use C++11ABI=0")
    else()
        set(_GLIBCXX_USE_CXX11_ABI 1 CACHE INTERNAL "Use C++11ABI=1")
    endif()
    message(STATUS "Now ColumnIO Using _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI} -DNDEBUG)
    add_definitions(-DINTERNAL_VERSION=1)
else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1 -DNDEBUG)
    add_definitions(-DINTERNAL_VERSION=0)
endif(INTERNAL_VERSION)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

set(THIRD_PARTY_DIR "${PROJECT_BINARY_DIR}/third_party_install"
        CACHE PATH "Where to install third party headers and libs")
set(of_proto_python_dir "${PROJECT_BINARY_DIR}/of_proto_python")
add_custom_target(make_pyproto_dir ALL COMMAND ${CMAKE_COMMAND} -E make_directory
                                               ${of_proto_python_dir})
					       
add_subdirectory(third-party/abseil-cpp-20230802.1)
include(cmake/rapidjson.cmake)
include(cmake/zlib.cmake)
include(cmake/protobuf.cmake)
include(cmake/proto2cpp.cmake)
if (INTERNAL_VERSION)
    if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
        include(cmake/odps.cmake)
        set(ARROW_LIBRARY
            PUBLIC arrow
        )
    else()
        set(ARROW_LIBRARY
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libjemalloc_pic.a
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libarrow.a
        )
    endif()
    include(cmake/ailake.cmake)
else()
    set(ARROW_LIBRARY
        PUBLIC arrow
    )
endif(INTERNAL_VERSION)

set(DLPACK_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third-party/dlpack/include)
message(STATUS "dlpack: " ${DLPACK_INCLUDE_DIR})
add_subdirectory(third-party/pybind11)
add_subdirectory(cc)
