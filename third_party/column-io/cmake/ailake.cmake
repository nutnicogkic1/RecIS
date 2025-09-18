include(ExternalProject)
#FLAG: NEED_ODPS_COLUMN 表示启用ODPS storage接口. 副作用为使用Cxx11abi=1, 禁用直读接口和新ailake接口等不兼容新abi的功能
if (NOT DEFINED NEED_ODPS_COLUMN OR NEED_ODPS_COLUMN STREQUAL "0")
    set(_GLIBCXX_USE_CXX11_ABI 0 CACHE INTERNAL "Use C++11ABI=0 for ailake")
    set(lake_sdk_URL "")
    set(lake_sdk_MD5 "0")
else()
    set(_GLIBCXX_USE_CXX11_ABI 1 CACHE INTERNAL "Use C++11ABI=1 for ailake")
    # FIXME: ABI1功能下编译的lakebatchdataset模块, 试运行环境有可能无法正常处理schema、batch数据, 推荐用一层wrapper完全隔离ailake的符号
    message(WARNING "在CXX11ABI=1条件下编译Ailake模块, 注意本模块目前无法在ABI0环境下正常工作")
    set(lake_sdk_URL "")
    set(lake_sdk_MD5 "0")
endif()
ExternalProject_Add(
  lake_sdk
  PREFIX lake-sdk-${lake_sdk_MD5}
  URL ${lake_sdk_URL}
  URL_MD5 ${lake_sdk_MD5}
  CONFIGURE_COMMAND bash -c "echo skipping configuration step" 
  DOWNLOAD_EXTRACT_TIMESTAMP true 
  BUILD_COMMAND bash -c "echo skipping build step" 
  BUILD_IN_SOURCE true 
  INSTALL_COMMAND bash -c "echo skipping install step")

ExternalProject_Get_Property(lake_sdk SOURCE_DIR)
set(lake_sdk_LIBRARY_BASE ${SOURCE_DIR})
add_library(lake INTERFACE)
message("lake source dir: ${SOURCE_DIR}")
target_link_directories(lake INTERFACE ${SOURCE_DIR})
target_link_libraries(lake INTERFACE -l:lib_lake_IO.so -l:liblz4.so -l:libzstd.so)
add_dependencies(lake lake_sdk)
