set(BUILT ${PROJECT_BINARY_DIR})

message(STATUS "BUILT directory is: ${BUILT}")

set(STORAGE_ODPS_SDK_CPP_BASE
  ""
)
set(ODPS_SDK_CPP_PKG
  "odps_sdk_cpp_20250819_202903-noprint_failpushlog.tar.gz"
  #"odps_sdk_cpp_20250819_202903.tar.gz"
  # "odps_sdk_cpp_20250106_183745_noprint_failpushlog.tar.gz"
)
set(ODPS_SDK_CPP_PKG_URL "${STORAGE_ODPS_SDK_CPP_BASE}/${ODPS_SDK_CPP_PKG}")
set(ODPS_SDK_CPP_PATH "${BUILT}/${ODPS_SDK_CPP_PKG}")

if(NOT EXISTS ${ODPS_SDK_CPP_PATH})
    message(STATUS "Downloading ${ODPS_SDK_CPP_PKG} from ${ODPS_SDK_CPP_PKG_URL}...")
    file(DOWNLOAD ${ODPS_SDK_CPP_PKG_URL} ${ODPS_SDK_CPP_PATH} TIMEOUT 30)
else()
    message(STATUS "${ODPS_SDK_CPP_PKG} already exists. Skipping download.")
endif()

execute_process(
	COMMAND tar -xzf "${ODPS_SDK_CPP_PATH}" -C "${BUILT}"
	WORKING_DIRECTORY "${BUILT}"
  RESULT_VARIABLE TARGZ_RESULT
)

message(STATUS "Current Binary Directory: ${CMAKE_CURRENT_BINARY_DIR}")
if(NOT TARGZ_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to extract ${ODPS_SDK_CPP_PATH}")
endif()

################ 安装编译前依赖 ################
message(STATUS "installing rapidjson header for openstorage/plugin/ in kmonitor deps of autil deps")
set(RAPIDJSON_HEADER_PKG_NAME "rapidjson-v1.1.0.tar.gz")
set(RAPIDJSON_HEADER_PKG_URL "")
set(RAPIDJSON_HEADER_INSTALL_PATH "${BUILT}/odps_sdk_cpp/storage_api/thirdparty")
set(KMONITOR_ADAPTER_RPM_NAME "kmonitor_adapter-0.0.2-3.x86_64.rpm")

if(NOT EXISTS ${BUILT}/${RAPIDJSON_HEADER_PKG_NAME})
    message(STATUS "Downloading ${RAPIDJSON_HEADER_PKG_NAME} from ${RAPIDJSON_HEADER_PKG_URL}...")
    file(DOWNLOAD ${RAPIDJSON_HEADER_PKG_URL} ${BUILT}/${RAPIDJSON_HEADER_PKG_NAME} TIMEOUT 30)
else()
    message(STATUS "${BUILT}/${RAPIDJSON_HEADER_PKG_NAME} already exists. Skipping download.")
endif()
execute_process(
	COMMAND tar -xzf "${BUILT}/${RAPIDJSON_HEADER_PKG_NAME}" -C "${RAPIDJSON_HEADER_INSTALL_PATH}"
	WORKING_DIRECTORY "${BUILT}"
  RESULT_VARIABLE TARGZ_RESULT
)
if(NOT TARGZ_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to extract ${RAPIDJSON_HEADER_PKG_NAME}")
else()
    message(STATUS "finish installing rapidjson header")
endif()
message(STATUS "Current Binary Directory: ${CMAKE_CURRENT_BINARY_DIR}")

# 安装 kmonitor adapter
message(STATUS "installing ${KMONITOR_ADAPTER_RPM_NAME} for openstorage/plugin/ ")
execute_process(
	COMMAND yum install -y ""
)
message(STATUS "installed ${KMONITOR_ADAPTER_RPM_NAME} for openstorage/plugin/ ")

# 下载安装 static std, libidn.so.11, liblz4.a
execute_process(
	COMMAND yum install -y -b current libstdc++-static alicpp-gcc921-libidn
    # liblz4.a(by kimwang. yum liblz4.a is not fPic)
    COMMAND wget -qO /usr/local/lib64/liblz4.a
)

file(GLOB OPEN_STORAGE_PLUGIN_CC_FILTERED *.cc)
set(OPEN_STORAGE_PLUGIN_CC "")

foreach(file ${OPEN_STORAGE_PLUGIN_CC_FILTERED})
    if(NOT file MATCHES "test")
        list(APPEND OPEN_STORAGE_PLUGIN_CC ${file})
    endif()
endforeach()

message(STATUS "Filtered files: ${OPEN_STORAGE_PLUGIN_CC}")

################ 创建编译目标 ################
add_library(open_storage_plugin SHARED ${OPEN_STORAGE_PLUGIN_CC})

################ 添加编译头文件 ################
target_include_directories(open_storage_plugin
  PRIVATE
  ${CC_BASE_INCLUDE_DIR}
  ${BUILT}/odps_sdk_cpp
  ${BUILT}/odps_sdk_cpp/storage_api/thirdparty
  ${BUILT}/odps_sdk_cpp/storage_api/thirdparty/cpp-httplib
  ${BUILT}/odps_sdk_cpp/storage_api/thirdparty/arrow-1.0.0/include
)

################ 添加编译选项 ################
target_compile_options(open_storage_plugin PRIVATE -fvisibility=hidden)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fvisibility.ver "{
global:
  # 隐藏interface.py所需的 CAPI_ODPS_OPEN_STORAGE_* 以外的所有符号
  CAPI_ODPS_OPEN_STORAGE_*;

local:
  *;
};
")

################ 添加链接选项 ################
# find all static lib of autil: $(ls /usr/local/lib64/autil/lib*.a)
file(GLOB AUTIL_LIBS "/usr/local/lib64/autil/lib*.a")
target_link_libraries(open_storage_plugin 
  PRIVATE -fPIC

  -Wl,--start-group
  -lpthread  -ldl
  open_storage_common-util
  ################ ODSP 相关 ################
  PRIVATE /apsara/alicpp/built/gcc-9.2.1/libidn-1.33/lib/libidn.a
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libjemalloc_pic.a
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libarrow.a
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/zlib-1.2.11/lib/libz.a
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/zstd-1.4.5/lib/libzstd.a
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/openssl-1.1.1g/lib/libssl.a
  PRIVATE ${BUILT}/odps_sdk_cpp/third_party/gcc-8/openssl-1.1.1g/lib/libcrypto.a
  PRIVATE /usr/local/lib64/liblz4.a
  PRIVATE /opt/conda/lib/libbz2.a # NOTE: libbz2.a must behind libarrow.a if not group: some symbols like: BZ2_bzDecompress maybe overwide
  
  ################ kmonitor 相关 ################
  # /usr/local/lib64/kmonitor_cpp_adapter.a # kmonitor的另一个接口, 和libkmonitor_client_cpp可以换用
  PRIVATE /usr/local/lib64/libkmonitor_client_cpp.a
  PRIVATE /usr/local/lib64/libkmonitor_proto_cc_impl.a
  PRIVATE /usr/local/lib64/libanet_http_client.a
  # ${BUILT}/odps_sdk_cpp/third_party/gcc-8/protobuf-3.7.1/lib/libprotobuf.a
  # PRIVATE ${BUILT}/protobuf/src/protobuf/cmake/libprotobuf.a
  PRIVATE /usr/local/lib64/libprotobuf.a
  PRIVATE /usr/local/lib64/libprotobuf_lite.a # 3.8.0 from algo/tensorflow from kmonitor in alibaba/aios
  PRIVATE /usr/local/lib64/libarpc.a
  PRIVATE /usr/local/lib64/libanet.a
  PRIVATE /usr/local/lib64/libalog.a
  PRIVATE ${AUTIL_LIBS}
  PRIVATE -static-libgcc  # 静态链接 libc
  PRIVATE -static-libstdc++  # 静态链接 libstdc++
  -Wl,--end-group

  PRIVATE "-Wl,-rpath,'$ORIGIN/'" # 优先加载本动态库统计目录的符号[本条仅用作兼容, 全自依赖后理论无需加载旁路 只需glibc]
  -Wl,--exclude-libs,ALL    # 自动隐藏依赖符号
  -Wl,--as-needed   # 移除未依赖符号
  -Wl,-z,defs   # 确保无未定义符号
  -Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/fvisibility.ver  # 隐藏所有内部类, 暴露CAPI_*符号
  
)
