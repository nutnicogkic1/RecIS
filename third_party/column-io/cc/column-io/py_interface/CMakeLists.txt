file(GLOB pyinterface_CC *.cc)
pybind11_add_module(py_interface NO_EXTRAS ${pyinterface_CC})
set(INTERFACE_LIBRARY
    PRIVATE "-Wl,--whole-archive"
    PRIVATE dataset
    PRIVATE dataset_impl
    PRIVATE "-Wl,--no-whole-archive"
    PRIVATE absl::statusor 
    PRIVATE absl::strings 
    PRIVATE absl::status 
    PRIVATE absl::log_initialize
    PRIVATE absl::flat_hash_map
    PRIVATE pybind11::pybind11
    PRIVATE pybind11::module
    PRIVATE pybind11::embed
    PRIVATE "-rdynamic")
set(INTERFACE_INCLUDE_DIR
    ${CC_BASE_INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}/rapidjson_package-16be2bfba6f9153a138e5a0c04213076/src/rapidjson_package/include
    )
set(COPY_SOURCE $<TARGET_FILE_DIR:py_interface>/$<TARGET_FILE_NAME:py_interface>)
if (INTERNAL_VERSION)
    if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
        target_compile_definitions(py_interface PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
        set(COPY_SOURCE
            ${COPY_SOURCE}
            $<TARGET_FILE_DIR:odps_plugin>/$<TARGET_FILE_NAME:odps_plugin>
        )
        target_link_libraries(py_interface PRIVATE "-Wl,-rpath,'$ORIGIN/'")
    else()
        set(INTERFACE_LIBRARY
            ${INTERFACE_LIBRARY}
            PRIVATE open_storage_wrapper)
        set(INTERFACE_INCLUDE_DIR
            ${INTERFACE_INCLUDE_DIR}
            ${PROJECT_BINARY_DIR}/odps_sdk_cpp)
        target_link_libraries(py_interface 
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libarrow.a
            PRIVATE ${PROJECT_BINARY_DIR}/odps_sdk_cpp/third_party/gcc-8/arrow-1.0.0/lib64/libjemalloc_pic.a
            PRIVATE /usr/local/lib64/liblz4.a
            PRIVATE ${PROJECT_BINARY_DIR}/protobuf/src/protobuf/cmake/libprotobuf.a
            PRIVATE "-Wl,-rpath,'$ORIGIN/'"
        )
        #FLAG: _GLIBCXX_USE_CXX11_ABI默认不传给g++ -o py_interface.o, 需要通过target_compile_definitions传递编译参数
        target_compile_definitions(py_interface PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
        set(COPY_SOURCE
            ${COPY_SOURCE}
            $<TARGET_FILE_DIR:open_storage_plugin>/$<TARGET_FILE_NAME:open_storage_plugin>
            )
    endif()
    file(GLOB lake_sdk_files "${lake_sdk_LIBRARY_BASE}/*")
    set(COPY_SOURCE
        ${COPY_SOURCE}
        ${lake_sdk_files})
endif()
target_link_libraries(py_interface ${INTERFACE_LIBRARY})
target_include_directories(py_interface PRIVATE ${INTERFACE_INCLUDE_DIR})

add_custom_target(copy_cxx_lib ALL 
    COMMAND ${CMAKE_COMMAND} -E copy 
        ${COPY_SOURCE}
        ${CMAKE_SOURCE_DIR}/column_io/lib
    )
add_dependencies(copy_cxx_lib py_interface)
