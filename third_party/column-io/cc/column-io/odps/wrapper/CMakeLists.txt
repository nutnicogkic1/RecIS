
if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
    file(GLOB ODPS_WRAPPER_CC *.cc)
    add_library(odps_wrapper ${ODPS_WRAPPER_CC})
    target_link_libraries(odps_wrapper PRIVATE odps_proxy arrow absl::strings absl::log)
    target_compile_definitions(odps_wrapper PUBLIC TF_ENABLE_ODPS_COLUMN)
else()
    # NO CXX11ABI=1  odps-algosdk to use now
endif()