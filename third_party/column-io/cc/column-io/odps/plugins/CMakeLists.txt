
if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
    file(GLOB ODPS_PLUGIN_CC *.cc)
    add_library(odps_plugin SHARED ${ODPS_PLUGIN_CC})
    target_link_libraries(odps_plugin
                        PRIVATE algo_package 
                        PRIVATE -lalgo_facade)
    target_include_directories(odps_plugin PRIVATE ${CC_BASE_INCLUDE_DIR})
else()
    # NO CXX11ABI=1  odps-algosdk to use now
endif()
