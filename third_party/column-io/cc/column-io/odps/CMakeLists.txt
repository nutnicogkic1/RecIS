if (_GLIBCXX_USE_CXX11_ABI EQUAL 0)
    message(WARNING "The Column IO algosdk library requires a C++03 ABI instread of C++11 ABI.")
    add_subdirectory(plugins)
    add_subdirectory(proxy)
    add_subdirectory(wrapper)
else()
    # NO odps-algosdk for CXX11ABI=1 now
endif()
