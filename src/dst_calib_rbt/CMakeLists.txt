cmake_minimum_required(VERSION 3.22)
project(dst_calib_rbt VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
find_package(rbt CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)

option(DST_ENABLE_QNN "Build the optional native HTP evaluator adapter" OFF)
if(DST_ENABLE_QNN)
    if(NOT TARGET qnn_runtime)
        add_subdirectory(../qnn_runtime "${CMAKE_CURRENT_BINARY_DIR}/qnn-runtime")
    endif()
    add_library(dst_qnn SHARED src/qnn_evaluator.cpp)
    target_compile_features(dst_qnn PRIVATE cxx_std_20)
    target_link_libraries(dst_qnn PRIVATE qnn_runtime)
    target_compile_options(dst_qnn PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(dst_qnn PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dst-calib/lib")
    install(TARGETS dst_qnn LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

add_executable(dst_calib_rbt_node src/core.cpp src/main.cpp src/orbbec_input.cpp)
set_target_properties(dst_calib_rbt_node PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dst-calib/bin")
target_compile_features(dst_calib_rbt_node PRIVATE cxx_std_20)
target_include_directories(dst_calib_rbt_node PRIVATE
    include
    "${MID_WORKSPACE_DIR}/generated/cpp")
target_link_libraries(dst_calib_rbt_node PRIVATE
    rbt::runtime
    yaml-cpp)
target_compile_options(dst_calib_rbt_node PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(BUILD_TESTING)
    add_executable(dst_calib_rbt_sampling_test tests/test_sampling.cpp)
    target_compile_features(dst_calib_rbt_sampling_test PRIVATE cxx_std_20)
    target_include_directories(dst_calib_rbt_sampling_test PRIVATE include)
    add_test(NAME dst_calib_rbt_sampling COMMAND dst_calib_rbt_sampling_test)
    add_test(NAME dst_calib_rbt_config
        COMMAND dst_calib_rbt_node --config "${MID_WORKSPACE_DIR}/config/dst_calib_rbt/default.yaml"
        --check-config)
    set_tests_properties(dst_calib_rbt_config PROPERTIES TIMEOUT 10)
    add_test(NAME dst_calib_rbt_npu_experimental_config
        COMMAND dst_calib_rbt_node --config "${MID_WORKSPACE_DIR}/config/dst_calib_rbt/npu-experimental.yaml"
        --check-config)
    set_tests_properties(dst_calib_rbt_npu_experimental_config PROPERTIES TIMEOUT 10)
    add_test(NAME dst_calib_v2_config COMMAND dst_calib_rbt_node
        --config "${MID_WORKSPACE_DIR}/config/dst_calib_rbt/deployment-v2-experimental.yaml" --check-config)
    add_executable(dst_input_test tests/test_orbbec_input.cpp src/orbbec_input.cpp)
    target_compile_features(dst_input_test PRIVATE cxx_std_20)
    target_include_directories(dst_input_test PRIVATE include)
    target_link_libraries(dst_input_test PRIVATE rbt::runtime)
    add_test(NAME dst_input_contract COMMAND dst_input_test)
endif()

install(TARGETS dst_calib_rbt_node RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY scripts python DESTINATION ${CMAKE_INSTALL_DATADIR}/dst_calib_rbt)
install(FILES README.md UPSTREAM_INTERFACES.md UPSTREAM_ONLINE.md
        DESTINATION ${CMAKE_INSTALL_DATADIR}/dst_calib_rbt)
install(FILES "${MID_WORKSPACE_DIR}/generated/cpp/rbt/dst_calib.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rbt)
