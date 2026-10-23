# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(rl_sar_cann LANGUAGES CXX)

include(CTest)

option(RL_SAR_ENABLE_OM "Enable CANN OM inference" ON)
set(RL_SAR_SOURCE_DIR ""
    CACHE PATH "Path to the patched rl_sar source tree")

if(NOT EXISTS "${RL_SAR_SOURCE_DIR}/library/core/rl_sdk/rl_sdk.cpp")
    message(FATAL_ERROR
        "rl_sar source is incomplete at '${RL_SAR_SOURCE_DIR}'. "
        "Initialize submodules recursively and build through build.sh.")
endif()

if(RL_SAR_ENABLE_OM)
    file(READ "${RL_SAR_SOURCE_DIR}/library/core/rl_sdk/rl_sdk.hpp" RL_SDK_HEADER)
    string(FIND "${RL_SDK_HEADER}" "InitExternalBackend" BACKEND_DECLARATION)
    if(BACKEND_DECLARATION EQUAL -1)
        message(FATAL_ERROR "The source tree is not patched. Build through build.sh.")
    endif()
    # Keep Torch's imported targets visible to the offline executable as well
    # as the upstream subdirectory (its interface uses target expressions).
    if(DEFINED ENV{Torch_DIR} AND EXISTS "$ENV{Torch_DIR}/share/cmake/Torch")
        list(APPEND CMAKE_PREFIX_PATH "$ENV{Torch_DIR}/share/cmake/Torch")
    endif()
    find_package(Torch REQUIRED)
endif()

add_subdirectory("${RL_SAR_SOURCE_DIR}" rl_sar)

if(RL_SAR_ENABLE_OM)
    if(DEFINED ENV{ASCEND_TOOLKIT_HOME} AND NOT "$ENV{ASCEND_TOOLKIT_HOME}" STREQUAL "")
        set(ASCEND_TOOLKIT_HOME "$ENV{ASCEND_TOOLKIT_HOME}")
    else()
        set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(ASCEND_ARCH "aarch64-linux")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        set(ASCEND_ARCH "x86_64-linux")
    else()
        message(FATAL_ERROR "Unsupported CANN architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    find_path(ASCENDCL_INCLUDE_DIR
        NAMES acl/acl.h
        HINTS
            "${ASCEND_TOOLKIT_HOME}/${ASCEND_ARCH}/include"
            "${ASCEND_TOOLKIT_HOME}/include"
        NO_DEFAULT_PATH
    )
    find_library(ASCENDCL_LIBRARY
        NAMES ascendcl
        HINTS
            "${ASCEND_TOOLKIT_HOME}/${ASCEND_ARCH}/lib64"
            "${ASCEND_TOOLKIT_HOME}/${ASCEND_ARCH}/devlib"
            "${ASCEND_TOOLKIT_HOME}/lib64"
        NO_DEFAULT_PATH
    )
    if(NOT ASCENDCL_INCLUDE_DIR OR NOT ASCENDCL_LIBRARY)
        message(FATAL_ERROR
            "CANN ACL was not found under '${ASCEND_TOOLKIT_HOME}'. "
            "Source set_env.sh or set ASCEND_TOOLKIT_HOME.")
    endif()

    target_sources(rl_sdk PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/cann/src/om_backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cann/src/om_engine.cpp"
    )
    target_include_directories(rl_sdk PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/cann/include"
        "${ASCENDCL_INCLUDE_DIR}"
    )
    target_compile_definitions(rl_sdk PUBLIC RL_SAR_ENABLE_EXTERNAL_BACKEND)
    target_link_libraries(rl_sdk PUBLIC "${ASCENDCL_LIBRARY}")

    add_executable(rl_infer tools/rl_infer.cpp)
    target_link_libraries(rl_infer PRIVATE rl_sdk)
    set_target_properties(rl_infer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    if(BUILD_TESTING)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_test(NAME ascend_inference
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_ascend_test.py"
                "$<TARGET_FILE:rl_infer>")
        set_tests_properties(ascend_inference PROPERTIES
            LABELS "ascend;hardware;integration"
            SKIP_RETURN_CODE 77
            TIMEOUT 130)
        add_test(NAME ascend_runner_contract
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_device_runner.py")
        set_tests_properties(ascend_runner_contract PROPERTIES LABELS "host")
    endif()
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
    add_subdirectory(tests/backend)
endif()
