# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(rl_sar_backend_tests LANGUAGES CXX)
include(CTest)
add_executable(test_om_backend ../../cann/src/om_backend.cpp test_om_backend.cpp)
target_compile_features(test_om_backend PRIVATE cxx_std_17)
target_include_directories(test_om_backend PRIVATE ../fake_acl ../../cann/include)
target_compile_options(test_om_backend PRIVATE -Wall -Wextra -Werror)
add_test(NAME om_backend_mapping COMMAND test_om_backend)
set_tests_properties(om_backend_mapping PROPERTIES LABELS "host;mock")

# Exercise both production layers; replace only the external ACL runtime.
add_executable(test_om_integration
    ../../cann/src/om_backend.cpp
    ../../cann/src/om_engine.cpp
    ../fake_acl_runtime.cpp
    test_om_integration.cpp
)
target_compile_features(test_om_integration PRIVATE cxx_std_17)
target_include_directories(test_om_integration PRIVATE ../fake_acl .. ../../cann/include)
target_compile_options(test_om_integration PRIVATE -Wall -Wextra -Werror)
add_test(NAME om_integration_mock_acl COMMAND test_om_integration)
set_tests_properties(om_integration_mock_acl PROPERTIES LABELS "host;mock;integration")
