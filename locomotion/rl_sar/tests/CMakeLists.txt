# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(rl_sar_host_tests LANGUAGES CXX)
include(CTest)

add_executable(test_om_engine
    ../cann/src/om_engine.cpp
    fake_acl_runtime.cpp
    test_om_engine.cpp
)
target_compile_features(test_om_engine PRIVATE cxx_std_17)
target_include_directories(test_om_engine PRIVATE fake_acl ../cann/include)
target_compile_options(test_om_engine PRIVATE -Wall -Wextra -Werror)
add_test(NAME om_engine COMMAND test_om_engine)
set_tests_properties(om_engine PROPERTIES LABELS "host;mock")
