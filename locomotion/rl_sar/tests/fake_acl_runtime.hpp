/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ENGINE_STAGE_FAKE_ACL_RUNTIME_HPP
#define ENGINE_STAGE_FAKE_ACL_RUNTIME_HPP
#include <string>
#include <vector>
#include "acl/acl.h"
namespace fake_acl
{
struct ModelSpec
{
    std::vector<size_t> input_sizes{8, 4};
    std::vector<aclDataType> input_types{ACL_FLOAT, ACL_FLOAT};
    std::vector<size_t> output_sizes{4, 4, 4, 4, 128};
    std::vector<aclDataType> output_types{ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT};
    std::vector<std::vector<float>> output_values;
    std::string fail_operation;
    int fail_malloc_at = -1;
    int fail_data_buffer_at = -1;
};
struct ResourceCounts
{
    size_t device_buffers = 0;
    size_t data_buffers = 0;
    size_t datasets = 0;
    size_t contexts = 0;
    size_t loaded_models = 0;
};
void Reset(const ModelSpec& spec = {});
void SwitchToForeignContext();
const std::vector<std::string>& Events();
ResourceCounts Resources();
}
#endif
