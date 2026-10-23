/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef RL_SAR_G1_BEYONDDANCE_CONTRACT_HPP
#define RL_SAR_G1_BEYONDDANCE_CONTRACT_HPP

#include <cstddef>

// The exported G1 BeyondDance model's ABI, not a generic robot configuration.
// YAML dimensions must agree with this protocol before any model is loaded.
namespace g1_beyonddance
{
inline constexpr std::size_t OBSERVATION_COUNT = 154;
inline constexpr std::size_t JOINT_COUNT = 29;
inline constexpr std::size_t ACTION_OUTPUT_INDEX = 0;
inline constexpr std::size_t REF_JOINT_POS_OUTPUT_INDEX = 1;
inline constexpr std::size_t REF_JOINT_VEL_OUTPUT_INDEX = 2;
inline constexpr std::size_t BODY_QUAT_OUTPUT_INDEX = 4;
inline constexpr std::size_t ANCHOR_QUAT_OFFSET = 28;
inline constexpr std::size_t QUATERNION_SIZE = 4;
}

#endif
