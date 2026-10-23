/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SAR_INFERENCE_BACKEND_HPP
#define RL_SAR_INFERENCE_BACKEND_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct PolicyDimensions
{
    std::size_t observation_count;
    std::size_t joint_count;
};

struct InferenceResult
{
    std::vector<float> actions;
    std::vector<float> ref_joint_pos;
    std::vector<float> ref_joint_vel;
    std::vector<float> ref_body_quat;
};

class InferenceBackend
{
public:
    virtual ~InferenceBackend() = default;

    virtual void LoadModel(const std::string& model_path) = 0;
    virtual bool IsModelLoaded() const = 0;
    virtual InferenceResult Forward(const std::vector<float>& observations, float time_step) = 0;
};

std::unique_ptr<InferenceBackend> CreateExternalInferenceBackend(
    const std::string& model_path, const PolicyDimensions& policy);

#endif
