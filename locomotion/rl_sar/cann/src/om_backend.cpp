/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference_backend.hpp"
#include "om_engine.hpp"
#include "g1_beyonddance_contract.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace
{
using namespace g1_beyonddance;

void ValidatePolicy(const PolicyDimensions& policy)
{
    if (policy.observation_count != OBSERVATION_COUNT || policy.joint_count != JOINT_COUNT)
    {
        throw std::runtime_error(
            "G1 BeyondDance requires 154 observations and 29 joints; configuration provides " +
            std::to_string(policy.observation_count) + " observations and " +
            std::to_string(policy.joint_count) + " joints");
    }
}

bool HasSuffix(const std::string& value, const std::string& suffix)
{
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void ValidateFinite(const std::vector<float>& output, size_t index)
{
    for (float value : output)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error("OM output " + std::to_string(index) + " contains non-finite values");
        }
    }
}

void ValidateOutputs(const std::vector<std::vector<float>>& outputs, const PolicyDimensions& policy)
{
    if (outputs.size() <= BODY_QUAT_OUTPUT_INDEX || outputs[ACTION_OUTPUT_INDEX].size() != policy.joint_count ||
        outputs[REF_JOINT_POS_OUTPUT_INDEX].size() != policy.joint_count ||
        outputs[REF_JOINT_VEL_OUTPUT_INDEX].size() != policy.joint_count ||
        outputs[BODY_QUAT_OUTPUT_INDEX].size() < ANCHOR_QUAT_OFFSET + QUATERNION_SIZE)
    {
        throw std::runtime_error("OM model output contract does not match the G1 policy");
    }
    for (size_t index = 0; index < outputs.size(); ++index)
    {
        ValidateFinite(outputs[index], index);
    }
}

class OmBackend final : public InferenceBackend
{
public:
    explicit OmBackend(const PolicyDimensions& policy) : policy_(policy)
    {
        ValidatePolicy(policy_);
    }

    void LoadModel(const std::string& model_path) override
    {
        engine_.LoadModel(model_path);
    }

    bool IsModelLoaded() const override
    {
        return engine_.IsModelLoaded();
    }

    InferenceResult Forward(const std::vector<float>& observations, float time_step) override
    {
        if (observations.size() != policy_.observation_count)
        {
            throw std::runtime_error("Observation count differs from the validated policy configuration");
        }
        std::vector<std::vector<float>> outputs = engine_.Forward(observations, time_step);
        ValidateOutputs(outputs, policy_);

        const auto& body_quat = outputs[BODY_QUAT_OUTPUT_INDEX];
        InferenceResult result;
        result.actions = outputs[ACTION_OUTPUT_INDEX];
        result.ref_joint_pos = outputs[REF_JOINT_POS_OUTPUT_INDEX];
        result.ref_joint_vel = outputs[REF_JOINT_VEL_OUTPUT_INDEX];
        result.ref_body_quat.assign(body_quat.begin() + ANCHOR_QUAT_OFFSET,
                                    body_quat.begin() + ANCHOR_QUAT_OFFSET + QUATERNION_SIZE);
        return result;
    }

private:
    PolicyDimensions policy_;
    OmEngine engine_;
};
}

std::unique_ptr<InferenceBackend> CreateExternalInferenceBackend(
    const std::string& model_path, const PolicyDimensions& policy)
{
    if (HasSuffix(model_path, ".om"))
    {
        return std::make_unique<OmBackend>(policy);
    }
    return nullptr;
}
