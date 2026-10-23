/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "inference_backend.hpp"
#include "om_engine.hpp"

#include <limits>
#include <stdexcept>

namespace
{
std::vector<std::vector<float>> model_outputs;
constexpr PolicyDimensions POLICY{154, 29};

void Check(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void ResetOutputs()
{
    model_outputs = {std::vector<float>(29, 1.0F), std::vector<float>(29, 2.0F),
                     std::vector<float>(29, 3.0F), {4.0F}, std::vector<float>(32, 0.0F)};
    model_outputs[4][28] = 1.0F;
}

void ExpectInvalid(InferenceBackend& backend)
{
    bool rejected = false;
    try
    {
        backend.Forward(std::vector<float>(154, 0.0F), 0.0F);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Check(rejected, "invalid backend output was accepted");
}
}

// Stub only the engine: this executable tests production G1 output mapping.
OmEngine::OmEngine(int32_t device_id) : device_id_(device_id) {}
OmEngine::~OmEngine() = default;
void OmEngine::LoadModel(const std::string&) { model_loaded_ = true; }
bool OmEngine::IsModelLoaded() const { return model_loaded_; }
std::vector<std::vector<float>> OmEngine::Forward(const std::vector<float>&, float)
{
    return model_outputs;
}

int main()
{
    Check(!CreateExternalInferenceBackend("policy.onnx", POLICY), "ONNX path unexpectedly selected OM backend");
    Check(!CreateExternalInferenceBackend("policy.om.backup", POLICY), "suffix match accepted a non-OM path");
    auto backend = CreateExternalInferenceBackend("policy.om", POLICY);
    Check(backend != nullptr && !backend->IsModelLoaded(), "OM backend was not created unloaded");
    backend->LoadModel("policy.om");
    Check(backend->IsModelLoaded(), "OM backend did not report loaded");
    ResetOutputs();
    auto result = backend->Forward(std::vector<float>(154, 0.0F), 0.0F);
    Check(result.actions == std::vector<float>(29, 1.0F), "action output mapping mismatch");
    Check(result.ref_joint_pos == std::vector<float>(29, 2.0F), "joint position mapping mismatch");
    Check(result.ref_joint_vel == std::vector<float>(29, 3.0F), "joint velocity mapping mismatch");
    Check(result.ref_body_quat == std::vector<float>{1.0F, 0.0F, 0.0F, 0.0F}, "body quaternion mapping mismatch");
    for (size_t index : {0U, 1U, 2U, 4U})
    {
        ResetOutputs();
        model_outputs[index].pop_back();
        ExpectInvalid(*backend);
        ResetOutputs();
        model_outputs[index][0] = std::numeric_limits<float>::quiet_NaN();
        ExpectInvalid(*backend);
    }
    ResetOutputs();
    model_outputs.resize(4);
    ExpectInvalid(*backend);
}
