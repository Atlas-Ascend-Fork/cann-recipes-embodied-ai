/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "fake_acl_runtime.hpp"
#include "inference_backend.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
constexpr PolicyDimensions POLICY{154, 29};

void Check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

fake_acl::ModelSpec Model()
{
    fake_acl::ModelSpec spec;
    spec.input_sizes = {154 * sizeof(float), sizeof(float)};
    spec.output_sizes = {29 * sizeof(float), 29 * sizeof(float), 29 * sizeof(float),
                         sizeof(float), 32 * sizeof(float)};
    spec.output_values = {std::vector<float>(29, 1.0F), std::vector<float>(29, 2.0F),
                          std::vector<float>(29, 3.0F), {0.0F}, std::vector<float>(32, 0.0F)};
    spec.output_values[4][28] = 1.0F;
    return spec;
}

void ExpectFailure(const std::function<void()>& operation, const std::string& message)
{
    try
    {
        operation();
    }
    catch (const std::runtime_error& error)
    {
        Check(std::string(error.what()).find(message) != std::string::npos, "unexpected error diagnostic");
        return;
    }
    throw std::runtime_error("invalid operation was accepted");
}

void CheckReleased()
{
    const auto count = fake_acl::Resources();
    Check(count.device_buffers == 0 && count.data_buffers == 0 && count.datasets == 0 &&
          count.contexts == 0 && count.loaded_models == 0, "ACL resources leaked");
}

void TestProductionPipeline()
{
    fake_acl::Reset(Model());
    {
        auto backend = CreateExternalInferenceBackend("policy.om", POLICY);
        backend->LoadModel("policy.om");
        Check(backend->IsModelLoaded(), "model not loaded");
        for (int step = 0; step < 3; ++step)
        {
            const auto result = backend->Forward(std::vector<float>(154, 0.0F), static_cast<float>(step));
            Check(result.actions == std::vector<float>(29, 1.0F), "actions differ");
            Check(result.ref_joint_pos == std::vector<float>(29, 2.0F), "positions differ");
            Check(result.ref_joint_vel == std::vector<float>(29, 3.0F), "velocities differ");
            Check(result.ref_body_quat == std::vector<float>({1, 0, 0, 0}), "quaternion differs");
        }
    }
    const auto& events = fake_acl::Events();
    Check(std::count(events.begin(), events.end(), "execute") == 3, "production engine did not execute");
    Check(std::count(events.begin(), events.end(), "memcpy") == 21, "input/output copies were bypassed");
    CheckReleased();
}

void TestConfigurationMismatch()
{
    fake_acl::Reset(Model());
    for (const PolicyDimensions invalid : {PolicyDimensions{145, 29}, PolicyDimensions{154, 12}})
    {
        ExpectFailure([&] { CreateExternalInferenceBackend("policy.om", invalid); }, "configuration provides");
    }
    Check(fake_acl::Events().empty(), "invalid configuration reached ACL");
}

void TestModelAndInputMismatch()
{
    auto spec = Model();
    spec.input_sizes[0] = 145 * sizeof(float);
    fake_acl::Reset(spec);
    {
        auto backend = CreateExternalInferenceBackend("policy.om", POLICY);
        backend->LoadModel("policy.om");
        ExpectFailure([&] { backend->Forward(std::vector<float>(154, 0.0F), 0); }, "Observation size");
    }
    CheckReleased();

    spec = Model();
    spec.output_sizes[0] = 28 * sizeof(float);
    spec.output_values[0].resize(28);
    fake_acl::Reset(spec);
    {
        auto backend = CreateExternalInferenceBackend("policy.om", POLICY);
        backend->LoadModel("policy.om");
        ExpectFailure([&] { backend->Forward(std::vector<float>(154, 0.0F), 0); }, "output contract");
    }
    CheckReleased();

    spec = Model();
    spec.output_values[0][0] = std::numeric_limits<float>::quiet_NaN();
    fake_acl::Reset(spec);
    {
        auto backend = CreateExternalInferenceBackend("policy.om", POLICY);
        backend->LoadModel("policy.om");
        ExpectFailure([&] { backend->Forward(std::vector<float>(153, 0.0F), 0); }, "Observation count");
        ExpectFailure([&] { backend->Forward(std::vector<float>(154, 0.0F), 0); }, "non-finite");
    }
    CheckReleased();
}

void TestRuntimeFailures()
{
    for (const char* operation : {"init", "set_device", "create_context", "load", "memcpy", "execute"})
    {
        auto spec = Model();
        spec.fail_operation = operation;
        fake_acl::Reset(spec);
        ExpectFailure([&] {
            auto backend = CreateExternalInferenceBackend("policy.om", POLICY);
            backend->LoadModel("policy.om");
            backend->Forward(std::vector<float>(154, 0.0F), 0);
        }, "ACL error");
        CheckReleased();
    }
}
}

int main()
{
    try
    {
        TestProductionPipeline();
        TestConfigurationMismatch();
        TestModelAndInputMismatch();
        TestRuntimeFailures();
        std::cout << "PASS: production backend + production engine (MOCK ACL; no Ascend inference)\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
