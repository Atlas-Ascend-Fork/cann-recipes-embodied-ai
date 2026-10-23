/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "rl_sdk.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace
{
// Reuse the real RL initialization, observation, inference and command calculation.
// No Unitree channels, state subscriptions or command publishers are constructed.
class OfflineRL final : public RL
{
public:
    torch::Tensor Forward() override { return ForwardExternalBackend(); }
    void GetState(RobotState<double>*) override {}
    void SetCommand(const RobotCommand<double>*) override {}
};

void CheckTensor(const torch::Tensor& tensor, int count, const char* name)
{
    if (tensor.numel() != count || !torch::isfinite(tensor).all().item<bool>())
    {
        throw std::runtime_error(std::string(name) + " has an invalid size or non-finite values");
    }
}

void Usage()
{
    std::cout << "Usage: rl_infer --model /absolute/path/policy.om [--steps 3]\n"
                 "Run the G1 BeyondDance policy on Ascend using synthetic robot observations.\n"
                 "No robot connection or command transmission. Steps: 1..1000.\n";
}

struct Options
{
    std::string model;
    int steps = 3;
    bool help = false;
};

Options ParseArguments(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h")
        {
            Usage();
            options.help = true;
            return options;
        }
        if ((option != "--model" && option != "--steps") || index + 1 >= argc)
        {
            throw std::invalid_argument("Unknown or incomplete argument: " + option);
        }
        const std::string value = argv[++index];
        if (option == "--model")
        {
            options.model = value;
        }
        else
        {
            size_t consumed = 0;
            options.steps = std::stoi(value, &consumed);
            if (consumed != value.size() || options.steps < 1 || options.steps > 1000)
            {
                throw std::invalid_argument("--steps must be an integer between 1 and 1000");
            }
        }
    }
    if (options.model.empty() || std::filesystem::path(options.model).extension() != ".om" ||
        !std::filesystem::is_regular_file(options.model))
    {
        throw std::invalid_argument("--model must name an existing .om file");
    }
    return options;
}

void RunStep(OfflineRL& rl, int step)
{
    rl.episode_length_buf = step;
    const auto start = std::chrono::steady_clock::now();
    const auto actions = rl.Forward();
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    CheckTensor(actions, rl.params.num_of_dofs, "actions");
    CheckTensor(rl.ref_joint_pos, rl.params.num_of_dofs, "reference positions");
    CheckTensor(rl.ref_joint_vel, rl.params.num_of_dofs, "reference velocities");
    CheckTensor(rl.ref_body_quat_w, 4, "reference quaternion");
    rl.ComputeOutput(actions, rl.output_dof_pos, rl.output_dof_vel, rl.output_dof_tau);
    CheckTensor(rl.output_dof_pos, rl.params.num_of_dofs, "position commands");
    CheckTensor(rl.output_dof_vel, rl.params.num_of_dofs, "velocity commands");
    CheckTensor(rl.output_dof_tau, rl.params.num_of_dofs, "torque commands");
    std::cout << "Step " << step << ": " << rl.params.num_observations
              << " observations -> Ascend -> " << actions.numel()
              << " finite joint commands; forward_ms=" << elapsed << '\n';
}

void RunInference(const Options& options)
{
    const auto path = std::filesystem::absolute(options.model).string();
    if (setenv("RL_SAR_MODEL_PATH", path.c_str(), 1) != 0)
    {
        throw std::runtime_error("Could not set the model path");
    }
    torch::set_num_threads(1);
    torch::autograd::GradMode::set_enabled(false);
    OfflineRL rl;
    rl.ReadYamlBase("g1");
    rl.InitRL("g1/robomimic/beyonddance");
    if (!rl.HasExternalBackend()) throw std::runtime_error("OM backend was not initialized");

    std::cout << "Policy: G1 BeyondDance; model: " << path << '\n'
              << "Synthetic observations; command values remain in memory.\n";
    for (int step = 0; step < options.steps; ++step)
    {
        RunStep(rl, step);
    }
    std::cout << "PASS: Ascend offline inference completed (" << options.steps << " steps)\n";
}
}

int main(int argc, char** argv)
{
    Options options;
    try
    {
        options = ParseArguments(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Argument error: " << error.what() << '\n';
        Usage();
        return 2;
    }
    if (options.help)
    {
        return 0;
    }
    try
    {
        RunInference(options);
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: Ascend offline inference: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
