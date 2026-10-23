/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "om_engine.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>
namespace
{
constexpr size_t EXPECTED_INPUT_COUNT = 2;
constexpr size_t MIN_OUTPUT_COUNT = 5;
void CheckAcl(aclError status, const char* operation)
{
    if (status == ACL_SUCCESS)
    {
        return;
    }
    std::ostringstream message;
    message << operation << " failed with ACL error " << status;
    const char* detail = aclGetRecentErrMsg();
    if (detail != nullptr && detail[0] != '\0')
    {
        message << ": " << detail;
    }
    throw std::runtime_error(message.str());
}
void RequireFloat32(aclDataType type, const char* tensor_name, size_t index)
{
    if (type != ACL_FLOAT)
    {
        std::ostringstream message;
        message << tensor_name << " " << index << " must be ACL_FLOAT32";
        throw std::runtime_error(message.str());
    }
}
}
OmEngine::OmEngine(int32_t device_id) : device_id_(device_id) {}
OmEngine::~OmEngine()
{
    ReleaseRuntime();
}
void OmEngine::InitializeRuntime()
{
    if (acl_initialized_)
    {
        return;
    }
    CheckAcl(aclInit(nullptr), "aclInit");
    acl_initialized_ = true;
    try
    {
        CheckAcl(aclrtSetDevice(device_id_), "aclrtSetDevice");
        device_set_ = true;
        CheckAcl(aclrtCreateContext(&context_, device_id_), "aclrtCreateContext");
        CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
    }
    catch (...)
    {
        ReleaseRuntime();
        throw;
    }
}
void OmEngine::LoadModel(const std::string& model_path)
{
    InitializeRuntime();
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
    ReleaseModel();
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
    try
    {
        CheckAcl(aclmdlLoadFromFile(model_path.c_str(), &model_id_), "aclmdlLoadFromFile");
        model_loaded_ = true;
        model_desc_ = aclmdlCreateDesc();
        if (model_desc_ == nullptr)
        {
            throw std::runtime_error("aclmdlCreateDesc returned null");
        }
        CheckAcl(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");
        CreateModelBuffers();
    }
    catch (...)
    {
        ReleaseModel();
        throw;
    }
}
bool OmEngine::IsModelLoaded() const
{
    return model_loaded_;
}
void OmEngine::ValidateInputContract(size_t input_count) const
{
    if (input_count != EXPECTED_INPUT_COUNT)
    {
        throw std::runtime_error("OM model must expose observation and time-step inputs");
    }
    for (size_t index = 0; index < input_count; ++index)
    {
        const size_t size = aclmdlGetInputSizeByIndex(model_desc_, index);
        if (size == 0 || size % sizeof(float) != 0)
        {
            throw std::runtime_error("OM input must contain non-empty float32 data");
        }
        RequireFloat32(aclmdlGetInputDataType(model_desc_, index), "OM input", index);
        if (index == 1 && size != sizeof(float))
        {
            throw std::runtime_error("OM time-step input must contain one float32 value");
        }
    }
}

void OmEngine::ValidateOutputContract(size_t output_count) const
{
    if (output_count < MIN_OUTPUT_COUNT)
    {
        throw std::runtime_error("OM model must expose five policy outputs");
    }
    for (size_t index = 0; index < output_count; ++index)
    {
        const size_t size = aclmdlGetOutputSizeByIndex(model_desc_, index);
        if (size == 0 || size % sizeof(float) != 0)
        {
            throw std::runtime_error("OM output must contain non-empty float32 data");
        }
        RequireFloat32(aclmdlGetOutputDataType(model_desc_, index), "OM output", index);
    }
}

void OmEngine::CreateModelBuffers()
{
    const size_t input_count = aclmdlGetNumInputs(model_desc_);
    ValidateInputContract(input_count);
    const size_t output_count = aclmdlGetNumOutputs(model_desc_);
    ValidateOutputContract(output_count);
    CreateInputDataset(input_count);
    CreateOutputDataset(output_count);
}

void OmEngine::CreateInputDataset(size_t input_count)
{
    input_dataset_ = aclmdlCreateDataset();
    if (input_dataset_ == nullptr)
    {
        throw std::runtime_error("aclmdlCreateDataset failed for model inputs");
    }
    input_sizes_.reserve(input_count);
    input_device_buffers_.reserve(input_count);
    input_data_buffers_.reserve(input_count);
    for (size_t index = 0; index < input_count; ++index)
    {
        AddInputBuffer(index);
    }
}

void OmEngine::CreateOutputDataset(size_t output_count)
{
    output_dataset_ = aclmdlCreateDataset();
    if (output_dataset_ == nullptr)
    {
        throw std::runtime_error("aclmdlCreateDataset failed for model outputs");
    }
    output_sizes_.reserve(output_count);
    output_device_buffers_.reserve(output_count);
    output_data_buffers_.reserve(output_count);
    for (size_t index = 0; index < output_count; ++index)
    {
        AddOutputBuffer(index);
    }
}

void OmEngine::AddInputBuffer(size_t index)
{
    const size_t size = aclmdlGetInputSizeByIndex(model_desc_, index);
    void* device_buffer = nullptr;
    CheckAcl(aclrtMalloc(&device_buffer, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(input)");
    input_sizes_.push_back(size);
    input_device_buffers_.push_back(device_buffer);
    aclDataBuffer* data_buffer = aclCreateDataBuffer(device_buffer, size);
    if (data_buffer == nullptr)
    {
        throw std::runtime_error("aclCreateDataBuffer failed for model input");
    }
    input_data_buffers_.push_back(data_buffer);
    CheckAcl(aclmdlAddDatasetBuffer(input_dataset_, data_buffer), "aclmdlAddDatasetBuffer(input)");
}

void OmEngine::AddOutputBuffer(size_t index)
{
    const size_t size = aclmdlGetOutputSizeByIndex(model_desc_, index);
    void* device_buffer = nullptr;
    CheckAcl(aclrtMalloc(&device_buffer, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(output)");
    output_sizes_.push_back(size);
    output_device_buffers_.push_back(device_buffer);
    aclDataBuffer* data_buffer = aclCreateDataBuffer(device_buffer, size);
    if (data_buffer == nullptr)
    {
        throw std::runtime_error("aclCreateDataBuffer failed for model output");
    }
    output_data_buffers_.push_back(data_buffer);
    CheckAcl(aclmdlAddDatasetBuffer(output_dataset_, data_buffer), "aclmdlAddDatasetBuffer(output)");
}
std::vector<std::vector<float>> OmEngine::Forward(const std::vector<float>& observations, float time_step)
{
    if (!model_loaded_)
    {
        throw std::runtime_error("OM model is not loaded");
    }
    if (observations.empty())
    {
        throw std::runtime_error("Observation input must not be empty");
    }
    for (float observation : observations)
    {
        if (!std::isfinite(observation))
        {
            throw std::runtime_error("Observation input must contain finite values");
        }
    }
    if (!std::isfinite(time_step))
    {
        throw std::runtime_error("Time-step input must be finite");
    }
    if (observations.size() * sizeof(float) != input_sizes_.at(0))
    {
        throw std::runtime_error("Observation size does not match the OM model input");
    }
    CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
    CheckAcl(aclrtMemcpy(input_device_buffers_.at(0), input_sizes_.at(0), observations.data(), input_sizes_.at(0),
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(observations)");
    CheckAcl(aclrtMemcpy(input_device_buffers_.at(1), input_sizes_.at(1), &time_step, sizeof(time_step),
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(time_step)");
    CheckAcl(aclmdlExecute(model_id_, input_dataset_, output_dataset_), "aclmdlExecute");
    std::vector<std::vector<float>> outputs;
    outputs.reserve(output_sizes_.size());
    for (size_t index = 0; index < output_sizes_.size(); ++index)
    {
        std::vector<float> output(output_sizes_[index] / sizeof(float));
        CheckAcl(aclrtMemcpy(output.data(), output_sizes_[index], output_device_buffers_[index], output_sizes_[index],
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy(output)");
        outputs.push_back(std::move(output));
    }
    return outputs;
}
void OmEngine::ReleaseDataBuffers() noexcept
{
    for (aclDataBuffer* buffer : input_data_buffers_)
    {
        if (buffer != nullptr)
        {
            (void)aclDestroyDataBuffer(buffer);
        }
    }
    for (aclDataBuffer* buffer : output_data_buffers_)
    {
        if (buffer != nullptr)
        {
            (void)aclDestroyDataBuffer(buffer);
        }
    }
}

void OmEngine::ReleaseDatasets() noexcept
{
    if (input_dataset_ != nullptr)
    {
        (void)aclmdlDestroyDataset(input_dataset_);
    }
    if (output_dataset_ != nullptr)
    {
        (void)aclmdlDestroyDataset(output_dataset_);
    }
}

void OmEngine::ReleaseDeviceBuffers() noexcept
{
    for (void* buffer : input_device_buffers_)
    {
        if (buffer != nullptr)
        {
            (void)aclrtFree(buffer);
        }
    }
    for (void* buffer : output_device_buffers_)
    {
        if (buffer != nullptr)
        {
            (void)aclrtFree(buffer);
        }
    }
}

void OmEngine::ClearModelStorage() noexcept
{
    input_data_buffers_.clear();
    output_data_buffers_.clear();
    input_device_buffers_.clear();
    output_device_buffers_.clear();
    input_sizes_.clear();
    output_sizes_.clear();
    input_dataset_ = nullptr;
    output_dataset_ = nullptr;
}

void OmEngine::ReleaseLoadedModel() noexcept
{
    if (model_loaded_)
    {
        (void)aclmdlUnload(model_id_);
        model_loaded_ = false;
        model_id_ = 0;
    }
    if (model_desc_ != nullptr)
    {
        (void)aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }
}

void OmEngine::ReleaseModel() noexcept
{
    if (context_ != nullptr)
    {
        (void)aclrtSetCurrentContext(context_);
    }
    ReleaseDataBuffers();
    ReleaseDatasets();
    ReleaseDeviceBuffers();
    ClearModelStorage();
    ReleaseLoadedModel();
}
void OmEngine::ReleaseRuntime() noexcept
{
    ReleaseModel();
    if (context_ != nullptr)
    {
        (void)aclrtDestroyContext(context_);
        context_ = nullptr;
    }
    if (device_set_)
    {
        (void)aclrtResetDevice(device_id_);
        device_set_ = false;
    }
    if (acl_initialized_)
    {
        (void)aclFinalize();
        acl_initialized_ = false;
    }
}
