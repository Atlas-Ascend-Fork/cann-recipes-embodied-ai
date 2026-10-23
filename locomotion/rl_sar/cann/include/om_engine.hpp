/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef RL_SAR_OM_ENGINE_HPP
#define RL_SAR_OM_ENGINE_HPP
#include <acl/acl.h>
#include <cstdint>
#include <string>
#include <vector>
class OmEngine
{
public:
    explicit OmEngine(int32_t device_id = 0);
    ~OmEngine();
    OmEngine(const OmEngine&) = delete;
    OmEngine& operator=(const OmEngine&) = delete;
    OmEngine(OmEngine&&) = delete;
    OmEngine& operator=(OmEngine&&) = delete;
    void LoadModel(const std::string& model_path);
    bool IsModelLoaded() const;
    std::vector<std::vector<float>> Forward(const std::vector<float>& observations, float time_step);
private:
    void InitializeRuntime();
    void CreateModelBuffers();
    void ValidateInputContract(size_t input_count) const;
    void ValidateOutputContract(size_t output_count) const;
    void CreateInputDataset(size_t input_count);
    void CreateOutputDataset(size_t output_count);
    void AddInputBuffer(size_t index);
    void AddOutputBuffer(size_t index);
    void ReleaseModel() noexcept;
    void ReleaseDataBuffers() noexcept;
    void ReleaseDatasets() noexcept;
    void ReleaseDeviceBuffers() noexcept;
    void ClearModelStorage() noexcept;
    void ReleaseLoadedModel() noexcept;
    void ReleaseRuntime() noexcept;
    int32_t device_id_;
    bool acl_initialized_ = false;
    bool device_set_ = false;
    bool model_loaded_ = false;
    uint32_t model_id_ = 0;
    aclrtContext context_ = nullptr;
    aclmdlDesc* model_desc_ = nullptr;
    aclmdlDataset* input_dataset_ = nullptr;
    aclmdlDataset* output_dataset_ = nullptr;
    std::vector<void*> input_device_buffers_;
    std::vector<void*> output_device_buffers_;
    std::vector<aclDataBuffer*> input_data_buffers_;
    std::vector<aclDataBuffer*> output_data_buffers_;
    std::vector<size_t> input_sizes_;
    std::vector<size_t> output_sizes_;
};
#endif
