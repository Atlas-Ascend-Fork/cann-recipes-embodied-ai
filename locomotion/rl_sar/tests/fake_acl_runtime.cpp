/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "fake_acl_runtime.hpp"
#include <algorithm>
#include <new>
struct aclDataBuffer
{
    void* data = nullptr;
    size_t size = 0;
};
struct aclmdlDataset
{
    std::vector<aclDataBuffer*> buffers;
};
struct aclmdlDesc
{
    fake_acl::ModelSpec spec;
};
namespace
{
fake_acl::ModelSpec model_spec;
std::vector<std::string> events;
aclrtContext current_context = nullptr;
uint32_t next_model_id = 1;
const char* last_error = "";
int malloc_calls = 0;
int data_buffer_calls = 0;
fake_acl::ResourceCounts resources;
void Record(const char* event)
{
    events.emplace_back(event);
}
bool Fail(const char* operation)
{
    if (model_spec.fail_operation != operation)
    {
        return false;
    }
    last_error = "injected ACL failure";
    return true;
}
}
namespace fake_acl
{
void Reset(const ModelSpec& spec)
{
    model_spec = spec;
    events.clear();
    current_context = nullptr;
    next_model_id = 1;
    last_error = "";
    malloc_calls = 0;
    data_buffer_calls = 0;
    resources = {};
}
void SwitchToForeignContext()
{
    current_context = reinterpret_cast<aclrtContext>(static_cast<uintptr_t>(0x1234));
    Record("foreign_context");
}
const std::vector<std::string>& Events()
{
    return events;
}
ResourceCounts Resources()
{
    return resources;
}
}
extern "C" aclError aclInit(const char*)
{
    Record("init");
    if (Fail("init")) return 1;
    return ACL_SUCCESS;
}
extern "C" aclError aclFinalize(void)
{
    Record("finalize");
    return ACL_SUCCESS;
}
extern "C" const char* aclGetRecentErrMsg(void)
{
    return last_error;
}
extern "C" aclError aclrtSetDevice(int32_t)
{
    Record("set_device");
    if (Fail("set_device")) return 1;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtResetDevice(int32_t)
{
    Record("reset_device");
    return ACL_SUCCESS;
}
extern "C" aclError aclrtCreateContext(aclrtContext* context, int32_t device_id)
{
    Record("create_context");
    if (Fail("create_context")) return 1;
    *context = new int(device_id);
    ++resources.contexts;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtSetCurrentContext(aclrtContext context)
{
    current_context = context;
    Record("set_context");
    return ACL_SUCCESS;
}
extern "C" aclError aclrtDestroyContext(aclrtContext context)
{
    Record("destroy_context");
    delete static_cast<int*>(context);
    --resources.contexts;
    if (current_context == context)
    {
        current_context = nullptr;
    }
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMalloc(void** device_ptr, size_t size, aclrtMemMallocPolicy)
{
    Record("malloc");
    ++malloc_calls;
    if (model_spec.fail_malloc_at == malloc_calls)
    {
        last_error = "injected malloc failure";
        *device_ptr = nullptr;
        return 1;
    }
    try
    {
        *device_ptr = ::operator new(size);
    }
    catch (const std::bad_alloc&)
    {
        *device_ptr = nullptr;
        last_error = "allocation failed";
        return 1;
    }
    ++resources.device_buffers;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtFree(void* device_ptr)
{
    Record("free");
    ::operator delete(device_ptr);
    --resources.device_buffers;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemcpy(void* dst, size_t dst_max, const void* src, size_t count, aclrtMemcpyKind)
{
    Record("memcpy");
    if (Fail("memcpy")) return 1;
    if (dst == nullptr || src == nullptr || count > dst_max)
    {
        last_error = "invalid memcpy";
        return 1;
    }
    std::copy_n(static_cast<const unsigned char*>(src), count, static_cast<unsigned char*>(dst));
    return ACL_SUCCESS;
}
extern "C" aclError aclmdlLoadFromFile(const char*, uint32_t* model_id)
{
    Record("load");
    if (Fail("load")) return 1;
    *model_id = next_model_id++;
    ++resources.loaded_models;
    return ACL_SUCCESS;
}
extern "C" aclError aclmdlUnload(uint32_t)
{
    Record("unload");
    --resources.loaded_models;
    return ACL_SUCCESS;
}
extern "C" aclmdlDesc* aclmdlCreateDesc(void)
{
    Record("create_desc");
    return new aclmdlDesc{model_spec};
}
extern "C" aclError aclmdlDestroyDesc(aclmdlDesc* desc)
{
    Record("destroy_desc");
    delete desc;
    return ACL_SUCCESS;
}
extern "C" aclError aclmdlGetDesc(aclmdlDesc*, uint32_t)
{
    Record("get_desc");
    return ACL_SUCCESS;
}
extern "C" size_t aclmdlGetNumInputs(const aclmdlDesc* desc)
{
    return desc->spec.input_sizes.size();
}
extern "C" size_t aclmdlGetNumOutputs(const aclmdlDesc* desc)
{
    return desc->spec.output_sizes.size();
}
extern "C" size_t aclmdlGetInputSizeByIndex(const aclmdlDesc* desc, size_t index)
{
    return index < desc->spec.input_sizes.size() ? desc->spec.input_sizes[index] : 0;
}
extern "C" size_t aclmdlGetOutputSizeByIndex(const aclmdlDesc* desc, size_t index)
{
    return index < desc->spec.output_sizes.size() ? desc->spec.output_sizes[index] : 0;
}
extern "C" aclDataType aclmdlGetInputDataType(const aclmdlDesc* desc, size_t index)
{
    return index < desc->spec.input_types.size() ? desc->spec.input_types[index] : -1;
}
extern "C" aclDataType aclmdlGetOutputDataType(const aclmdlDesc* desc, size_t index)
{
    return index < desc->spec.output_types.size() ? desc->spec.output_types[index] : -1;
}
extern "C" aclmdlDataset* aclmdlCreateDataset(void)
{
    Record("create_dataset");
    ++resources.datasets;
    return new aclmdlDataset;
}
extern "C" aclError aclmdlDestroyDataset(aclmdlDataset* dataset)
{
    Record("destroy_dataset");
    delete dataset;
    --resources.datasets;
    return ACL_SUCCESS;
}
extern "C" aclError aclmdlAddDatasetBuffer(aclmdlDataset* dataset, aclDataBuffer* buffer)
{
    Record("add_buffer");
    dataset->buffers.push_back(buffer);
    return ACL_SUCCESS;
}
extern "C" aclError aclmdlExecute(uint32_t, aclmdlDataset*, aclmdlDataset* outputs)
{
    Record("execute");
    if (Fail("execute")) return 1;
    for (size_t index = 0; index < outputs->buffers.size(); ++index)
    {
        auto* buffer = outputs->buffers[index];
        auto* values = static_cast<float*>(buffer->data);
        const size_t count = buffer->size / sizeof(float);
        std::fill_n(values, count, 0.0F);
        if (index < model_spec.output_values.size())
        {
            const auto& expected = model_spec.output_values[index];
            if (expected.size() != count) return 1;
            std::copy(expected.begin(), expected.end(), values);
        }
    }
    return ACL_SUCCESS;
}
extern "C" aclDataBuffer* aclCreateDataBuffer(void* data, size_t size)
{
    Record("create_data");
    ++data_buffer_calls;
    if (model_spec.fail_data_buffer_at == data_buffer_calls)
    {
        last_error = "injected data buffer failure";
        return nullptr;
    }
    ++resources.data_buffers;
    return new aclDataBuffer{data, size};
}
extern "C" aclError aclDestroyDataBuffer(aclDataBuffer* buffer)
{
    Record("destroy_data");
    delete buffer;
    --resources.data_buffers;
    return ACL_SUCCESS;
}
