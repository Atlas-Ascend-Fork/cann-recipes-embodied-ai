/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ENGINE_STAGE_FAKE_ACL_H
#define ENGINE_STAGE_FAKE_ACL_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef int32_t aclError;
typedef int32_t aclDataType;
typedef void* aclrtContext;
typedef int32_t aclrtMemMallocPolicy;
typedef int32_t aclrtMemcpyKind;
typedef struct aclmdlDesc aclmdlDesc;
typedef struct aclmdlDataset aclmdlDataset;
typedef struct aclDataBuffer aclDataBuffer;
enum {
    ACL_SUCCESS = 0,
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_MEM_MALLOC_HUGE_FIRST = 0,
    ACL_MEMCPY_HOST_TO_DEVICE = 1,
    ACL_MEMCPY_DEVICE_TO_HOST = 2,
};
aclError aclInit(const char* config_path);
aclError aclFinalize(void);
const char* aclGetRecentErrMsg(void);
aclError aclrtSetDevice(int32_t device_id);
aclError aclrtResetDevice(int32_t device_id);
aclError aclrtCreateContext(aclrtContext* context, int32_t device_id);
aclError aclrtSetCurrentContext(aclrtContext context);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtMalloc(void** device_ptr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtFree(void* device_ptr);
aclError aclrtMemcpy(void* dst, size_t dst_max, const void* src, size_t count, aclrtMemcpyKind kind);
aclError aclmdlLoadFromFile(const char* model_path, uint32_t* model_id);
aclError aclmdlUnload(uint32_t model_id);
aclmdlDesc* aclmdlCreateDesc(void);
aclError aclmdlDestroyDesc(aclmdlDesc* desc);
aclError aclmdlGetDesc(aclmdlDesc* desc, uint32_t model_id);
size_t aclmdlGetNumInputs(const aclmdlDesc* desc);
size_t aclmdlGetNumOutputs(const aclmdlDesc* desc);
size_t aclmdlGetInputSizeByIndex(const aclmdlDesc* desc, size_t index);
size_t aclmdlGetOutputSizeByIndex(const aclmdlDesc* desc, size_t index);
aclDataType aclmdlGetInputDataType(const aclmdlDesc* desc, size_t index);
aclDataType aclmdlGetOutputDataType(const aclmdlDesc* desc, size_t index);
aclmdlDataset* aclmdlCreateDataset(void);
aclError aclmdlDestroyDataset(aclmdlDataset* dataset);
aclError aclmdlAddDatasetBuffer(aclmdlDataset* dataset, aclDataBuffer* buffer);
aclError aclmdlExecute(uint32_t model_id, aclmdlDataset* input, aclmdlDataset* output);
aclDataBuffer* aclCreateDataBuffer(void* data, size_t size);
aclError aclDestroyDataBuffer(aclDataBuffer* buffer);
#ifdef __cplusplus
}
#endif
#endif
