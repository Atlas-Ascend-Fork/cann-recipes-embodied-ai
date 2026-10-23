/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "fake_acl_runtime.hpp"
#include "om_engine.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace
{
constexpr size_t NO_EVENT = static_cast<size_t>(-1);
void Check(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}
size_t FindEvent(const std::vector<std::string>& events, const char* name, size_t start = 0)
{
    for (size_t index = start; index < events.size(); ++index)
    {
        if (events[index] == name)
        {
            return index;
        }
    }
    return NO_EVENT;
}
void ExpectFailure(const std::function<void()>& operation)
{
    bool failed = false;
    try
    {
        operation();
    }
    catch (const std::exception&)
    {
        failed = true;
    }
    Check(failed, "operation unexpectedly succeeded");
}
void TestForwardAndReleaseOrder()
{
    fake_acl::Reset();
    {
        OmEngine engine;
        engine.LoadModel("policy.om");
        Check(engine.IsModelLoaded(), "model did not load");
        const auto outputs = engine.Forward({1.0F, 2.0F}, 0.25F);
        Check(outputs.size() == 5, "unexpected output count");
        Check(outputs[4].size() == 32, "unexpected body quaternion output size");
    }
    const auto& events = fake_acl::Events();
    const size_t data = FindEvent(events, "destroy_data");
    const size_t dataset = FindEvent(events, "destroy_dataset");
    const size_t free = FindEvent(events, "free");
    const size_t unload = FindEvent(events, "unload");
    const size_t desc = FindEvent(events, "destroy_desc");
    Check(data != NO_EVENT && dataset != NO_EVENT && free != NO_EVENT, "missing buffer cleanup event");
    Check(unload != NO_EVENT && desc != NO_EVENT, "missing model cleanup event");
    Check(data < dataset && dataset < free && free < unload && unload < desc, "invalid cleanup order");
}
void TestReloadRestoresContext()
{
    fake_acl::Reset();
    OmEngine engine;
    engine.LoadModel("first.om");
    fake_acl::SwitchToForeignContext();
    const size_t reload_start = fake_acl::Events().size();
    engine.LoadModel("second.om");
    const size_t set_context = FindEvent(fake_acl::Events(), "set_context", reload_start);
    const size_t destroy_data = FindEvent(fake_acl::Events(), "destroy_data", reload_start);
    Check(set_context != NO_EVENT && destroy_data != NO_EVENT, "missing reload cleanup event");
    Check(set_context < destroy_data, "reload did not restore context before cleanup");
}
void TestDestructorRestoresContext()
{
    fake_acl::Reset();
    {
        OmEngine engine;
        engine.LoadModel("policy.om");
        fake_acl::SwitchToForeignContext();
    }
    const auto& events = fake_acl::Events();
    const size_t foreign = FindEvent(events, "foreign_context");
    const size_t set_context = FindEvent(events, "set_context", foreign + 1);
    const size_t destroy_data = FindEvent(events, "destroy_data", foreign + 1);
    Check(foreign != NO_EVENT && set_context != NO_EVENT && destroy_data != NO_EVENT,
          "missing destructor context event");
    Check(set_context < destroy_data, "destructor did not restore context before cleanup");
}
void TestDtypeAndShapeValidation()
{
    fake_acl::ModelSpec spec;
    spec.input_types[1] = ACL_FLOAT16;
    fake_acl::Reset(spec);
    ExpectFailure([] { OmEngine engine; engine.LoadModel("bad-input-dtype.om"); });
    spec = {};
    spec.output_types[0] = ACL_FLOAT16;
    fake_acl::Reset(spec);
    ExpectFailure([] { OmEngine engine; engine.LoadModel("bad-output-dtype.om"); });
    spec = {};
    spec.input_sizes.clear();
    spec.input_types.clear();
    fake_acl::Reset(spec);
    ExpectFailure([] { OmEngine engine; engine.LoadModel("empty-input.om"); });
    spec = {};
    spec.output_sizes.clear();
    spec.output_types.clear();
    fake_acl::Reset(spec);
    ExpectFailure([] { OmEngine engine; engine.LoadModel("empty-output.om"); });
}
void TestForwardInputValidation()
{
    fake_acl::Reset();
    OmEngine engine;
    engine.LoadModel("policy.om");
    ExpectFailure([&] { engine.Forward({}, 0.0F); });
    ExpectFailure([&] { engine.Forward({1.0F, std::numeric_limits<float>::quiet_NaN()}, 0.0F); });
    ExpectFailure([&] { engine.Forward({1.0F, 2.0F}, std::numeric_limits<float>::infinity()); });
}
void TestPartialAllocationCleanup()
{
    fake_acl::ModelSpec spec;
    spec.fail_malloc_at = 2;
    fake_acl::Reset(spec);
    ExpectFailure([] { OmEngine engine; engine.LoadModel("malloc-failure.om"); });
    Check(fake_acl::Resources().device_buffers == 0, "device buffer leaked after malloc failure");
    Check(fake_acl::Resources().data_buffers == 0, "data buffer leaked after malloc failure");
    Check(fake_acl::Resources().datasets == 0, "dataset leaked after malloc failure");
    spec = {};
    spec.fail_data_buffer_at = 2;
    fake_acl::Reset(spec);
    ExpectFailure([] { OmEngine engine; engine.LoadModel("data-buffer-failure.om"); });
    Check(fake_acl::Resources().device_buffers == 0, "device buffer leaked after data failure");
    Check(fake_acl::Resources().data_buffers == 0, "data buffer leaked after data failure");
    Check(fake_acl::Resources().datasets == 0, "dataset leaked after data failure");
}
}
static_assert(!std::is_copy_constructible_v<OmEngine>);
static_assert(!std::is_copy_assignable_v<OmEngine>);
static_assert(!std::is_move_constructible_v<OmEngine>);
static_assert(!std::is_move_assignable_v<OmEngine>);
int main()
{
    TestForwardAndReleaseOrder();
    TestReloadRestoresContext();
    TestDestructorRestoresContext();
    TestDtypeAndShapeValidation();
    TestForwardInputValidation();
    TestPartialAllocationCleanup();
    std::cout << "engine-stage tests passed\n";
}
