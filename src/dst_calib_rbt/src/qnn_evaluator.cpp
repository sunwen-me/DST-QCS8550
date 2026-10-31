// Persistent, HTP-only evaluator C ABI used by the isolated Python worker.
#include <qnn_runtime/session.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
struct Evaluator {
    qnn_runtime::Session session;
    std::string input, output;
    bool nhwc;
    std::vector<float> scratch;
    Evaluator(const char* runtime, const char* context, bool profile) : session(runtime, profile) {
        session.load(context);
        const auto inputs = session.inputs(), outputs = session.outputs();
        if (inputs.size() != 1 || outputs.size() != 1)
            throw std::runtime_error("DST requires exactly one input and one output");
        input = inputs.front(); output = outputs.front();
        const auto& shape = session.tensor(input).shape;
        nhwc = shape == qnn_runtime::Shape{1, 256, 512, 3};
        if (!nhwc && shape != qnn_runtime::Shape{1, 3, 256, 512})
            throw std::runtime_error("unsupported DST input shape: " + qnn_runtime::shape_string(shape));
        if (session.tensor(output).elements() != 6)
            throw std::runtime_error("DST output must contain six pose values");
        scratch.resize(3 * 256 * 512);
    }
};
}
extern "C" {
const char* dst_qnn_error() noexcept { return error.c_str(); }
void* dst_qnn_create(const char* runtime, const char* context, int profile) noexcept {
    try { error.clear(); return new Evaluator(runtime, context, profile != 0); }
    catch (const std::exception& e) { error = e.what(); return nullptr; }
}
int dst_qnn_layout(void* handle) noexcept { return static_cast<Evaluator*>(handle)->nhwc ? 1 : 0; }
int dst_qnn_infer(void* handle, const float* input, std::size_t count, float* output) noexcept {
    try {
        error.clear();
        if (!handle || !input || !output || count != 3 * 256 * 512)
            throw std::runtime_error("invalid DST input buffer");
        auto& e = *static_cast<Evaluator*>(handle);
        for (std::size_t i = 0; i < count; ++i)
            if (!std::isfinite(input[i])) throw std::runtime_error("nonfinite DST input");
        if (e.nhwc) {
            constexpr std::size_t pixels = 256 * 512;
            for (std::size_t p = 0; p < pixels; ++p)
                for (std::size_t c = 0; c < 3; ++c) e.scratch[p * 3 + c] = input[c * pixels + p];
            e.session.write_f32(e.input, e.scratch);
        } else e.session.write_f32(e.input, std::span<const float>(input, count));
        e.session.run();
        const auto values = e.session.read_f32(e.output);
        if (!std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("nonfinite DST output");
        std::copy(values.begin(), values.end(), output);
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
int dst_qnn_profile(void* handle, const char* path) noexcept {
    try { error.clear(); static_cast<Evaluator*>(handle)->session.write_profile(path); return 0; }
    catch (const std::exception& e) { error = e.what(); return -1; }
}
void dst_qnn_destroy(void* handle) noexcept { delete static_cast<Evaluator*>(handle); }
}
