#include "dst_calib_rbt/core.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace dst_calib_rbt {
namespace {

template <typename T>
T value(const YAML::Node& node, const char* key, T fallback) {
    return node[key] ? node[key].as<T>() : fallback;
}

} // namespace

Options load_options(const std::string& path, const std::string& override_id) {
    const auto root = YAML::LoadFile(path);
    if (!root.IsMap()) throw std::invalid_argument("DST config must be a YAML mapping");
    Options result;
    result.component_id = value(root, "component_id", result.component_id);
    result.lidar_capability = value(root, "lidar_capability", result.lidar_capability);
    result.depth_capability = value(root, "depth_capability", result.depth_capability);
    result.camera_info_capability = value(root, "camera_info_capability", result.camera_info_capability);
    if (root["frames"]) {
        result.parent_frame = value(root["frames"], "parent", result.parent_frame);
        result.child_frame = value(root["frames"], "child", result.child_frame);
    }
    if (root["pairing"]) {
        result.max_skew_ns = value(root["pairing"], "max_skew_ns", result.max_skew_ns);
        result.window_size = value(root["pairing"], "window_size", result.window_size);
        result.window_stride = value(root["pairing"], "window_stride", result.window_stride);
        result.max_lidar_points = value(root["pairing"], "max_lidar_points", result.max_lidar_points);
        result.min_lidar_points = value(root["pairing"], "min_lidar_points", result.min_lidar_points);
        result.depth_min_m = value(root["pairing"], "depth_min_m", result.depth_min_m);
        result.depth_max_m = value(root["pairing"], "depth_max_m", result.depth_max_m);
    }
    if (root["worker"]) {
        result.python = value(root["worker"], "python", result.python);
        result.worker_script = value(root["worker"], "script", result.worker_script);
        result.worker_config = value(root["worker"], "config", result.worker_config);
        result.mode = value(root["worker"], "mode", result.mode);
        result.backend = value(root["worker"], "backend", result.backend);
        result.eva_onnx = value(root["worker"], "eva_onnx", result.eva_onnx);
        result.eva_ckpt = value(root["worker"], "eva_ckpt", result.eva_ckpt);
        result.qnn_context = value(root["worker"], "qnn_context", result.qnn_context);
        result.qnn_runtime = value(root["worker"], "qnn_runtime", result.qnn_runtime);
        result.qnn_library = value(root["worker"], "qnn_library", result.qnn_library);
        result.qnn_manifest = value(root["worker"], "qnn_manifest", result.qnn_manifest);
        result.init_yaml = value(root["worker"], "init_yaml", result.init_yaml);
    }
    if (root["output"]) {
        result.output_directory = value(root["output"], "directory", result.output_directory);
        result.max_final_cd = value(root["output"], "max_final_cd", result.max_final_cd);
        result.publish_results = value(root["output"], "publish_results", result.publish_results);
    }
    if (!override_id.empty()) result.component_id = override_id;
    validate_options(result);
    return result;
}

void validate_options(const Options& options) {
    if (options.component_id.empty() || options.component_id.size() > 63)
        throw std::invalid_argument("component_id must contain 1..63 bytes");
    if (options.parent_frame.empty() || options.parent_frame.size() > 63
        || options.child_frame.empty() || options.child_frame.size() > 63
        || options.parent_frame == options.child_frame)
        throw std::invalid_argument("frames must be distinct names of 1..63 bytes");
    if (options.max_skew_ns < 0 || options.max_skew_ns > 1'000'000'000)
        throw std::invalid_argument("pairing.max_skew_ns is outside 0..1e9");
    if (options.window_size < 2 || options.window_size > 128 || options.window_stride == 0
        || options.window_stride > options.window_size)
        throw std::invalid_argument("window_size/window_stride are invalid");
    if (options.max_lidar_points < options.min_lidar_points || options.max_lidar_points > 1'000'000
        || options.min_lidar_points < 16)
        throw std::invalid_argument("LiDAR point limits are invalid");
    if (!std::isfinite(options.depth_min_m) || !std::isfinite(options.depth_max_m)
        || options.depth_min_m <= 0.0F || options.depth_max_m <= options.depth_min_m)
        throw std::invalid_argument("depth range is invalid");
    if (options.python.empty() || options.worker_script.empty() || options.worker_config.empty())
        throw std::invalid_argument("worker paths must not be empty");
    if (options.mode != "selfsup" && options.mode != "fast" && options.mode != "both")
        throw std::invalid_argument("worker.mode must be selfsup, fast, or both");
    if (options.backend != "auto" && options.backend != "qnn" && options.backend != "cpu")
        throw std::invalid_argument("worker.backend must be auto, qnn, or cpu");
    if (options.backend == "qnn" && options.mode == "selfsup")
        throw std::invalid_argument("QNN runs the fast/both evaluator only, not selfsup autograd");
    if (options.backend == "qnn" && (options.qnn_context.empty() || options.qnn_runtime.empty()
                                     || options.qnn_library.empty()))
        throw std::invalid_argument("QNN context, runtime, and library must not be empty");
    if (!std::isfinite(options.max_final_cd) || options.max_final_cd < 0.0)
        throw std::invalid_argument("output.max_final_cd must be finite and non-negative");
}

} // namespace dst_calib_rbt
