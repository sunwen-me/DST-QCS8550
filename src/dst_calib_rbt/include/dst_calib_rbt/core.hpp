#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dst_calib_rbt {

struct LidarInput {
    inline static constexpr char TYPE_NAME[] = "Mid360PointCloud";
};

using DepthInput = struct DepthInputCapability {
    inline static constexpr char TYPE_NAME[] = "rbt.orbbec.Depth@1";
};

using CameraInfoInput = struct CameraInfoInputCapability {
    inline static constexpr char TYPE_NAME[] = "rbt.orbbec.CameraInfo@1";
};

struct ExtrinsicCapability {
    inline static constexpr char TYPE_NAME[] = "dst_calib.Extrinsic@1";
};

struct StatusCapability {
    inline static constexpr char TYPE_NAME[] = "dst_calib.Status@1";
};

struct Options {
    std::string component_id = "dst-calib";
    std::string lidar_capability = LidarInput::TYPE_NAME;
    std::string depth_capability = DepthInput::TYPE_NAME;
    std::string camera_info_capability = CameraInfoInput::TYPE_NAME;
    std::string parent_frame = "camera_color_optical_frame";
    std::string child_frame = "livox_frame";
    std::int64_t max_skew_ns = 50'000'000;
    std::size_t window_size = 12;
    std::size_t window_stride = 4;
    std::size_t max_lidar_points = 12'000;
    std::size_t min_lidar_points = 64;
    float depth_min_m = 0.25F;
    float depth_max_m = 6.0F;
    std::string python = "python3";
    std::string worker_script = "src/dst_calib_rbt/scripts/dst_worker.py";
    std::string worker_config = "src/dst_calib_rbt/python/default.yaml";
    std::string mode = "selfsup";
    std::string backend = "auto";
    std::string eva_onnx = "models/dst-calib/eva_sb.onnx";
    std::string eva_ckpt = "models/dst-calib/kitti-validation-best/best.pt";
    std::string qnn_context = "models/dst-calib/kitti-validation-best/eva_sb_qairt250.bin";
    std::string qnn_runtime = "runtime/qairt250";
    std::string qnn_library = "build/dst-calib/lib/libdst_qnn.so";
    std::string qnn_manifest;
    std::string init_yaml;
    std::string output_directory = "runs/dst-calib";
    double max_final_cd = 0.15;
    bool publish_results = false;
};

Options load_options(const std::string& path, const std::string& override_id = {});
void validate_options(const Options& options);

} // namespace dst_calib_rbt
