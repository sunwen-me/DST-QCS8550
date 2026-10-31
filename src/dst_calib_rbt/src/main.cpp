#include "dst_calib_rbt/core.hpp"
#include "dst_calib_rbt/sampling.hpp"

#include <dst_calib_rbt/orbbec_input.hpp>
#include <rbt/dst_calib.hpp>
#include <rbt/mid360.hpp>
#include <rbt/orbbec.hpp>
#include <rbt/rpc.hpp>
#include <rbt/runtime.hpp>
#include <rbt/tf_component.hpp>
#include <rbt/time.hpp>
#include <rbt/time_stream.hpp>
#include <rbt/wait.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace dst_calib_rbt {
namespace {

using ExtrinsicPublisher = rbt::FixedPublisher<dst_calib::Extrinsic>;
using StatusPublisher = rbt::FixedPublisher<dst_calib::Status>;

struct ExtrinsicTfCapability {
    inline static constexpr char TYPE_NAME[] = "dst_calib.ExtrinsicTf@1";
};

struct CameraModel {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

struct LidarPacket {
    std::uint64_t timestamp_ns = 0;
    std::vector<std::array<float, 3>> points;
};

struct DepthPacket {
    std::uint64_t timestamp_ns = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> depth_m;
};

struct Frame {
    std::uint64_t timestamp_ns = 0;
    std::vector<std::array<float, 3>> points;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> depth_m;
};

struct Snapshot {
    std::vector<Frame> frames;
    CameraModel camera;
};

struct Job {
    std::uint64_t sequence = 0;
    Snapshot snapshot;
};

struct Pending {
    std::mutex mutex;
    std::condition_variable_any changed;
    std::deque<LidarPacket> lidars;
    std::deque<DepthPacket> depths;
    std::deque<Frame> window;
    std::deque<Job> jobs;
    std::size_t frames_since_job = 0;
    std::size_t paired_frames = 0;
    std::size_t dropped_frames = 0;
    std::size_t rejected_results = 0;
    std::uint64_t next_job = 0;
    bool running = false;
};

template <typename Result>
auto take(Result&& result, const char* action) {
    if (!result) {
        throw std::runtime_error(std::string(action) + ": " + std::string(rbt::to_string(result.error())));
    }
    return std::move(result).value();
}

template <typename T>
T take_time(rbt::time::Result<T>&& result, const char* action) {
    if (!result) {
        throw std::runtime_error(std::string(action) + ": " +
                                 std::string(rbt::time::to_string(result.error())));
    }
    return std::move(result).value();
}

template <class Tag, class Factory>
auto provide(rbt::ActivationContext& context, const rbt::MessageTypeDescriptor& type,
             Factory&& factory) {
    const auto capability = rbt::generate_uuid();
    const auto descriptor = rbt::OwnedStreamDescriptor::from(rbt::StreamDescriptor(
        "rbt/cap/" + capability + "/stream", type, rbt::StreamProfile::Sensor));
    auto publisher = take(factory(descriptor.borrowed()), "create DST publisher");
    if (!context.provide<Tag>(capability, descriptor)) {
        throw std::runtime_error("provide DST capability failed");
    }
    return publisher;
}

std::string shell_quote(const std::string& value) {
    std::string result = "'";
    for (const char c : value) {
        if (c == '\'') result += "'\\''";
        else result += c;
    }
    result += '\'';
    return result;
}

void write_binary(std::ofstream& out, const auto& value) {
    out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(value)));
    if (!out) throw std::runtime_error("failed to write DST worker input");
}

void write_window(const std::filesystem::path& path, const Snapshot& snapshot) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create DST worker input: " + path.string());
    const std::array<char, 8> magic {'D', 'S', 'T', 'W', 'I', 'N', '0', '1'};
    out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::uint32_t version = 1;
    const std::uint32_t count = static_cast<std::uint32_t>(snapshot.frames.size());
    write_binary(out, version);
    write_binary(out, count);
    write_binary(out, snapshot.camera.width);
    write_binary(out, snapshot.camera.height);
    write_binary(out, snapshot.camera.fx);
    write_binary(out, snapshot.camera.fy);
    write_binary(out, snapshot.camera.cx);
    write_binary(out, snapshot.camera.cy);
    for (const auto& frame : snapshot.frames) {
        write_binary(out, frame.timestamp_ns);
        const auto point_count = static_cast<std::uint32_t>(frame.points.size());
        write_binary(out, point_count);
        for (const auto& point : frame.points) {
            for (const float value : point) write_binary(out, value);
        }
        const auto depth_count = static_cast<std::uint32_t>(frame.depth_m.size());
        write_binary(out, depth_count);
        for (const float value : frame.depth_m) write_binary(out, value);
    }
    out.flush();
    if (!out) throw std::runtime_error("failed to flush DST worker input");
}

struct WorkerResult {
    std::array<double, 16> transform {};
    std::array<double, 3> rotation_deg {};
    std::array<double, 3> translation_m {};
    double final_cd = std::numeric_limits<double>::quiet_NaN();
    std::string mode;
    std::string backend;
    bool pipeline_approved = true;
};

std::array<double, 16> read_transform(const YAML::Node& node) {
    if (!node || !node.IsSequence() || node.size() != 4) {
        throw std::runtime_error("worker result T_cam_lidar must be a 4x4 matrix");
    }
    std::array<double, 16> result {};
    for (std::size_t row = 0; row < 4; ++row) {
        if (!node[row].IsSequence() || node[row].size() != 4)
            throw std::runtime_error("worker result T_cam_lidar must be a 4x4 matrix");
        for (std::size_t col = 0; col < 4; ++col)
            result[row * 4 + col] = node[row][col].as<double>();
    }
    return result;
}

WorkerResult read_result(const std::filesystem::path& path) {
    const auto root = YAML::LoadFile(path.string());
    WorkerResult result;
    result.transform = read_transform(root["T_cam_lidar"]);
    for (double value : result.transform) {
        if (!std::isfinite(value)) throw std::runtime_error("worker result has a non-finite transform");
    }
    if (root["rotation_deg"]) {
        for (std::size_t i = 0; i < 3; ++i) result.rotation_deg[i] = root["rotation_deg"][i].as<double>();
    } else if (root["euler_zyx_deg"]) {
        for (std::size_t i = 0; i < 3; ++i) result.rotation_deg[i] = root["euler_zyx_deg"][i].as<double>();
    }
    if (root["translation_m"])
        for (std::size_t i = 0; i < 3; ++i) result.translation_m[i] = root["translation_m"][i].as<double>();
    else
        for (std::size_t i = 0; i < 3; ++i) result.translation_m[i] = result.transform[i * 4 + 3];
    result.final_cd = root["final_cd"] ? root["final_cd"].as<double>() : 0.0;
    result.mode = root["mode"] ? root["mode"].as<std::string>() : "selfsup";
    result.backend = root["backend"] ? root["backend"].as<std::string>() : "cpu";
    if (result.backend == "qnn") {
        const auto evaluator = root["evaluator"];
        result.pipeline_approved = evaluator && evaluator["pipeline_approved"]
            && evaluator["pipeline_approved"].as<bool>();
    }
    return result;
}

std::uint32_t backend_id(std::string_view backend) {
    if (backend == "qnn" || backend == "QNNExecutionProvider") return 2;
    if (backend == "cuda" || backend == "CUDAExecutionProvider") return 3;
    return 1;
}

void copy_text(std::uint8_t* destination, std::size_t capacity, std::string_view value) {
    std::fill(destination, destination + capacity, 0);
    const auto count = std::min(capacity - 1, value.size());
    std::copy_n(value.begin(), count, destination);
}

void copy_clock(std::uint8_t* destination, const rbt::time::ClockIdentity& clock) {
    std::copy(clock.source_id().bytes().begin(), clock.source_id().bytes().end(), destination);
}

void check_transform(const std::array<double, 16>& transform) {
    if (std::abs(transform[12]) > 1e-6 || std::abs(transform[13]) > 1e-6
        || std::abs(transform[14]) > 1e-6 || std::abs(transform[15] - 1.0) > 1e-6)
        throw std::runtime_error("worker result has an invalid homogeneous bottom row");
    for (std::size_t row = 0; row < 3; ++row) {
        double norm = 0.0;
        for (std::size_t col = 0; col < 3; ++col) norm += transform[row * 4 + col] * transform[row * 4 + col];
        if (std::abs(norm - 1.0) > 2e-2) throw std::runtime_error("worker result rotation is not normalized");
    }
}

rbt::tf::Quaternion quaternion_from_rotation(const std::array<double, 16>& transform) {
    const double trace = transform[0] + transform[5] + transform[10];
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (trace > 0.0) {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        w = 0.25 * s;
        x = (transform[9] - transform[6]) / s;
        y = (transform[2] - transform[8]) / s;
        z = (transform[4] - transform[1]) / s;
    } else if (transform[0] > transform[5] && transform[0] > transform[10]) {
        const double s = std::sqrt(1.0 + transform[0] - transform[5] - transform[10]) * 2.0;
        w = (transform[9] - transform[6]) / s;
        x = 0.25 * s;
        y = (transform[1] + transform[4]) / s;
        z = (transform[2] + transform[8]) / s;
    } else if (transform[5] > transform[10]) {
        const double s = std::sqrt(1.0 + transform[5] - transform[0] - transform[10]) * 2.0;
        w = (transform[2] - transform[8]) / s;
        x = (transform[1] + transform[4]) / s;
        y = 0.25 * s;
        z = (transform[6] + transform[9]) / s;
    } else {
        const double s = std::sqrt(1.0 + transform[10] - transform[0] - transform[5]) * 2.0;
        w = (transform[4] - transform[1]) / s;
        x = (transform[2] + transform[8]) / s;
        y = (transform[6] + transform[9]) / s;
        z = 0.25 * s;
    }
    const double norm = std::sqrt(w * w + x * x + y * y + z * z);
    if (!std::isfinite(norm) || norm < 1e-12) throw std::runtime_error("worker result quaternion is invalid");
    return rbt::tf::Quaternion(x / norm, y / norm, z / norm, w / norm);
}

std::filesystem::path run_worker(const Options& options, const Snapshot& snapshot, std::uint64_t sequence) {
    const auto job_dir = std::filesystem::path(options.output_directory) /
        ("job_" + std::to_string(sequence));
    std::filesystem::create_directories(job_dir);
    const auto input = job_dir / "window.bin";
    const auto output = job_dir / "extrinsic.yaml";
    write_window(input, snapshot);
    std::ostringstream command;
    command << shell_quote(options.python) << ' ' << shell_quote(options.worker_script)
            << " --input " << shell_quote(input.string())
            << " --output " << shell_quote(output.string())
            << " --config " << shell_quote(options.worker_config)
            << " --mode " << shell_quote(options.mode)
            << " --backend " << shell_quote(options.backend)
            << " --eva-onnx " << shell_quote(options.eva_onnx)
            << " --eva-ckpt " << shell_quote(options.eva_ckpt)
            << " --qnn-context " << shell_quote(options.qnn_context)
            << " --qnn-runtime " << shell_quote(options.qnn_runtime)
            << " --qnn-library " << shell_quote(options.qnn_library)
            << " --qnn-manifest " << shell_quote(options.qnn_manifest);
    if (!options.init_yaml.empty()) command << " --init-yaml " << shell_quote(options.init_yaml);
    const auto code = std::system(command.str().c_str());
    if (code != 0) throw std::runtime_error("DST Python worker failed with exit code " + std::to_string(code));
    if (!std::filesystem::is_regular_file(output)) throw std::runtime_error("DST worker did not write extrinsic.yaml");
    return output;
}

void pair_queues(Pending& pending, const Options& options, const CameraModel& camera) {
    while (!pending.lidars.empty() && !pending.depths.empty()) {
        auto lidar_it = pending.lidars.begin();
        auto depth_it = pending.depths.begin();
        auto best_delta = std::numeric_limits<std::uint64_t>::max();
        for (auto it = pending.depths.begin(); it != pending.depths.end(); ++it) {
            const auto delta = it->timestamp_ns > lidar_it->timestamp_ns
                ? it->timestamp_ns - lidar_it->timestamp_ns
                : lidar_it->timestamp_ns - it->timestamp_ns;
            if (delta < best_delta) { best_delta = delta; depth_it = it; }
        }
        if (best_delta > static_cast<std::uint64_t>(options.max_skew_ns)) {
            if (lidar_it->timestamp_ns + static_cast<std::uint64_t>(options.max_skew_ns)
                < pending.depths.front().timestamp_ns) {
                pending.lidars.pop_front();
                ++pending.dropped_frames;
                continue;
            }
            if (pending.depths.front().timestamp_ns + static_cast<std::uint64_t>(options.max_skew_ns)
                < lidar_it->timestamp_ns) {
                pending.depths.pop_front();
                ++pending.dropped_frames;
                continue;
            }
            break;
        }
        Frame frame;
        frame.timestamp_ns = (lidar_it->timestamp_ns + depth_it->timestamp_ns) / 2;
        frame.points = std::move(lidar_it->points);
        frame.width = depth_it->width;
        frame.height = depth_it->height;
        frame.depth_m = std::move(depth_it->depth_m);
        pending.lidars.erase(lidar_it);
        pending.depths.erase(depth_it);
        if (frame.points.size() < options.min_lidar_points || frame.depth_m.empty()) {
            ++pending.dropped_frames;
            continue;
        }
        pending.window.push_back(std::move(frame));
        while (pending.window.size() > options.window_size) pending.window.pop_front();
        ++pending.paired_frames;
        ++pending.frames_since_job;
        if (pending.window.size() == options.window_size && pending.frames_since_job >= options.window_stride) {
            pending.frames_since_job = 0;
            if (pending.running || !pending.jobs.empty()) {
                ++pending.dropped_frames;
                continue;
            }
            Job job;
            job.sequence = ++pending.next_job;
            job.snapshot.camera = camera;
            job.snapshot.frames.assign(pending.window.begin(), pending.window.end());
            pending.jobs.push_back(std::move(job));
            pending.changed.notify_one();
        }
    }
}

class Component final : public rbt::Component {
public:
    explicit Component(Options options) : options_(std::move(options)) {}
    rbt::ComponentId id() const override { return options_.component_id; }
    std::vector<rbt::RequirementSpec> requirements() const override {
        return {{"lidar", options_.lidar_capability}, {"depth", options_.depth_capability},
                {"camera_info", options_.camera_info_capability}};
    }
    rbt::Result<void> prepare() override {
        try { validate_options(options_); return {}; }
        catch (const std::exception& error) { return rbt::Result<void>::failure(error.what()); }
    }

    rbt::Result<void> activate(rbt::ActivationContext& context) override {
        try {
            auto lidar = take(context.require<LidarInput>("lidar").slice_subscriber<
                mid360::CloudHeader, mid360::TimedPoint>(), "open Mid360 subscriber");
            auto depth = take(context.require<DepthInput>("depth").slice_subscriber<std::uint64_t, std::uint8_t>(),
                              "open depth subscriber");
            const auto camera_binding = context.require<CameraInfoInput>("camera_info");
            if (!camera_binding) return rbt::Result<void>::failure("missing CameraInfo capability");

            using CameraInfoRpc = rbt::RpcClient<rbt::orbbec::CameraInfoRequest,
                                                 rbt::orbbec::CameraInfoResponse>;
            auto camera_descriptor = camera_binding.rpc().borrowed();
            camera_descriptor.request_type.fields =
                rbt::FixedMessageBinding<rbt::orbbec::CameraInfoRequest>::descriptor().fields;
            camera_descriptor.response_type.fields =
                rbt::FixedMessageBinding<rbt::orbbec::CameraInfoResponse>::descriptor().fields;
            auto camera_rpc = take(CameraInfoRpc::open_for_activation(
                camera_descriptor, camera_binding.capability_instance(),
                camera_binding.binding().descriptor.activation), "open CameraInfo RPC");
            const auto color_info = take(camera_rpc.call_copy(
                {1U, 0U}, std::chrono::seconds(1)), "read color CameraInfo");
            CameraModel camera;
            camera.width = color_info.payload().width;
            camera.height = color_info.payload().height;
            camera.fx = color_info.payload().fx;
            camera.fy = color_info.payload().fy;
            camera.cx = color_info.payload().cx;
            camera.cy = color_info.payload().cy;
            if (camera.width == 0 || camera.height == 0 || camera.fx <= 0.0 || camera.fy <= 0.0)
                throw std::runtime_error("CameraInfo returned invalid color intrinsics");

            auto extrinsic = provide<ExtrinsicCapability>(context, dst_calib::EXTRINSIC_DESCRIPTOR,
                [](auto descriptor) { return ExtrinsicPublisher::create(descriptor, dst_calib::EXTRINSIC_DESCRIPTOR); });
            auto status = provide<StatusCapability>(context, dst_calib::STATUS_DESCRIPTOR,
                [](auto descriptor) { return StatusPublisher::create(descriptor, dst_calib::STATUS_DESCRIPTOR); });
            auto tf = rbt::tf::provide_dynamic_tf<ExtrinsicTfCapability>(context);
            auto host_clock = take_time(rbt::time::HostClock::create(), "create host clock");
            auto pending = std::make_shared<Pending>();
            const auto process = context.process_instance();
            const auto activation = context.activation_id();

            const auto input_task = context.spawn_critical("dst-input",
                [lidar = std::move(lidar), depth = std::move(depth), pending, camera, options = options_,
                 process, activation](std::stop_token stop) mutable {
                    try {
                        auto wait = take(rbt::WaitContext::create(process, activation), "create DST wait context");
                        const auto lidar_token = rbt::WaitToken::new_token(1);
                        const auto depth_token = rbt::WaitToken::new_token(2);
                        std::string error;
                        const auto result = wait.run(stop,
                            {lidar.event_source(lidar_token), depth.event_source(depth_token)},
                            [&](rbt::ReadyEvent event, std::size_t budget) -> rbt::WaitResult<rbt::WaitProgress> {
                                try {
                                    std::size_t processed = 0;
                                    while (processed < budget) {
                                        if (event.token_value() == lidar_token) {
                                            auto sample = take(lidar.receive(), "receive Mid360 cloud");
                                            if (!sample) break;
                                            LidarPacket packet;
                                            packet.timestamp_ns = sample->header().timestamp_ns;
                                            const auto& points = sample->payload();
                                            const auto stride = bounded_point_stride(points.size(), options.max_lidar_points);
                                            for (std::size_t i = 0; i < points.size(); i += stride) {
                                                const auto& point = points[i];
                                                const auto range2 = point.x * point.x + point.y * point.y + point.z * point.z;
                                                if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)
                                                    && range2 >= 0.01F && range2 <= 1600.0F)
                                                    packet.points.push_back({point.x, point.y, point.z});
                                            }
                                            std::lock_guard lock(pending->mutex);
                                            pending->lidars.push_back(std::move(packet));
                                            while (pending->lidars.size() > 32) pending->lidars.pop_front();
                                            pair_queues(*pending, options, camera);
                                        } else {
                                            auto sample = take(depth.receive(), "receive Orbbec depth");
                                            if (!sample) break;
                                            auto decoded = dst_calib_rbt::input::decode_stamped_orbbec(sample->payload());
                                            if (std::holds_alternative<std::string>(decoded))
                                                throw std::runtime_error(std::get<std::string>(decoded));
                                            auto frame = std::get<dst_calib_rbt::input::DecodedOrbbecSample>(std::move(decoded));
                                            if (frame.frame.kind != dst_calib_rbt::input::OrbbecKind::Depth
                                                || frame.frame.format != dst_calib_rbt::input::OrbbecFormat::Depth16)
                                                throw std::runtime_error("DST requires Orbbec Depth16 input");
                                            if (frame.frame.width != camera.width || frame.frame.height != camera.height)
                                                throw std::runtime_error("depth frame is not aligned to color CameraInfo");
                                            DepthPacket packet;
                                            packet.timestamp_ns = static_cast<std::uint64_t>(frame.stamp.timestamp.ns);
                                            packet.width = frame.frame.width;
                                            packet.height = frame.frame.height;
                                            packet.depth_m.resize(static_cast<std::size_t>(packet.width) * packet.height);
                                            for (std::uint32_t y = 0; y < packet.height; ++y) {
                                                const auto* row = frame.frame.payload.data() +
                                                    static_cast<std::size_t>(y) * frame.frame.stride;
                                                for (std::uint32_t x = 0; x < packet.width; ++x) {
                                                    const auto raw = dst_calib_rbt::input::load_u16_le(row + 2U * x);
                                                    const float depth_m = static_cast<float>(raw * frame.frame.unit_scale);
                                                    packet.depth_m[static_cast<std::size_t>(y) * packet.width + x] =
                                                        std::isfinite(depth_m) && depth_m >= options.depth_min_m
                                                            && depth_m <= options.depth_max_m ? depth_m : 0.0F;
                                                }
                                            }
                                            std::lock_guard lock(pending->mutex);
                                            pending->depths.push_back(std::move(packet));
                                            while (pending->depths.size() > 32) pending->depths.pop_front();
                                            pair_queues(*pending, options, camera);
                                        }
                                        ++processed;
                                    }
                                    return rbt::WaitResult<rbt::WaitProgress>(processed
                                        ? rbt::WaitProgress::samples(processed)
                                        : rbt::WaitProgress::empty());
                                } catch (const std::exception& exception) {
                                    error = exception.what();
                                    return rbt::WaitResult<rbt::WaitProgress>(rbt::WaitError::Handler);
                                }
                            });
                        if (!result && !stop.stop_requested()) return rbt::TaskOutcome::error(error);
                        return rbt::TaskOutcome::completed();
                    } catch (const std::exception& exception) {
                        return stop.stop_requested() ? rbt::TaskOutcome::completed()
                                                     : rbt::TaskOutcome::error(exception.what());
                    }
                });
            if (!input_task) return rbt::Result<void>::failure("failed to start DST input task");

            const auto worker_task = context.spawn_critical("dst-worker",
                [pending, options = options_, extrinsic = std::move(extrinsic), status = std::move(status),
                 tf = std::move(tf), host_clock = std::move(host_clock)](std::stop_token stop) mutable {
                    std::uint64_t output_sequence = 0;
                    std::uint64_t last_timestamp = 0;
                    auto publish_status = [&](std::uint32_t state, std::uint32_t backend,
                                              std::string_view message, double final_cd = 0.0) {
                        auto loan = status.loan();
                        if (!loan) throw std::runtime_error("DST status loan failed");
                        auto& value = loan.value().payload();
                        value = {};
                        value.sequence = ++output_sequence;
                        value.timestamp_ns = last_timestamp;
                        value.state = state;
                        value.backend = backend;
                        {
                            std::lock_guard lock(pending->mutex);
                            value.paired_frames = static_cast<std::uint32_t>(pending->paired_frames);
                            value.dropped_frames = static_cast<std::uint32_t>(pending->dropped_frames);
                            value.rejected_results = static_cast<std::uint32_t>(pending->rejected_results);
                            value.running_jobs = pending->running ? 1U : 0U;
                        }
                        value.final_cd = final_cd;
                        copy_text(value.message, sizeof(value.message), message);
                        if (!loan.value().send()) throw std::runtime_error("DST status send failed");
                    };
                    try {
                        publish_status(1U, 1U, "ready");
                        while (!stop.stop_requested()) {
                            Job job;
                            {
                                std::unique_lock lock(pending->mutex);
                                pending->changed.wait(lock, stop, [&] { return !pending->jobs.empty(); });
                                if (stop.stop_requested()) break;
                                job = std::move(pending->jobs.front());
                                pending->jobs.pop_front();
                                pending->running = true;
                                last_timestamp = job.snapshot.frames.back().timestamp_ns;
                            }
                            publish_status(2U, 1U, "worker running");
                            try {
                                const auto result_path = run_worker(options, job.snapshot, job.sequence);
                                const auto result = read_result(result_path);
                                check_transform(result.transform);
                                if (options.publish_results && !result.pipeline_approved)
                                    throw std::runtime_error("unapproved experimental HTP result: refusing extrinsic/TF publication");
                                if (!options.publish_results) {
                                    {
                                        std::lock_guard lock(pending->mutex);
                                        pending->running = false;
                                    }
                                    publish_status(3U, backend_id(result.backend),
                                                   "offline-only result saved; no extrinsic or TF published", result.final_cd);
                                    std::cout << "DST offline-only result=" << result_path << std::endl;
                                    continue;
                                }
                                if (!std::isfinite(result.final_cd)
                                    || (options.max_final_cd > 0.0 && result.final_cd > options.max_final_cd)) {
                                    {
                                        std::lock_guard lock(pending->mutex);
                                        ++pending->rejected_results;
                                        pending->running = false;
                                    }
                                    publish_status(5U, backend_id(result.backend),
                                                   "result rejected by final_cd gate", result.final_cd);
                                    continue;
                                }
                                const auto stamp = take_time(rbt::time::Timestamp::create(
                                    host_clock.monotonic_identity(), static_cast<std::int64_t>(last_timestamp)),
                                    "create DST result timestamp");
                                auto loan = extrinsic.loan();
                                if (!loan) throw std::runtime_error("DST extrinsic loan failed");
                                auto& value = loan.value().payload();
                                value = {};
                                value.sequence = ++output_sequence;
                                value.timestamp_ns = last_timestamp;
                                value.clock_kind = 1U;
                                copy_clock(value.clock_source_id, stamp.clock);
                                value.clock_generation = stamp.clock.generation();
                                copy_text(value.parent_frame, sizeof(value.parent_frame), options.parent_frame);
                                copy_text(value.child_frame, sizeof(value.child_frame), options.child_frame);
                                copy_text(value.mode, sizeof(value.mode), result.mode);
                                value.status = 1U;
                                value.backend = backend_id(result.backend);
                                value.iteration = static_cast<std::uint32_t>(job.sequence);
                                value.final_cd = result.final_cd;
                                for (std::size_t i = 0; i < 3; ++i) {
                                    value.rotation_deg[i] = result.rotation_deg[i];
                                    value.translation_m[i] = result.translation_m[i];
                                }
                                std::copy(result.transform.begin(), result.transform.end(), value.transform_row_major);
                                if (!loan.value().send()) throw std::runtime_error("DST extrinsic send failed");
                                const auto& t = result.transform;
                                const auto transform = rbt::tf::Transform3(
                                    rbt::tf::Vec3(t[3], t[7], t[11]), quaternion_from_rotation(result.transform));
                                tf.publish({rbt::tf::FrameId(options.parent_frame), rbt::tf::FrameId(options.child_frame),
                                            stamp, std::nullopt, job.sequence, transform});
                                {
                                    std::lock_guard lock(pending->mutex);
                                    pending->running = false;
                                }
                                publish_status(3U, backend_id(result.backend), "published", result.final_cd);
                                std::cout << "DST result job=" << job.sequence << " mode=" << result.mode
                                          << " backend=" << result.backend << " final_cd=" << result.final_cd << std::endl;
                            } catch (const std::exception& exception) {
                                {
                                    std::lock_guard lock(pending->mutex);
                                    pending->running = false;
                                }
                                publish_status(4U, 1U, exception.what());
                                std::cerr << "DST worker: " << exception.what() << std::endl;
                            }
                        }
                        return rbt::TaskOutcome::completed();
                    } catch (const std::exception& exception) {
                        return stop.stop_requested() ? rbt::TaskOutcome::completed()
                                                     : rbt::TaskOutcome::error(exception.what());
                    }
                });
            if (!worker_task) return rbt::Result<void>::failure("failed to start DST worker task");
            std::cout << "DST ready mode=" << options_.mode << " backend=" << options_.backend
                      << " window=" << options_.window_size << " stride=" << options_.window_stride << std::endl;
            return {};
        } catch (const std::exception& error) {
            return rbt::Result<void>::failure(error.what());
        }
    }

private:
    Options options_;
};

} // namespace

} // namespace dst_calib_rbt

int main(int argc, char** argv) {
    try {
        std::string config = "config/dst_calib_rbt/default.yaml";
        std::string component_id;
        std::string mode;
        std::string backend;
        std::string options_python;
        std::string output_directory;
        std::string init_yaml;
        bool check_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--help" || argument == "-h") {
                std::cout << "dst_calib_rbt_node --config FILE [--component-id ID]"
                          << " [--mode selfsup|fast|both] [--backend auto|qnn|cpu]"
                          << " [--python FILE] [--init-yaml FILE] [--out DIR] [--check-config]\n";
                return 0;
            }
            if (argument == "--check-config") { check_only = true; continue; }
            if (argument != "--config" && argument != "--component-id" && argument != "--mode"
                && argument != "--backend" && argument != "--python" && argument != "--out"
                && argument != "--init-yaml")
                throw std::invalid_argument("unknown option: " + std::string(argument));
            if (++i >= argc) throw std::invalid_argument("missing value for " + std::string(argument));
            if (argument == "--config") config = argv[i];
            else if (argument == "--component-id") component_id = argv[i];
            else if (argument == "--mode") mode = argv[i];
            else if (argument == "--backend") backend = argv[i];
            else if (argument == "--python") options_python = argv[i];
            else if (argument == "--out") output_directory = argv[i];
            else init_yaml = argv[i];
        }
        auto options = dst_calib_rbt::load_options(config, component_id);
        if (!mode.empty()) options.mode = mode;
        if (!backend.empty()) options.backend = backend;
        if (!options_python.empty()) options.python = options_python;
        if (!output_directory.empty()) options.output_directory = output_directory;
        if (!init_yaml.empty()) options.init_yaml = init_yaml;
        dst_calib_rbt::validate_options(options);
        if (check_only) {
            std::cout << "DST-RBT configuration OK mode=" << options.mode
                      << " backend=" << options.backend << std::endl;
            return 0;
        }
        rbt::RuntimeOptions runtime;
        if (const auto* address = std::getenv("RBT_COMPOSER_ADDR"); address && *address)
            runtime.composer_address = address;
        rbt::ComponentRuntime component(std::make_unique<dst_calib_rbt::Component>(std::move(options)), runtime);
        return component.run();
    } catch (const std::exception& error) {
        std::cerr << "dst_calib_rbt: " << error.what() << '\n';
        return 1;
    }
}
