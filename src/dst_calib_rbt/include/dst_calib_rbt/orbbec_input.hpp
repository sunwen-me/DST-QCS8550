#pragma once
// RBTORB01 wire decoder, extracted from the DST input boundary in the MID workspace.
// Only the depth/timestamp contract is included; no Hydra or camera SDK dependency.
#include <rbt/time.hpp>
#include <cstdint>
#include <cstddef>
#include <span>
#include <string>
#include <variant>
#include <vector>
namespace dst_calib_rbt::input {
inline constexpr std::size_t kOrbbecHeaderBytes = 128;
inline constexpr std::size_t kOrbbecFrameIdBytes = 64;
inline constexpr std::size_t kMaxOrbbecPayloadBytes = 32 * 1024 * 1024;
enum class OrbbecKind : std::uint32_t {Color=1, Depth=2, LeftIr=3, RightIr=4, Points=5, Accel=6, Gyro=7};
enum class OrbbecFormat : std::uint32_t {Rgb8=1, Mono8=2, Mono16=3, Depth16=4, Xyz32f=5, Accel32f=6, Gyro32f=7};
struct OrbbecFrame {
    OrbbecKind kind{}; OrbbecFormat format{};
    std::uint32_t width=0, height=0, stride=0;
    std::uint64_t device_frame_index=0;
    double unit_scale=1.0;
    std::string frame_id;
    std::vector<std::uint8_t> payload;
};
enum class OrbbecDecodeError : std::uint8_t {
    TooSmall, InvalidMagic, InvalidKind, InvalidFormat, InvalidDimensions,
    InvalidStride, InvalidScale, InvalidFrameId, PayloadTooLarge, TruncatedPayload
};
struct CaptureStamp {rbt::time::Timestamp timestamp;};
struct DecodedOrbbecSample {CaptureStamp stamp; OrbbecFrame frame;};
using OrbbecDecodeResult = std::variant<OrbbecFrame, OrbbecDecodeError>;
const char* to_string(OrbbecDecodeError error) noexcept;
std::uint16_t load_u16_le(const std::uint8_t* bytes) noexcept;
std::uint32_t load_u32_le(const std::uint8_t* bytes) noexcept;
std::uint64_t load_u64_le(const std::uint8_t* bytes) noexcept;
double load_f64_le(const std::uint8_t* bytes) noexcept;
OrbbecDecodeResult decode_orbbec_frame(std::span<const std::uint8_t> bytes);
std::variant<DecodedOrbbecSample,std::string> decode_stamped_orbbec(std::span<const std::uint8_t> bytes);
} // namespace dst_calib_rbt::input
