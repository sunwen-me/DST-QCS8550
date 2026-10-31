#include "dst_calib_rbt/orbbec_input.hpp"
#include <rbt/time_stream.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
namespace dst_calib_rbt::input {
namespace {

constexpr std::array<char, 8> kMagic {'R', 'B', 'T', 'O', 'R', 'B', '0', '1'};

bool valid_kind(std::uint32_t value) noexcept {
    return value >= static_cast<std::uint32_t>(OrbbecKind::Color)
        && value <= static_cast<std::uint32_t>(OrbbecKind::Gyro);
}

bool valid_format(std::uint32_t value) noexcept {
    return value >= static_cast<std::uint32_t>(OrbbecFormat::Rgb8)
        && value <= static_cast<std::uint32_t>(OrbbecFormat::Gyro32f);
}

bool all_zero(std::span<const std::uint8_t> bytes) noexcept {
    return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t value) { return value == 0U; });
}

} // namespace

std::uint16_t load_u16_le(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint16_t>(bytes[0])
        | static_cast<std::uint16_t>(bytes[1] << 8U);
}

std::uint32_t load_u32_le(const std::uint8_t* bytes) noexcept {
    std::uint32_t value = 0U;
    for (std::size_t i = 0U; i < 4U; ++i) value |= static_cast<std::uint32_t>(bytes[i]) << (8U * i);
    return value;
}

std::uint64_t load_u64_le(const std::uint8_t* bytes) noexcept {
    std::uint64_t value = 0U;
    for (std::size_t i = 0U; i < 8U; ++i) value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
    return value;
}

double load_f64_le(const std::uint8_t* bytes) noexcept {
    const auto bits = load_u64_le(bytes);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

const char* to_string(OrbbecDecodeError error) noexcept {
    switch (error) {
    case OrbbecDecodeError::TooSmall: return "TooSmall";
    case OrbbecDecodeError::InvalidMagic: return "InvalidMagic";
    case OrbbecDecodeError::InvalidKind: return "InvalidKind";
    case OrbbecDecodeError::InvalidFormat: return "InvalidFormat";
    case OrbbecDecodeError::InvalidDimensions: return "InvalidDimensions";
    case OrbbecDecodeError::InvalidStride: return "InvalidStride";
    case OrbbecDecodeError::InvalidScale: return "InvalidScale";
    case OrbbecDecodeError::InvalidFrameId: return "InvalidFrameId";
    case OrbbecDecodeError::PayloadTooLarge: return "PayloadTooLarge";
    case OrbbecDecodeError::TruncatedPayload: return "TruncatedPayload";
    }
    return "Unknown";
}

OrbbecDecodeResult decode_orbbec_frame(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < kOrbbecHeaderBytes) return OrbbecDecodeError::TooSmall;
    if (!std::equal(kMagic.begin(), kMagic.end(), reinterpret_cast<const char*>(bytes.data())))
        return OrbbecDecodeError::InvalidMagic;
    const auto kind = load_u32_le(bytes.data() + 8U);
    const auto format = load_u32_le(bytes.data() + 12U);
    if (!valid_kind(kind)) return OrbbecDecodeError::InvalidKind;
    if (!valid_format(format)) return OrbbecDecodeError::InvalidFormat;
    const auto width = load_u32_le(bytes.data() + 16U);
    const auto height = load_u32_le(bytes.data() + 20U);
    const auto stride = load_u32_le(bytes.data() + 24U);
    if (width == 0U || height == 0U) return OrbbecDecodeError::InvalidDimensions;
    if (stride == 0U || stride < width) return OrbbecDecodeError::InvalidStride;
    const auto unit_scale = load_f64_le(bytes.data() + 40U);
    if (!std::isfinite(unit_scale) || unit_scale <= 0.0) return OrbbecDecodeError::InvalidScale;
    const auto payload_size = load_u64_le(bytes.data() + 48U);
    if (payload_size > kMaxOrbbecPayloadBytes) return OrbbecDecodeError::PayloadTooLarge;
    if (payload_size != bytes.size() - kOrbbecHeaderBytes)
        return OrbbecDecodeError::TruncatedPayload;
    if (!all_zero(bytes.subspan(28U, 4U)) || !all_zero(bytes.subspan(56U, 8U)))
        return OrbbecDecodeError::InvalidDimensions;
    const auto frame_id_bytes = bytes.subspan(64U, kOrbbecFrameIdBytes);
    const auto nul = std::find(frame_id_bytes.begin(), frame_id_bytes.end(), std::uint8_t {0U});
    if (nul == frame_id_bytes.begin()) return OrbbecDecodeError::InvalidFrameId;
    if (nul != frame_id_bytes.end()) {
        for (auto it = nul; it != frame_id_bytes.end(); ++it) {
            if (*it != 0U) return OrbbecDecodeError::InvalidFrameId;
        }
    }
    // Bound Depth16 row access before the calibration node reads pixels.
    if (static_cast<OrbbecFormat>(format) == OrbbecFormat::Depth16) {
        if (static_cast<std::uint64_t>(stride) < static_cast<std::uint64_t>(width) * 2)
            return OrbbecDecodeError::InvalidStride;
        if (static_cast<std::uint64_t>(stride) * height != payload_size)
            return OrbbecDecodeError::TruncatedPayload;
    }
    OrbbecFrame result;
    result.kind = static_cast<OrbbecKind>(kind);
    result.format = static_cast<OrbbecFormat>(format);
    result.width = width;
    result.height = height;
    result.stride = stride;
    result.device_frame_index = load_u64_le(bytes.data() + 32U);
    result.unit_scale = unit_scale;
    result.frame_id.assign(reinterpret_cast<const char*>(frame_id_bytes.data()),
                           static_cast<std::size_t>((nul == frame_id_bytes.end())
                               ? frame_id_bytes.size() : nul - frame_id_bytes.begin()));
    result.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(kOrbbecHeaderBytes), bytes.end());
    return result;
}

std::variant<DecodedOrbbecSample, std::string> decode_stamped_orbbec(
    std::span<const std::uint8_t> transport_payload) {
    const auto decoded = rbt::time::decode_stamped_bytes(transport_payload);
    if (!std::holds_alternative<rbt::time::DecodedStampedBytes>(decoded)) {
        return std::string("StampedBytes decode failed: ")
            + rbt::time::to_string(std::get<rbt::time::PrefixCodecError>(decoded));
    }
    const auto& stamped = std::get<rbt::time::DecodedStampedBytes>(decoded);
    // Hydra and LIO use the host monotonic clock. Prefer a capture timestamp
    // when the Orbbec provider already mapped it to that clock; otherwise use
    // the stamped host receive time as the explicit ingress timestamp. A raw
    // device timestamp is never compared directly with the host timeline.
    const auto* timestamp = stamped.timing.capture.has_value()
        && stamped.timing.capture->clock.kind() == rbt::time::ClockKind::HostMonotonic
        ? &*stamped.timing.capture : &stamped.timing.received;
    if (timestamp->clock.kind() != rbt::time::ClockKind::HostMonotonic || timestamp->ns <= 0) {
        return std::string("Orbbec sample requires a positive HostMonotonic timestamp");
    }
    const auto frame = decode_orbbec_frame(stamped.bytes);
    if (!std::holds_alternative<OrbbecFrame>(frame)) {
        return std::string("Orbbec frame decode failed: ")
            + to_string(std::get<OrbbecDecodeError>(frame));
    }
    return DecodedOrbbecSample {CaptureStamp {*timestamp}, std::get<OrbbecFrame>(frame)};
}

} // namespace dst_calib_rbt::input
