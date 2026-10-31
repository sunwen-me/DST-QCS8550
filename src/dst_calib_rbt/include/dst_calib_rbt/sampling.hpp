#pragma once
#include <cstddef>
#include <stdexcept>

namespace dst_calib_rbt {
// Ceil division without overflow: the old floor stride could exceed the cap.
inline std::size_t bounded_point_stride(std::size_t count, std::size_t cap) {
    if (cap == 0) throw std::invalid_argument("point cap must be positive");
    return count <= cap ? 1 : count / cap + (count % cap != 0);
}
}
