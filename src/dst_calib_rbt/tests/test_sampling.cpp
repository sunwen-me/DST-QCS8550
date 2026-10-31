#include "dst_calib_rbt/sampling.hpp"
#include <iostream>
#include <limits>
#include <utility>

int main() {
    using dst_calib_rbt::bounded_point_stride;
    for (const auto [count, cap] : {
        std::pair<std::size_t, std::size_t>{0, 12000}, {1, 1}, {11999, 12000},
        {12000, 12000}, {12001, 12000}, {23999, 12000}, {24000, 12000},
        {121184, 12000}, {120018, 12000},
        {std::numeric_limits<std::size_t>::max(), 1},
        {std::numeric_limits<std::size_t>::max(), 12000},
        {std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()}}) {
        const auto stride = bounded_point_stride(count, cap);
        const auto selected = count ? 1 + (count - 1) / stride : 0;
        if (!stride || selected > cap) {
            std::cerr << "point cap invariant violated\n";
            return 1;
        }
    }
    try {
        (void)bounded_point_stride(1, 0);
        return 2;
    } catch (const std::invalid_argument&) {}
    return 0;
}
