#include "dst_calib_rbt/orbbec_input.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <vector>
using namespace dst_calib_rbt::input;
namespace {
void put32(std::vector<std::uint8_t>& b, std::size_t pos, std::uint32_t v) {
    for (std::size_t i=0;i<4;++i) b[pos+i]=static_cast<std::uint8_t>(v>>(i*8));
}
void put64(std::vector<std::uint8_t>& b, std::size_t pos, std::uint64_t v) {
    for (std::size_t i=0;i<8;++i) b[pos+i]=static_cast<std::uint8_t>(v>>(i*8));
}
std::vector<std::uint8_t> good() {
    std::vector<std::uint8_t> b(136,0);
    const std::array<std::uint8_t,8> magic{'R','B','T','O','R','B','0','1'};
    std::copy(magic.begin(),magic.end(),b.begin());
    put32(b,8,2);put32(b,12,4);put32(b,16,2);put32(b,20,2);put32(b,24,4);
    double scale=.001;std::uint64_t bits=0;std::memcpy(&bits,&scale,8);put64(b,40,bits);
    put64(b,48,8);b[64]='c';b[128]=0xe8;b[129]=3;
    return b;
}
}
int main() {
    auto b=good();auto result=decode_orbbec_frame(b);
    if (!std::holds_alternative<OrbbecFrame>(result))return 1;
    const auto& frame=std::get<OrbbecFrame>(result);
    if(frame.width!=2||frame.height!=2||frame.unit_scale!=.001||load_u16_le(frame.payload.data())!=1000)return 2;
    b=good();put32(b,24,2);
    const auto bad_stride=decode_orbbec_frame(b);
    if(!std::holds_alternative<OrbbecDecodeError>(bad_stride)||std::get<OrbbecDecodeError>(bad_stride)!=OrbbecDecodeError::InvalidStride)return 3;
    b=good();put32(b,20,3);
    const auto short_payload=decode_orbbec_frame(b);
    if(!std::holds_alternative<OrbbecDecodeError>(short_payload)||std::get<OrbbecDecodeError>(short_payload)!=OrbbecDecodeError::TruncatedPayload)return 4;
    b=good();b[0]=0;
    if(!std::holds_alternative<OrbbecDecodeError>(decode_orbbec_frame(b)))return 5;
    if(!std::holds_alternative<std::string>(decode_stamped_orbbec({})))return 6;
    std::cout<<"DST depth wire geometry and invalid payload tests passed\n";
    return 0;
}
