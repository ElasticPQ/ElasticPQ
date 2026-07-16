#pragma once

#include <cstdint>

namespace epq::detail {

inline uint16_t unpack_variable_bits(
        const uint8_t* code,
        int bit_offset,
        int width) {
    const int byte_offset = bit_offset / 8;
    const int shift = bit_offset % 8;
    uint32_t word = code[byte_offset];
    if (shift + width > 8) {
        word |= static_cast<uint32_t>(code[byte_offset + 1]) << 8;
    }
    if (shift + width > 16) {
        word |= static_cast<uint32_t>(code[byte_offset + 2]) << 16;
    }
    return static_cast<uint16_t>(
            (word >> shift) & ((uint32_t{1} << width) - 1));
}

inline void pack_variable_bits(
        uint8_t* code,
        int bit_offset,
        int width,
        uint16_t value) {
    const int byte_offset = bit_offset / 8;
    const int shift = bit_offset % 8;
    const uint32_t word = static_cast<uint32_t>(value) << shift;
    code[byte_offset] |= static_cast<uint8_t>(word & 0xffu);
    if (shift + width > 8) {
        code[byte_offset + 1] |= static_cast<uint8_t>((word >> 8) & 0xffu);
    }
    if (shift + width > 16) {
        code[byte_offset + 2] |= static_cast<uint8_t>((word >> 16) & 0xffu);
    }
}

}  // namespace epq::detail
