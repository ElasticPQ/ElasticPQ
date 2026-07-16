#include "epq/variable_bit_packing.h"

#include <array>
#include <cstdint>
#include <iostream>

int main() {
    std::array<uint8_t, 4> code{};
    constexpr std::array<int, 3> widths = {7, 15, 10};
    constexpr std::array<uint16_t, 3> values = {0x55, 0x6abc, 0x2d5};

    int offset = 0;
    for (size_t i = 0; i < widths.size(); ++i) {
        epq::detail::pack_variable_bits(
                code.data(), offset, widths[i], values[i]);
        offset += widths[i];
    }

    offset = 0;
    for (size_t i = 0; i < widths.size(); ++i) {
        const uint16_t decoded = epq::detail::unpack_variable_bits(
                code.data(), offset, widths[i]);
        if (decoded != values[i]) {
            std::cerr << "variable-width round trip failed at field " << i
                      << ": expected=" << values[i]
                      << " actual=" << decoded << '\n';
            return 1;
        }
        offset += widths[i];
    }

    std::cout << "VAQ variable-width bit packing smoke passed\n";
    return 0;
}
