#ifndef ORIGINAL_STC_ADAPTER_H
#define ORIGINAL_STC_ADAPTER_H
#include <vector>
std::vector<unsigned char> original_stc_embed_adapter(
    const std::vector<unsigned char>& cover_bits,
    const std::vector<double>& costs,
    const std::vector<unsigned char>& message_bits,
    int constraint_height
);
std::vector<unsigned char> original_stc_extract_adapter(
    const std::vector<unsigned char>& stego_bits,
    int message_bit_length,
    int constraint_height
);
#endif
