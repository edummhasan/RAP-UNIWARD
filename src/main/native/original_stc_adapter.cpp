#include "original_stc_adapter.h"
#include "original_stc/stc_embed_c.h"
#include "original_stc/stc_extract_c.h"
#include <stdexcept>
#include <vector>
#include <cmath>
#include <climits>

static void validate_bits(const std::vector<unsigned char>& bits, const char* name) {
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] != 0 && bits[i] != 1) throw std::runtime_error(std::string(name) + " contains non-binary values");
    }
}

std::vector<unsigned char> original_stc_embed_adapter(
    const std::vector<unsigned char>& cover_bits,
    const std::vector<double>& costs,
    const std::vector<unsigned char>& message_bits,
    int constraint_height
) {
    if (constraint_height < 4 || constraint_height > 20) {
        throw std::runtime_error("constraint_height must be in 4..20");
    }
    if (cover_bits.empty()) throw std::runtime_error("cover_bits is empty");
    if (message_bits.empty()) throw std::runtime_error("message_bits is empty");
    if (message_bits.size() < (size_t)constraint_height) {
        throw std::runtime_error(
                "message_bits length must be at least constraint_height");
    }
    if (cover_bits.size() != costs.size()) throw std::runtime_error("cover_bits and costs length mismatch");
    if (message_bits.size() > cover_bits.size()) throw std::runtime_error("message longer than cover bits");
    if (cover_bits.size() > (size_t)INT_MAX
            || message_bits.size() > (size_t)INT_MAX) {
        throw std::runtime_error("STC vector length exceeds native integer range");
    }
    validate_bits(cover_bits, "cover_bits");
    validate_bits(message_bits, "message_bits");

    constexpr double maximum_cost = 1e13;
    std::vector<double> safe_costs(costs.size());
    for (size_t i = 0; i < costs.size(); ++i) {
        const double value = costs[i];
        if (!std::isfinite(value) || value < 0.0
                || value > maximum_cost) {
            throw std::runtime_error(
                    "STC cost is outside the permitted [0, 1e13] range");
        }
        safe_costs[i] = value;
    }

    std::vector<unsigned char> stego(cover_bits.size(), 0);
    double distortion = stc_embed(
        cover_bits.data(), (int)cover_bits.size(),
        message_bits.data(), (int)message_bits.size(),
        safe_costs.data(), stego.data(), constraint_height
    );
    if (distortion < 0) throw std::runtime_error("stc_embed failed");
    return stego;
}

std::vector<unsigned char> original_stc_extract_adapter(
    const std::vector<unsigned char>& stego_bits,
    int message_bit_length,
    int constraint_height
) {
    if (constraint_height < 4 || constraint_height > 20) {
        throw std::runtime_error("constraint_height must be in 4..20");
    }
    if (stego_bits.empty()) throw std::runtime_error("stego_bits is empty");
    if (message_bit_length <= 0) throw std::runtime_error("message_bit_length must be positive");
    if (message_bit_length < constraint_height) {
        throw std::runtime_error(
                "message_bit_length must be at least constraint_height");
    }
    if ((size_t)message_bit_length > stego_bits.size()) throw std::runtime_error("requested message length exceeds stego bits");
    if (stego_bits.size() > (size_t)INT_MAX) {
        throw std::runtime_error("STC vector length exceeds native integer range");
    }
    validate_bits(stego_bits, "stego_bits");
    std::vector<unsigned char> message((size_t)message_bit_length, 0);
    int rc = stc_extract(stego_bits.data(), (int)stego_bits.size(), message.data(), message_bit_length, constraint_height);
    if (rc != 0) throw std::runtime_error("stc_extract failed");
    message.resize((size_t)message_bit_length);
    return message;
}
