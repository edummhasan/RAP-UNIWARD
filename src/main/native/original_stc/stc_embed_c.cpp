#include "stc_embed_c.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <vector>

namespace {

struct FreeDeleter {
    template<typename T>
    void operator()(T *pointer) const noexcept {
        std::free(pointer);
    }
};

template<typename T>
using malloc_ptr = std::unique_ptr<T, FreeDeleter>;

[[noreturn]] void invalid_argument(const char *message) {
    throw stc_exception(message, 1);
}

} // namespace

/*
 * Binary syndrome-trellis encoder with IEEE-754 binary64 path metrics.
 *
 * The parity-check column construction and backtracking layout are retained
 * from the bundled STC snapshot. The active price recurrence is deliberately
 * scalar and binary64: no x86 intrinsic and no float cast is used. This makes
 * the result easier to audit and prevents cost distinctions computed in Java
 * from being discarded at the JNI boundary.
 */
double stc_embed(const u8 *cover,
        int coverlength,
        const u8 *message,
        int messagelength,
        const double *costs,
        u8 *stego,
        int constraint_height) {
    if (constraint_height < 4 || constraint_height > 20) {
        invalid_argument("STC constraint height must be in 4..20.");
    }
    if (cover == nullptr || message == nullptr || costs == nullptr) {
        invalid_argument("STC cover, message, and cost pointers must be non-null.");
    }
    if (coverlength <= 0 || messagelength <= 0) {
        invalid_argument("STC cover and message lengths must be positive.");
    }
    if (messagelength > coverlength) {
        throw stc_exception("The message cannot be longer than the cover object.", 3);
    }
    if (messagelength < constraint_height) {
        invalid_argument("STC message length must be at least the constraint height.");
    }

    constexpr double maximum_cost = 1e13;
    for (int i = 0; i < coverlength; ++i) {
        if ((cover[i] != 0 && cover[i] != 1)
                || !std::isfinite(costs[i]) || costs[i] < 0.0
                || costs[i] > maximum_cost) {
            invalid_argument("STC cover bits must be binary and costs must lie in [0, 1e13].");
        }
    }
    for (int i = 0; i < messagelength; ++i) {
        if (message[i] != 0 && message[i] != 1) {
            invalid_argument("STC message must contain only binary values.");
        }
    }

    const u32 state_count = 1u << constraint_height;
    u32 column_mask = state_count - 1u;
    const size_t path_words_per_carrier =
            ((size_t)state_count + 31u) / 32u;
    if (path_words_per_carrier != 0
            && static_cast<size_t>(coverlength)
                    > std::numeric_limits<size_t>::max()
                            / path_words_per_carrier) {
        throw stc_exception("STC backtracking-path size overflow.", 2);
    }
    const size_t path_word_count = static_cast<size_t>(coverlength)
            * path_words_per_carrier;
    if (path_word_count > std::numeric_limits<size_t>::max()
            / sizeof(u32)) {
        throw stc_exception("STC backtracking-path byte-size overflow.", 2);
    }
    const size_t path_byte_count = path_word_count * sizeof(u32);

    std::vector<u32> path;
    if (stego != nullptr) {
        try {
            path.assign(path_word_count, 0u);
        } catch (const std::bad_alloc&) {
            std::ostringstream detail;
            detail << "Not enough memory for the "
                    << path_byte_count
                    << "-byte STC backtracking path.";
            throw stc_exception(detail.str(), 2);
        }
    }

    const stc_width_plan width_plan = stc_make_width_plan(
            coverlength, messagelength);
    const int shorter = width_plan.shorter;
    const int longer = width_plan.longer;
    malloc_ptr<u32> short_columns(getMatrix(shorter, constraint_height));
    if (!short_columns) {
        throw stc_exception("Could not construct the shorter STC submatrix.", 2);
    }
    malloc_ptr<u32> long_columns(getMatrix(longer, constraint_height));
    if (!long_columns) {
        throw stc_exception("Could not construct the longer STC submatrix.", 2);
    }

    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> prices((size_t)state_count, infinity);
    std::vector<double> next_prices((size_t)state_count, infinity);
    prices[0] = 0.0;

    int carrier_index = 0;
    for (int message_index = 0;
            message_index < messagelength;
            ++message_index) {
        const int choice = width_plan.matrix_choice[
                static_cast<size_t>(message_index)];
        const u32 *columns = choice == 0
                ? short_columns.get() : long_columns.get();
        const int width = width_plan.widths[
                static_cast<size_t>(message_index)];

        for (int k = 0; k < width; ++k, ++carrier_index) {
            const u32 column = columns[k] & column_mask;
            const double output_zero_cost = cover[carrier_index] == 0
                    ? 0.0 : costs[carrier_index];
            const double output_one_cost = cover[carrier_index] == 1
                    ? 0.0 : costs[carrier_index];
            const size_t path_base = (size_t)carrier_index
                    * path_words_per_carrier;

            for (u32 state = 0; state < state_count; ++state) {
                const double candidate_zero = prices[state]
                        + output_zero_cost;
                const double candidate_one = prices[state ^ column]
                        + output_one_cost;
                const bool choose_one = candidate_one <= candidate_zero;
                next_prices[state] = choose_one
                        ? candidate_one : candidate_zero;
                if (stego != nullptr && choose_one) {
                    path[path_base + (state >> 5)] |=
                            1u << (state & 31u);
                }
            }
            prices.swap(next_prices);
        }

        const u32 constrained_bit = message[message_index];
        for (u32 state = 0; state < state_count / 2u; ++state) {
            prices[state] = prices[(state << 1u) | constrained_bit];
        }
        std::fill(prices.begin() + state_count / 2u,
                prices.end(), infinity);

        if (messagelength - message_index <= constraint_height) {
            column_mask >>= 1u;
        }
    }

    const double minimum_cost = prices[0];
    if (!std::isfinite(minimum_cost)) {
        throw stc_exception("No finite STC solution exists.", 4);
    }

    if (stego != nullptr) {
        int reverse_carrier = coverlength - 1;
        u32 state = 0u;
        u32 reverse_mask = 0u;

        for (int message_index = messagelength - 1;
                message_index >= 0;
                --message_index) {
            const int choice = width_plan.matrix_choice[
                    static_cast<size_t>(message_index)];
            const u32 *columns = choice == 0
                    ? short_columns.get() : long_columns.get();
            const int width = width_plan.widths[
                    static_cast<size_t>(message_index)];

            for (int k = width - 1; k >= 0; --k, --reverse_carrier) {
                if (k == width - 1) {
                    state = (state << 1u) | message[message_index];
                    if (messagelength - message_index <= constraint_height) {
                        reverse_mask = (reverse_mask << 1u) | 1u;
                    }
                }

                const size_t path_base = (size_t)reverse_carrier
                        * path_words_per_carrier;
                const bool chose_one = (path[path_base + (state >> 5)]
                        & (1u << (state & 31u))) != 0u;
                stego[reverse_carrier] = chose_one ? 1u : 0u;
                if (chose_one) state ^= columns[k] & reverse_mask;
            }
        }
        if (reverse_carrier != -1) {
            throw stc_exception("STC backtracking did not consume every carrier.", 1);
        }
    }

    return minimum_cost;
}
