#include "stc_extract_c.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
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

int extraction_error(const char *) {
    return -1;
}

} // namespace

int stc_extract(const u8 *stego,
        int stegolength,
        u8 *message,
        int messagelength,
        int constraint_height) {
    try {
        if (constraint_height < 4 || constraint_height > 20) {
            return extraction_error("STC constraint height must be in 4..20.");
        }
        if (stego == nullptr || message == nullptr) {
            return extraction_error("STC stego and message pointers must be non-null.");
        }
        if (stegolength <= 0 || messagelength <= 0) {
            return extraction_error("STC stego and message lengths must be positive.");
        }
        if (messagelength > stegolength) {
            return extraction_error("The message cannot be longer than the stego object.");
        }
        if (messagelength < constraint_height) {
            return extraction_error("STC message length must be at least the constraint height.");
        }
        for (int i = 0; i < stegolength; ++i) {
            if (stego[i] != 0 && stego[i] != 1) {
                return extraction_error("STC stego vector contains a non-binary value.");
            }
        }

        const stc_width_plan width_plan = stc_make_width_plan(
                stegolength, messagelength);
        const int shorter = width_plan.shorter;
        const int longer = width_plan.longer;
        malloc_ptr<u32> short_columns(getMatrix(shorter, constraint_height));
        malloc_ptr<u32> long_columns(getMatrix(longer, constraint_height));
        if (!short_columns || !long_columns) {
            return extraction_error("Could not construct STC extraction matrices.");
        }

        const size_t height_size = static_cast<size_t>(constraint_height);
        if (static_cast<size_t>(shorter)
                    > std::numeric_limits<size_t>::max() / height_size
                || static_cast<size_t>(longer)
                    > std::numeric_limits<size_t>::max() / height_size
                || static_cast<size_t>(messagelength)
                    > std::numeric_limits<size_t>::max()
                            - height_size) {
            return extraction_error("STC extraction workspace size overflow.");
        }
        std::vector<u8> short_binary(static_cast<size_t>(shorter)
                * height_size);
        std::vector<u8> long_binary(static_cast<size_t>(longer)
                * height_size);
        for (int k = 0; k < shorter; ++k) {
            for (int bit = 0; bit < constraint_height; ++bit) {
                short_binary[(size_t)k * (size_t)constraint_height
                        + (size_t)bit] =
                        (short_columns.get()[k] & (1u << bit)) ? 1u : 0u;
            }
        }
        for (int k = 0; k < longer; ++k) {
            for (int bit = 0; bit < constraint_height; ++bit) {
                long_binary[(size_t)k * (size_t)constraint_height
                        + (size_t)bit] =
                        (long_columns.get()[k] & (1u << bit)) ? 1u : 0u;
            }
        }

        /*
         * The convolution extends up to h-1 positions beyond the requested
         * message. Keep that tail in private workspace and copy only the
         * documented messagelength bytes to the caller.
         */
        std::vector<u8> syndrome(static_cast<size_t>(messagelength)
                + height_size, 0u);
        int active_height = constraint_height;
        int carrier_index = 0;
        for (int message_index = 0;
                message_index < messagelength;
                ++message_index) {
            const int choice = width_plan.matrix_choice[
                    static_cast<size_t>(message_index)];
            const std::vector<u8>& binary = choice == 0
                    ? short_binary : long_binary;
            const int width = width_plan.widths[
                    static_cast<size_t>(message_index)];
            for (int k = 0; k < width; ++k, ++carrier_index) {
                if (stego[carrier_index] == 0) continue;
                const size_t base = (size_t)k
                        * (size_t)constraint_height;
                for (int bit = 0; bit < active_height; ++bit) {
                    syndrome[(size_t)message_index + (size_t)bit] ^=
                            binary[base + (size_t)bit];
                }
            }
            if (messagelength - message_index <= constraint_height) {
                --active_height;
            }
        }
        if (carrier_index != stegolength) {
            return extraction_error("STC extraction did not consume every carrier.");
        }

        std::copy_n(syndrome.begin(), messagelength, message);
        return 0;
    } catch (const std::bad_alloc&) {
        return extraction_error("Not enough memory for STC extraction.");
    } catch (...) {
        return extraction_error("Unexpected STC extraction failure.");
    }
}
