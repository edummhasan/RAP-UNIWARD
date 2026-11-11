#include "original_stc_adapter.h"
#include "original_stc/common.h"
#include "original_stc/stc_embed_c.h"
#include "original_stc/stc_extract_c.h"

#include <cmath>
#include <climits>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <cstdlib>

static void reference_fallback_matrix() {
    static const u32 expected[27] = {
        549, 747, 909, 585, 767, 905, 715, 645, 847,
        961, 657, 641, 973, 583, 749, 1021, 903, 793,
        691, 789, 869, 915, 981, 907, 765, 1003, 669
    };
    u32* actual = getMatrix(27, 10);
    if (actual == nullptr) throw std::runtime_error("fallback matrix allocation failed");
    for (int i = 0; i < 27; ++i) {
        if (actual[i] != expected[i]) {
            std::free(actual);
            throw std::runtime_error("fallback matrix differs from public original STC");
        }
    }
    std::free(actual);
}

static void round_trip(int cover_length, int message_length, int height, bool include_zero_cost = false) {
    std::vector<unsigned char> cover((size_t)cover_length);
    std::vector<double> costs((size_t)cover_length);
    std::vector<unsigned char> message((size_t)message_length);
    for (int i = 0; i < cover_length; ++i) {
        cover[(size_t)i] = (unsigned char)((i * 17 + 3) & 1);
        costs[(size_t)i] = (include_zero_cost && i % 97 == 0)
            ? 0.0
            : 0.25 + (double)((i * 29) % 101) / 50.0;
    }
    for (int i = 0; i < message_length; ++i) {
        message[(size_t)i] = (unsigned char)((i * 13 + 5) & 1);
    }
    std::vector<unsigned char> stego = original_stc_embed_adapter(cover, costs, message, height);
    std::vector<unsigned char> recovered = original_stc_extract_adapter(stego, message_length, height);
    if (recovered != message) throw std::runtime_error("native STC round trip mismatch");
}

static double changed_cost(const std::vector<unsigned char>& cover,
        const std::vector<unsigned char>& stego,
        const std::vector<double>& costs) {
    double sum = 0.0;
    for (size_t i = 0; i < cover.size(); ++i) {
        if (cover[i] != stego[i]) sum += costs[i];
    }
    return sum;
}

static std::string bit_string(const std::vector<unsigned char>& bits) {
    std::string out;
    for (unsigned char bit : bits) out.push_back(bit ? '1' : '0');
    return out;
}

/*
 * Exhaustively checks a small h=7 instance. Every one of the 2^10 possible
 * stego vectors is decoded; the test compares the trellis result with the
 * minimum binary64 cost among all vectors that carry the requested syndrome.
 */
static void exhaustive_binary64_optimality() {
    const int cover_length = 10;
    const int message_length = 7;
    const int height = 7;
    const std::vector<unsigned char> cover = {0, 1, 1, 0, 1, 0, 0, 1, 1, 0};
    const std::vector<unsigned char> message = {1, 0, 1, 1, 0, 0, 1};
    const std::vector<double> costs = {
        0.1250000000003, 0.1250000000002, 0.7500000000001,
        1.2500000000004, 0.5000000000007, 0.3750000000006,
        0.8750000000008, 0.6250000000009, 1.5000000000011,
        0.2500000000013
    };

    std::vector<unsigned char> stego((size_t)cover_length, 0);
    const double returned = stc_embed(cover.data(), cover_length,
            message.data(), message_length, costs.data(), stego.data(), height);
    std::vector<unsigned char> decoded((size_t)message_length + 16, 0);
    if (stc_extract(stego.data(), cover_length, decoded.data(),
            message_length, height) != 0
            || !std::equal(message.begin(), message.end(), decoded.begin())) {
        throw std::runtime_error("binary64 exhaustive case did not decode");
    }

    double optimum = std::numeric_limits<double>::infinity();
    std::vector<unsigned char> optimum_stego;
    std::vector<unsigned char> trial((size_t)cover_length, 0);
    for (unsigned int mask = 0; mask < (1u << cover_length); ++mask) {
        for (int i = 0; i < cover_length; ++i) {
            trial[(size_t)i] = (unsigned char)((mask >> i) & 1u);
        }
        if (stc_extract(trial.data(), cover_length, decoded.data(),
                message_length, height) != 0) {
            throw std::runtime_error("binary64 exhaustive extraction failed");
        }
        if (std::equal(message.begin(), message.end(), decoded.begin())) {
            const double trial_cost = changed_cost(cover, trial, costs);
            if (trial_cost < optimum) {
                optimum = trial_cost;
                optimum_stego = trial;
            }
        }
    }

    const double realized = changed_cost(cover, stego, costs);
    const double tolerance = 32.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, optimum);
    if (std::abs(returned - realized) > tolerance
            || std::abs(realized - optimum) > tolerance) {
        throw std::runtime_error("binary64 trellis is not exhaustively optimal: returned="
                + std::to_string(returned) + ", realized="
                + std::to_string(realized) + ", optimum="
                + std::to_string(optimum) + ", stego="
                + bit_string(stego) + ", optimum_stego="
                + bit_string(optimum_stego));
    }
}

static void exact_width_planner_boundaries() {
    for (int cover = 1; cover <= 160; ++cover) {
        for (int message = 1; message <= cover; ++message) {
            const stc_width_plan plan = stc_make_width_plan(
                    cover, message);
            if ((int)plan.widths.size() != message
                    || (int)plan.matrix_choice.size() != message) {
                throw std::runtime_error("STC width-plan vector size mismatch");
            }
            long long sum = 0;
            for (int j = 0; j < message; ++j) {
                const int width = plan.widths[(size_t)j];
                if (width != plan.shorter && width != plan.longer) {
                    throw std::runtime_error(
                            "STC width plan used a third width");
                }
                if (plan.shorter != plan.longer
                        && ((plan.matrix_choice[(size_t)j] == 1)
                        != (width == plan.longer))) {
                    throw std::runtime_error(
                            "STC matrix choice and width disagree");
                }
                sum += width;
            }
            if (sum != cover) {
                throw std::runtime_error(
                        "STC exact width plan does not sum to cover length");
            }
        }
    }

    if (stc_shorter_width(INT_MAX, INT_MAX) != 1
            || stc_longer_width(INT_MAX, INT_MAX) != 1
            || !stc_should_use_longer(
                    (long long)INT_MAX - 1,
                    INT_MAX - 1, INT_MAX, INT_MAX)) {
        throw std::runtime_error(
                "STC width predicate failed at the signed-int boundary");
    }
    if (stc_shorter_width(INT_MAX, 1) != INT_MAX
            || stc_longer_width(INT_MAX, 1) != INT_MAX
            || !stc_should_use_longer(0, 0, INT_MAX, 1)) {
        throw std::runtime_error(
                "STC single-syndrome width failed at INT_MAX");
    }
}

static void exhaustive_small_profiles() {
    std::mt19937_64 generator(0x4558414354535443ULL);
    for (int height = 4; height <= 7; ++height) {
        for (int instance = 0; instance < 3; ++instance) {
            const int message_length = height;
            const int cover_length = height + 3 + instance;
            std::vector<unsigned char> cover((size_t)cover_length);
            std::vector<unsigned char> message((size_t)message_length);
            std::vector<double> costs((size_t)cover_length);
            for (int i = 0; i < cover_length; ++i) {
                cover[(size_t)i] = (unsigned char)(generator() & 1u);
                costs[(size_t)i] = 0.01
                        + (double)(generator() % 100000u) / 10007.0
                        + (double)(i + 1) * 1e-13;
            }
            for (int i = 0; i < message_length; ++i) {
                message[(size_t)i] =
                        (unsigned char)(generator() & 1u);
            }

            const std::vector<unsigned char> stego =
                    original_stc_embed_adapter(
                            cover, costs, message, height);
            const double realized = changed_cost(cover, stego, costs);
            double optimum = std::numeric_limits<double>::infinity();
            std::vector<unsigned char> trial((size_t)cover_length);
            std::vector<unsigned char> decoded((size_t)message_length);
            const unsigned int possibilities = 1u << cover_length;
            for (unsigned int mask = 0; mask < possibilities; ++mask) {
                for (int i = 0; i < cover_length; ++i) {
                    trial[(size_t)i] =
                            (unsigned char)((mask >> i) & 1u);
                }
                if (stc_extract(
                        trial.data(), cover_length, decoded.data(),
                        message_length, height) != 0) {
                    throw std::runtime_error(
                            "small exhaustive extraction failed");
                }
                if (decoded == message) {
                    optimum = std::min(
                            optimum, changed_cost(cover, trial, costs));
                }
            }
            const double tolerance = 64.0
                    * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, optimum);
            if (std::abs(realized - optimum) > tolerance) {
                throw std::runtime_error(
                        "small-profile STC result is not globally optimal");
            }
        }
    }
}

/* A former binary32 state array loses these sub-ulp-at-float perturbations. */
static void binary64_precision_retention() {
    const int cover_length = 2048;
    const int message_length = 512;
    const int height = 10;
    std::vector<unsigned char> cover((size_t)cover_length);
    std::vector<unsigned char> message((size_t)message_length);
    std::vector<double> costs((size_t)cover_length);
    for (int i = 0; i < cover_length; ++i) {
        cover[(size_t)i] = (unsigned char)((i * 17 + 3) & 1);
        costs[(size_t)i] = 0.75 + (double)((i * 29) % 101) / 50.0
                + (double)(i + 1) * 1e-12;
    }
    for (int i = 0; i < message_length; ++i) {
        message[(size_t)i] = (unsigned char)((i * 13 + 5) & 1);
    }

    std::vector<unsigned char> stego((size_t)cover_length, 0);
    const double returned = stc_embed(cover.data(), cover_length,
            message.data(), message_length, costs.data(), stego.data(), height);
    const double cost_only = stc_embed(cover.data(), cover_length,
            message.data(), message_length, costs.data(), nullptr, height);
    const double realized = changed_cost(cover, stego, costs);
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, realized);
    if (std::abs(returned - realized) > tolerance
            || std::abs(cost_only - realized) > tolerance) {
        throw std::runtime_error("STC price path narrowed below binary64");
    }
}

static void deterministic_randomized_round_trips() {
    std::mt19937_64 generator(0x4150485354433634ULL);
    const int heights[] = {4, 7, 10, 12};
    for (int height : heights) {
        for (int trial = 0; trial < 12; ++trial) {
            const int message_length = height
                    + (int)(generator() % 73u);
            const int cover_length = message_length
                    + (int)(generator() % (unsigned)(4 * message_length));
            std::vector<unsigned char> cover((size_t)cover_length);
            std::vector<unsigned char> message((size_t)message_length);
            std::vector<double> costs((size_t)cover_length);
            for (int i = 0; i < cover_length; ++i) {
                cover[(size_t)i] = (unsigned char)(generator() & 1u);
                costs[(size_t)i] = (generator() % 29u == 0u)
                        ? 0.0
                        : 0.001 + (double)(generator() % 1000000u) / 100003.0;
            }
            for (int i = 0; i < message_length; ++i) {
                message[(size_t)i] = (unsigned char)(generator() & 1u);
            }
            const std::vector<unsigned char> stego =
                    original_stc_embed_adapter(
                            cover, costs, message, height);
            const std::vector<unsigned char> recovered =
                    original_stc_extract_adapter(
                            stego, message_length, height);
            if (recovered != message) {
                throw std::runtime_error(
                        "deterministic randomized STC round trip failed");
            }
        }
    }
}

static void rejects_invalid_costs() {
    const std::vector<unsigned char> cover(16, 0u);
    const std::vector<unsigned char> message = {1u, 0u, 1u, 0u};
    const double invalid_values[] = {
        -0.25,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        1e13 + 1.0
    };
    for (double invalid : invalid_values) {
        std::vector<double> costs(cover.size(), 1.0);
        costs[3] = invalid;
        bool rejected = false;
        try {
            (void)original_stc_embed_adapter(
                    cover, costs, message, 4);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("invalid STC cost was silently accepted");
        }
    }

    std::vector<double> wet_costs(cover.size(), 1.0);
    wet_costs[3] = 1e13;
    const std::vector<unsigned char> stego = original_stc_embed_adapter(
            cover, wet_costs, message, 4);
    if (original_stc_extract_adapter(
            stego, static_cast<int>(message.size()), 4) != message) {
        throw std::runtime_error("the exact wet sentinel was not accepted");
    }
}

static void rejects_syndrome_shorter_than_height() {
    const std::vector<unsigned char> cover(8, 0u);
    const std::vector<double> costs(8, 1.0);
    const std::vector<unsigned char> message = {1u, 0u};
    bool embed_rejected = false;
    try {
        (void)original_stc_embed_adapter(cover, costs, message, 4);
    } catch (const std::runtime_error&) {
        embed_rejected = true;
    }
    if (!embed_rejected) {
        throw std::runtime_error("short STC syndrome was accepted by embedding");
    }

    std::vector<unsigned char> extracted(message.size(), 0u);
    if (stc_extract(cover.data(), (int)cover.size(), extracted.data(),
            (int)message.size(), 4) == 0) {
        throw std::runtime_error("short STC syndrome was accepted by extraction");
    }
}

static void extraction_respects_output_bounds() {
    const int cover_length = 64;
    const int message_length = 16;
    const int height = 10;
    std::vector<unsigned char> cover((size_t)cover_length);
    std::vector<double> costs((size_t)cover_length);
    std::vector<unsigned char> message((size_t)message_length);
    for (int i = 0; i < cover_length; ++i) {
        cover[(size_t)i] = (unsigned char)((i * 5 + 1) & 1);
        costs[(size_t)i] = 0.5 + (double)(i % 11) / 13.0;
    }
    for (int i = 0; i < message_length; ++i) {
        message[(size_t)i] = (unsigned char)((i * 7 + 1) & 1);
    }
    const std::vector<unsigned char> stego = original_stc_embed_adapter(
            cover, costs, message, height);

    std::vector<unsigned char> guarded((size_t)message_length + 2u, 0xa5u);
    if (stc_extract(stego.data(), cover_length, guarded.data() + 1,
            message_length, height) != 0) {
        throw std::runtime_error("bounded extraction failed");
    }
    if (guarded.front() != 0xa5u || guarded.back() != 0xa5u) {
        throw std::runtime_error("STC extraction wrote beyond message buffer");
    }
    if (!std::equal(message.begin(), message.end(), guarded.begin() + 1)) {
        throw std::runtime_error("bounded extraction recovered wrong syndrome");
    }
}

int main() {
    try {
        reference_fallback_matrix();
        round_trip(2048, 512, 10);  // precomputed public STC submatrices
        round_trip(997, 37, 10);    // deterministic public fallback submatrices
        round_trip(2048, 512, 10, true); // zero is a legal non-negative distortion cost
        round_trip(2048, 512, 12);  // RAP-UNIWARD bounded research-height profile
        round_trip(997, 37, 12);    // height-12 deterministic fallback path
        exhaustive_binary64_optimality();
        exact_width_planner_boundaries();
        exhaustive_small_profiles();
        binary64_precision_retention();
        deterministic_randomized_round_trips();
        rejects_invalid_costs();
        rejects_syndrome_shorter_than_height();
        extraction_respects_output_bounds();
        std::cout << "NativeStcSelfTest: ALL TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "NativeStcSelfTest: FAILED: " << e.what() << '\n';
        return 1;
    }
}
