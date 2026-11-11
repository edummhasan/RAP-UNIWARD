#include "original_stc_adapter.h"

#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

static void one_round_trip(int worker) {
    const int cover_length = 2048 + (worker % 5) * 16;
    const int message_length = 512;
    const int height = 10;
    std::vector<unsigned char> cover((size_t)cover_length);
    std::vector<double> costs((size_t)cover_length);
    std::vector<unsigned char> message((size_t)message_length);
    for (int i = 0; i < cover_length; ++i) {
        cover[(size_t)i] =
            (unsigned char)((i * 17 + worker * 7 + 3) & 1);
        costs[(size_t)i] =
            0.25 + (double)((i * 29 + worker * 11) % 101) / 50.0;
    }
    for (int i = 0; i < message_length; ++i) {
        message[(size_t)i] =
            (unsigned char)((i * 13 + worker * 5 + 1) & 1);
    }
    std::vector<unsigned char> stego = original_stc_embed_adapter(
        cover, costs, message, height);
    std::vector<unsigned char> recovered = original_stc_extract_adapter(
        stego, message_length, height);
    if (recovered != message) {
        throw std::runtime_error(
            "native concurrent STC round-trip mismatch");
    }
}

int main() {
    try {
        constexpr int workers = 64;
        std::vector<std::future<void>> futures;
        futures.reserve(workers);
        for (int worker = 0; worker < workers; ++worker) {
            futures.push_back(std::async(
                std::launch::async, [worker] { one_round_trip(worker); }));
        }
        for (auto& future : futures) future.get();
        std::cout
            << "NativeStcConcurrencyTest: 64 CONCURRENT ROUND TRIPS PASSED\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "NativeStcConcurrencyTest: FAILED: "
                  << failure.what() << '\n';
        return 1;
    }
}
