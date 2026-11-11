#pragma once

#include <cuda_runtime.h>

#include <atomic>

// Runtime-only crash containment shared by every CUDA translation unit in
// librapcuda.so.  This file intentionally contains no RAP/SI-UNIWARD/STC
// mathematics.  It only prevents reuse of a CUDA context after a terminal
// device execution fault has been observed.
namespace rap_cuda_crash_guard {

// C++17 inline variables provide one process-wide state across the CUDA
// translation units linked into librapcuda.so.
inline std::atomic<int> first_fatal_error_code{0};

inline bool is_terminal_device_error(cudaError_t status) noexcept {
    // CUDA execution faults for which NVIDIA documents the process/context as
    // unusable for continued work, or which are unsafe to recover from inside
    // the current JVM process.  Numeric codes keep this guard compatible with
    // CUDA toolkit headers that may not expose every newer symbolic enum.
    switch (static_cast<int>(status)) {
        case 700: // cudaErrorIllegalAddress
        case 702: // cudaErrorLaunchTimeout
        case 710: // cudaErrorAssert
        case 714: // cudaErrorHardwareStackError
        case 715: // cudaErrorIllegalInstruction
        case 716: // cudaErrorMisalignedAddress
        case 717: // cudaErrorInvalidAddressSpace
        case 718: // cudaErrorInvalidPc
        case 719: // cudaErrorLaunchFailure
            return true;
        default:
            return false;
    }
}

inline void observe(cudaError_t status) noexcept {
    if (!is_terminal_device_error(status)) return;
    int expected = 0;
    const int code = static_cast<int>(status);
    first_fatal_error_code.compare_exchange_strong(
            expected, code, std::memory_order_acq_rel,
            std::memory_order_acquire);
}

inline bool poisoned() noexcept {
    return first_fatal_error_code.load(std::memory_order_acquire) != 0;
}

inline int first_error_code() noexcept {
    return first_fatal_error_code.load(std::memory_order_acquire);
}

} // namespace rap_cuda_crash_guard
