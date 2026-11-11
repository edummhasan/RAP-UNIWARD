#include <jni.h>
#include <cuda_runtime.h>
#include "rap_cuda_crash_guard.cuh"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <new>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int kFilters = 3;
constexpr int kImpactN = 23;
constexpr int kImpactArea = kImpactN * kImpactN;
constexpr int kMaxBandCount = 64;

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
    if (env->ExceptionCheck()) return;
    jclass klass = env->FindClass(class_name);
    if (klass != nullptr) env->ThrowNew(klass, message.c_str());
}

std::string cuda_error(const char* where, cudaError_t status) {
    std::ostringstream out;
    out << where << " failed: " << cudaGetErrorString(status)
        << " (" << static_cast<int>(status) << ")";
    return out.str();
}

bool check_cuda(JNIEnv* env, const char* where, cudaError_t status) {
    if (status == cudaSuccess) return true;
    rap_cuda_crash_guard::observe(status);
    std::string message = cuda_error(where, status);
    if (rap_cuda_crash_guard::poisoned()) {
        message += "; CUDA context marked unusable after fatal device error; restart process before further CUDA work";
    }
    throw_java(env, "java/lang/IllegalStateException", message);
    return false;
}

bool preflight_cuda(JNIEnv* env) {
    if (rap_cuda_crash_guard::poisoned()) {
        throw_java(env, "java/lang/IllegalStateException",
                "CUDA context is already poisoned by fatal device error (first code="
                + std::to_string(rap_cuda_crash_guard::first_error_code())
                + "); restart process before further CUDA work");
        return false;
    }
    return check_cuda(env, "CUDA preflight pending error", cudaGetLastError());
}

int parse_env_int(const char* name, int fallback, int minimum, int maximum) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    char* end = nullptr;
    long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < minimum || parsed > maximum) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

int selected_device() {
    return parse_env_int("RAP_CUDA_DEVICE", 0, 0, 1024);
}

struct StreamWorkspace {
    cudaStream_t stream = nullptr;
    double* denominators = nullptr;
    double* impacts = nullptr;
    double* output = nullptr;
    std::size_t denominator_capacity = 0;
    std::size_t impact_capacity = 0;
    std::size_t output_capacity = 0;
    double* host_denominators = nullptr;
    double* host_impacts = nullptr;
    double* host_output = nullptr;
    std::size_t host_denominator_capacity = 0;
    std::size_t host_impact_capacity = 0;
    std::size_t host_output_capacity = 0;

    cudaError_t initialize() {
        if (stream != nullptr) return cudaSuccess;
        return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    static cudaError_t ensure_device(
            double** pointer, std::size_t* capacity, std::size_t required_bytes) {
        if (*pointer != nullptr && required_bytes <= *capacity) return cudaSuccess;
        if (*pointer != nullptr) {
            cudaError_t status = cudaFree(*pointer);
            *pointer = nullptr;
            *capacity = 0;
            if (status != cudaSuccess) return status;
        }
        cudaError_t status = cudaMalloc(reinterpret_cast<void**>(pointer), required_bytes);
        if (status == cudaSuccess) *capacity = required_bytes;
        return status;
    }

    static cudaError_t ensure_host(
            double** pointer, std::size_t* capacity, std::size_t required_bytes) {
        if (*pointer != nullptr && required_bytes <= *capacity) return cudaSuccess;
        if (*pointer != nullptr) {
            cudaError_t status = cudaFreeHost(*pointer);
            *pointer = nullptr;
            *capacity = 0;
            if (status != cudaSuccess) return status;
        }
        cudaError_t status = cudaHostAlloc(
                reinterpret_cast<void**>(pointer), required_bytes,
                cudaHostAllocPortable);
        if (status == cudaSuccess) *capacity = required_bytes;
        return status;
    }

    cudaError_t ensure(
            std::size_t denominator_bytes,
            std::size_t impact_bytes,
            std::size_t output_bytes) {
        cudaError_t status = initialize();
        if (status == cudaSuccess) status = ensure_device(
                &denominators, &denominator_capacity, denominator_bytes);
        if (status == cudaSuccess) status = ensure_device(
                &impacts, &impact_capacity, impact_bytes);
        if (status == cudaSuccess) status = ensure_device(
                &output, &output_capacity, output_bytes);
        if (status == cudaSuccess) status = ensure_host(
                &host_denominators, &host_denominator_capacity, denominator_bytes);
        if (status == cudaSuccess) status = ensure_host(
                &host_impacts, &host_impact_capacity, impact_bytes);
        if (status == cudaSuccess) status = ensure_host(
                &host_output, &host_output_capacity, output_bytes);
        return status;
    }

    ~StreamWorkspace() {
        // After a terminal execution fault the CUDA context is not safe to
        // reuse.  Avoid secondary runtime calls that can mask the first error.
        if (rap_cuda_crash_guard::poisoned()) return;
        if (denominators) cudaFree(denominators);
        if (impacts) cudaFree(impacts);
        if (output) cudaFree(output);
        if (host_denominators) cudaFreeHost(host_denominators);
        if (host_impacts) cudaFreeHost(host_impacts);
        if (host_output) cudaFreeHost(host_output);
        if (stream) cudaStreamDestroy(stream);
    }
};

class WorkspacePool {
public:
    class Lease {
    public:
        Lease() = default;
        Lease(WorkspacePool* owner, std::size_t index)
                : owner_(owner), index_(index) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept
                : owner_(other.owner_), index_(other.index_) {
            other.owner_ = nullptr;
        }
        ~Lease() { if (owner_) owner_->release(index_); }
        StreamWorkspace& get() { return *owner_->workspaces_[index_]; }
    private:
        WorkspacePool* owner_ = nullptr;
        std::size_t index_ = 0;
    };

    Lease acquire(JNIEnv* env) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (rap_cuda_crash_guard::poisoned()) {
            throw_java(env, "java/lang/IllegalStateException",
                    "CUDA SI workspace acquisition refused: CUDA context is poisoned");
            return Lease();
        }
        if (!initialized_) {
            const int configured = parse_env_int(
                    "RAP_CUDA_STREAMS", 8, 1, 64);
            workspaces_.reserve(static_cast<std::size_t>(configured));
            busy_.assign(static_cast<std::size_t>(configured), false);
            for (int i = 0; i < configured; ++i) {
                auto workspace = std::make_unique<StreamWorkspace>();
                cudaError_t status = workspace->initialize();
                if (status != cudaSuccess) {
                    throw_java(env, "java/lang/IllegalStateException",
                               cuda_error("cudaStreamCreateWithFlags", status));
                    return Lease();
                }
                workspaces_.push_back(std::move(workspace));
            }
            initialized_ = true;
        }
        condition_.wait(lock, [&]() {
            return rap_cuda_crash_guard::poisoned()
                    || std::any_of(busy_.begin(), busy_.end(), [](bool value) { return !value; });
        });
        if (rap_cuda_crash_guard::poisoned()) {
            throw_java(env, "java/lang/IllegalStateException",
                    "CUDA SI workspace wait aborted: CUDA context is poisoned");
            return Lease();
        }
        for (std::size_t i = 0; i < busy_.size(); ++i) {
            if (!busy_[i]) {
                busy_[i] = true;
                return Lease(this, i);
            }
        }
        throw_java(env, "java/lang/IllegalStateException", "CUDA workspace-pool invariant failed");
        return Lease();
    }

    int size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return initialized_ ? static_cast<int>(workspaces_.size())
                            : parse_env_int("RAP_CUDA_STREAMS", 8, 1, 64);
    }

private:
    void release(std::size_t index) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (index < busy_.size()) busy_[index] = false;
        }
        if (rap_cuda_crash_guard::poisoned()) condition_.notify_all();
        else condition_.notify_one();
    }

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool initialized_ = false;
    std::vector<std::unique_ptr<StreamWorkspace>> workspaces_;
    std::vector<bool> busy_;

    friend class Lease;
};

WorkspacePool& pool() {
    static WorkspacePool instance;
    return instance;
}

__global__ void si_cost_kernel(
        const double* denominators,
        int residual_width,
        int residual_height,
        const double* impacts_position_major,
        int width_blocks,
        int height_blocks,
        int band_count,
        double wet_cost,
        int window_offset,
        double* output) {
    const int block_linear = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int block_count = width_blocks * height_blocks;
    if (block_linear >= block_count || lane >= band_count) return;

    const int bx = block_linear % width_blocks;
    const int by = block_linear / width_blocks;
    const int x0 = bx * 8 + window_offset;
    const int y0 = by * 8 + window_offset;
    double cost = 0.0;
    const std::size_t residual_plane =
            static_cast<std::size_t>(residual_width) * residual_height;

    for (int f = 0; f < kFilters; ++f) {
        const std::size_t residual_filter_base =
                static_cast<std::size_t>(f) * residual_plane;
        for (int iy = 0; iy < kImpactN; ++iy) {
            const int ry = y0 + iy;
            if (ry < 0 || ry >= residual_height) {
                output[static_cast<std::size_t>(block_linear) * band_count + lane] = wet_cost;
                return;
            }
            const std::size_t denominator_index = residual_filter_base
                    + static_cast<std::size_t>(ry) * residual_width + x0;
            const std::size_t impact_row_base =
                    static_cast<std::size_t>(f * kImpactArea + iy * kImpactN)
                    * static_cast<std::size_t>(band_count);
            for (int ix = 0; ix < kImpactN; ++ix) {
                const int rx = x0 + ix;
                if (rx < 0 || rx >= residual_width) {
                    output[static_cast<std::size_t>(block_linear) * band_count + lane] = wet_cost;
                    return;
                }
                const std::size_t impact_index = impact_row_base
                        + static_cast<std::size_t>(ix) * band_count
                        + static_cast<std::size_t>(lane);
                cost += impacts_position_major[impact_index]
                        / denominators[denominator_index + ix];
            }
        }
    }

    const std::size_t out_index =
            static_cast<std::size_t>(block_linear) * band_count + lane;
    if (!isfinite(cost) || cost < 0.0) {
        output[out_index] = wet_cost;
    } else {
        const double legal_cap = wet_cost - 1.0;
        output[out_index] = cost < legal_cap ? cost : legal_cap;
    }
}
}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_thesis_rap_gpu_CudaSiUniwardNative_nativeRuntimeAvailable(JNIEnv*, jclass) {
    if (rap_cuda_crash_guard::poisoned()) return JNI_FALSE;
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) rap_cuda_crash_guard::observe(status);
    const int device = selected_device();
    return (status == cudaSuccess && count > device) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_thesis_rap_gpu_CudaSiUniwardNative_nativeDeviceName(JNIEnv* env, jclass) {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (!check_cuda(env, "cudaGetDeviceCount", status)) return nullptr;
    const int device = selected_device();
    if (device < 0 || device >= count) {
        throw_java(env, "java/lang/IllegalStateException", "Configured RAP_CUDA_DEVICE is unavailable");
        return nullptr;
    }
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, device);
    if (!check_cuda(env, "cudaGetDeviceProperties", status)) return nullptr;
    return env->NewStringUTF(prop.name);
}

extern "C" JNIEXPORT jint JNICALL
Java_thesis_rap_gpu_CudaSiUniwardNative_nativeStreamPoolSize(JNIEnv*, jclass) {
    return static_cast<jint>(pool().size());
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_thesis_rap_gpu_CudaSiUniwardNative_nativeComputeCosts(
        JNIEnv* env, jclass,
        jdoubleArray denominators_array,
        jint residual_width,
        jint residual_height,
        jdoubleArray impacts_array,
        jint band_count,
        jint width_blocks,
        jint height_blocks,
        jdouble wet_cost,
        jint window_offset) {
    if (denominators_array == nullptr || impacts_array == nullptr) {
        throw_java(env, "java/lang/IllegalArgumentException", "CUDA cost arrays must be non-null");
        return nullptr;
    }
    if (residual_width <= 0 || residual_height <= 0 || width_blocks <= 0 || height_blocks <= 0) {
        throw_java(env, "java/lang/IllegalArgumentException", "CUDA dimensions must be positive");
        return nullptr;
    }
    if (band_count <= 0 || band_count > kMaxBandCount) {
        throw_java(env, "java/lang/IllegalArgumentException", "band count must be in [1,64]");
        return nullptr;
    }

    const std::size_t expected_denominators = static_cast<std::size_t>(kFilters)
            * static_cast<std::size_t>(residual_width)
            * static_cast<std::size_t>(residual_height);
    const std::size_t expected_impacts = static_cast<std::size_t>(kFilters)
            * kImpactArea * static_cast<std::size_t>(band_count);
    const std::size_t output_count = static_cast<std::size_t>(width_blocks)
            * static_cast<std::size_t>(height_blocks)
            * static_cast<std::size_t>(band_count);

    if (static_cast<std::size_t>(env->GetArrayLength(denominators_array)) != expected_denominators
            || static_cast<std::size_t>(env->GetArrayLength(impacts_array)) != expected_impacts) {
        throw_java(env, "java/lang/IllegalArgumentException", "unexpected CUDA input array length");
        return nullptr;
    }
    if (output_count > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
        throw_java(env, "java/lang/IllegalArgumentException", "CUDA output is too large for a Java array");
        return nullptr;
    }

    const int device = selected_device();
    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(device))) return nullptr;
    if (!preflight_cuda(env)) return nullptr;

    const std::size_t denominator_bytes = expected_denominators * sizeof(double);
    const std::size_t impact_bytes = expected_impacts * sizeof(double);
    const std::size_t output_bytes = output_count * sizeof(double);

    auto lease = pool().acquire(env);
    if (env->ExceptionCheck()) return nullptr;
    StreamWorkspace& ws = lease.get();
    cudaError_t status = ws.ensure(denominator_bytes, impact_bytes, output_bytes);
    if (!check_cuda(env, "CUDA stream-workspace allocation", status)) return nullptr;

    env->GetDoubleArrayRegion(denominators_array, 0,
            static_cast<jsize>(expected_denominators), ws.host_denominators);
    if (env->ExceptionCheck()) return nullptr;
    env->GetDoubleArrayRegion(impacts_array, 0,
            static_cast<jsize>(expected_impacts), ws.host_impacts);
    if (env->ExceptionCheck()) return nullptr;

    status = cudaMemcpyAsync(ws.denominators, ws.host_denominators,
            denominator_bytes, cudaMemcpyHostToDevice, ws.stream);
    if (!check_cuda(env, "CUDA denominator async copy", status)) return nullptr;
    status = cudaMemcpyAsync(ws.impacts, ws.host_impacts,
            impact_bytes, cudaMemcpyHostToDevice, ws.stream);
    if (!check_cuda(env, "CUDA impact async copy", status)) return nullptr;

    const dim3 block(static_cast<unsigned int>(band_count), 1, 1);
    const dim3 grid(static_cast<unsigned int>(width_blocks * height_blocks), 1, 1);
    si_cost_kernel<<<grid, block, 0, ws.stream>>>(
            ws.denominators, residual_width, residual_height, ws.impacts,
            width_blocks, height_blocks, band_count,
            static_cast<double>(wet_cost), static_cast<int>(window_offset), ws.output);
    status = cudaGetLastError();
    if (!check_cuda(env, "CUDA SI kernel launch", status)) return nullptr;

    status = cudaMemcpyAsync(ws.host_output, ws.output,
            output_bytes, cudaMemcpyDeviceToHost, ws.stream);
    if (!check_cuda(env, "CUDA output async copy", status)) return nullptr;
    status = cudaStreamSynchronize(ws.stream);
    if (!check_cuda(env, "CUDA SI stream synchronize", status)) return nullptr;

    jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(output_count));
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output_count), ws.host_output);
    return env->ExceptionCheck() ? nullptr : result;
}
