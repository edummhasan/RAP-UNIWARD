#include <jni.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <memory>
#include <new>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

namespace {
constexpr int kFilters = 3;
constexpr int kImpactN = 23;
constexpr int kImpactArea = kImpactN * kImpactN;
constexpr int kMaxBandCount = 64;

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
    if (env->ExceptionCheck()) return;
    jclass klass = env->FindClass(class_name);
    if (klass != nullptr) {
        env->ThrowNew(klass, message.c_str());
    }
}

std::string hip_error(const char* where, hipError_t status) {
    std::ostringstream out;
    out << where << " failed: " << hipGetErrorString(status)
        << " (" << static_cast<int>(status) << ")";
    return out.str();
}

bool check_hip(JNIEnv* env, const char* where, hipError_t status) {
    if (status == hipSuccess) return true;
    throw_java(env, "java/lang/IllegalStateException", hip_error(where, status));
    return false;
}

/**
 * Process-lifetime workspace for the single Radeon device used by this build.
 * Java already bounds GPU access with RAP_GPU_SLOTS; the native mutex keeps
 * correctness even if a caller configures more than one Java GPU slot.
 */
struct DeviceWorkspace {
    std::mutex mutex;
    double* denominators = nullptr;
    double* impacts = nullptr;
    double* output = nullptr;
    std::size_t denominator_capacity = 0;
    std::size_t impact_capacity = 0;
    std::size_t output_capacity = 0;
    // Reused native staging memory: Java arrays are never held during HIP transfers.
    std::unique_ptr<double[]> host_staging;
    std::size_t host_capacity = 0;

    bool ensure_host(std::size_t required_elements) {
        if (host_staging && required_elements <= host_capacity) return true;
        std::unique_ptr<double[]> replacement(new (std::nothrow) double[required_elements]);
        if (!replacement) return false;
        host_staging = std::move(replacement);
        host_capacity = required_elements;
        return true;
    }

    hipError_t ensure_double(
            double** pointer,
            std::size_t* capacity,
            std::size_t required_bytes) {
        if (required_bytes <= *capacity && *pointer != nullptr) return hipSuccess;
        if (*pointer != nullptr) {
            hipError_t free_status = hipFree(*pointer);
            *pointer = nullptr;
            *capacity = 0;
            if (free_status != hipSuccess) return free_status;
        }
        hipError_t status = hipMalloc(
                reinterpret_cast<void**>(pointer), required_bytes);
        if (status == hipSuccess) *capacity = required_bytes;
        return status;
    }

    hipError_t ensure(
            std::size_t denominator_bytes,
            std::size_t impact_bytes,
            std::size_t output_bytes) {
        hipError_t status = ensure_double(
                &denominators, &denominator_capacity, denominator_bytes);
        if (status == hipSuccess) {
            status = ensure_double(&impacts, &impact_capacity, impact_bytes);
        }
        if (status == hipSuccess) {
            status = ensure_double(&output, &output_capacity, output_bytes);
        }
        return status;
    }
};

DeviceWorkspace& workspace() {
    static DeviceWorkspace instance;
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
                    + static_cast<std::size_t>(ry) * residual_width
                    + x0;
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
Java_thesis_rap_gpu_HipSiUniwardNative_nativeRuntimeAvailable(
        JNIEnv*, jclass) {
    int count = 0;
    const hipError_t status = hipGetDeviceCount(&count);
    return (status == hipSuccess && count > 0) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_thesis_rap_gpu_HipSiUniwardNative_nativeDeviceName(
        JNIEnv* env, jclass) {
    int count = 0;
    hipError_t status = hipGetDeviceCount(&count);
    if (!check_hip(env, "hipGetDeviceCount", status)) return nullptr;
    if (count < 1) {
        throw_java(env, "java/lang/IllegalStateException", "No HIP device is available");
        return nullptr;
    }
    hipDeviceProp_t prop{};
    status = hipGetDeviceProperties(&prop, 0);
    if (!check_hip(env, "hipGetDeviceProperties", status)) return nullptr;
    return env->NewStringUTF(prop.name);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_thesis_rap_gpu_HipSiUniwardNative_nativeComputeCosts(
        JNIEnv* env,
        jclass,
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
        throw_java(env, "java/lang/IllegalArgumentException",
                   "HIP cost arrays must be non-null");
        return nullptr;
    }
    if (residual_width <= 0 || residual_height <= 0
            || width_blocks <= 0 || height_blocks <= 0) {
        throw_java(env, "java/lang/IllegalArgumentException",
                   "HIP dimensions must be positive");
        return nullptr;
    }
    if (band_count <= 0 || band_count > kMaxBandCount) {
        throw_java(env, "java/lang/IllegalArgumentException",
                   "band count must be in [1,64]");
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

    if (static_cast<std::size_t>(env->GetArrayLength(denominators_array))
            != expected_denominators) {
        throw_java(env, "java/lang/IllegalArgumentException",
                   "unexpected denominator array length");
        return nullptr;
    }
    if (static_cast<std::size_t>(env->GetArrayLength(impacts_array))
            != expected_impacts) {
        throw_java(env, "java/lang/IllegalArgumentException",
                   "unexpected impact array length");
        return nullptr;
    }
    if (output_count > static_cast<std::size_t>(
            std::numeric_limits<jsize>::max())) {
        throw_java(env, "java/lang/IllegalArgumentException",
                   "HIP output is too large for a Java array");
        return nullptr;
    }

    if (!check_hip(env, "hipSetDevice", hipSetDevice(0))) return nullptr;

    const std::size_t denominator_bytes = expected_denominators * sizeof(double);
    const std::size_t impact_bytes = expected_impacts * sizeof(double);
    const std::size_t output_bytes = output_count * sizeof(double);

    DeviceWorkspace& ws = workspace();
    std::lock_guard<std::mutex> lock(ws.mutex);

    hipError_t status = ws.ensure(
            denominator_bytes, impact_bytes, output_bytes);
    if (status != hipSuccess) {
        throw_java(env, "java/lang/IllegalStateException",
                   hip_error("HIP workspace allocation", status));
        return nullptr;
    }

    if (!ws.ensure_host(std::max(expected_denominators,
            std::max(expected_impacts, output_count)))) {
        throw_java(env, "java/lang/OutOfMemoryError", "HIP host staging allocation failed");
        return nullptr;
    }
    env->GetDoubleArrayRegion(denominators_array, 0,
            static_cast<jsize>(expected_denominators), ws.host_staging.get());
    if (env->ExceptionCheck()) return nullptr;
    status = hipMemcpy(ws.denominators, ws.host_staging.get(),
            denominator_bytes, hipMemcpyHostToDevice);
    if (!check_hip(env, "HIP denominator copy", status)) return nullptr;

    env->GetDoubleArrayRegion(impacts_array, 0,
            static_cast<jsize>(expected_impacts), ws.host_staging.get());
    if (env->ExceptionCheck()) return nullptr;
    status = hipMemcpy(ws.impacts, ws.host_staging.get(),
            impact_bytes, hipMemcpyHostToDevice);
    if (!check_hip(env, "HIP impact copy", status)) return nullptr;

    const dim3 block(static_cast<unsigned int>(band_count), 1, 1);
    const dim3 grid(
            static_cast<unsigned int>(width_blocks * height_blocks), 1, 1);
    hipLaunchKernelGGL(si_cost_kernel,
                       grid,
                       block,
                       0,
                       0,
                       ws.denominators,
                       residual_width,
                       residual_height,
                       ws.impacts,
                       width_blocks,
                       height_blocks,
                       band_count,
                       static_cast<double>(wet_cost),
                       static_cast<int>(window_offset),
                       ws.output);

    status = hipGetLastError();
    if (status != hipSuccess) {
        throw_java(env, "java/lang/IllegalStateException",
                   hip_error("HIP kernel launch", status));
        return nullptr;
    }

    jdoubleArray result = env->NewDoubleArray(
            static_cast<jsize>(output_count));
    if (result == nullptr) return nullptr;

    // Synchronous copy waits for the preceding default-stream kernel.
    status = hipMemcpy(ws.host_staging.get(), ws.output,
            output_bytes, hipMemcpyDeviceToHost);

    if (status != hipSuccess) {
        throw_java(env, "java/lang/IllegalStateException",
                   hip_error("HIP output copy", status));
        return nullptr;
    }
    env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output_count),
            ws.host_staging.get());
    if (env->ExceptionCheck()) return nullptr;
    return result;
}
