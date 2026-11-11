#include <jni.h>
#include <cuda_runtime.h>
#include "rap_cuda_crash_guard.cuh"

// else included by this file is documented to provide it.
#include <math_constants.h>

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "../native/original_stc/common.h"

namespace {

constexpr double kMaximumCost = 1.0e13;

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
                "CUDA STC refused because CUDA context is already poisoned (first code="
                + std::to_string(rap_cuda_crash_guard::first_error_code()) + ")");
        return false;
    }
    return check_cuda(env, "CUDA STC preflight pending error", cudaGetLastError());
}

int parse_env_int(const char* name, int fallback, int minimum, int maximum) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < minimum || value > maximum) {
        return fallback;
    }
    return static_cast<int>(value);
}

int selected_device() {
    return parse_env_int("RAP_CUDA_DEVICE", 0, 0, 1024);
}

bool runtime_available() {
    if (rap_cuda_crash_guard::poisoned()) return false;
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) rap_cuda_crash_guard::observe(status);
    return status == cudaSuccess && selected_device() < count;
}

class StcStreamPool {
public:
    class Lease {
    public:
        Lease() = default;
        Lease(StcStreamPool* owner, std::size_t index) : owner_(owner), index_(index) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept : owner_(other.owner_), index_(other.index_) { other.owner_ = nullptr; }
        ~Lease() { if (owner_) owner_->release(index_); }
        cudaStream_t stream() const { return owner_ ? owner_->streams_[index_] : nullptr; }
    private:
        StcStreamPool* owner_ = nullptr;
        std::size_t index_ = 0;
    };

    Lease acquire(JNIEnv* env) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (rap_cuda_crash_guard::poisoned()) {
            throw_java(env, "java/lang/IllegalStateException",
                    "CUDA STC stream acquisition refused: CUDA context is poisoned");
            return Lease();
        }
        initialize_locked(env);
        if (env->ExceptionCheck() || streams_.empty()) return Lease();
        condition_.wait(lock, [&]() {
            return rap_cuda_crash_guard::poisoned()
                    || std::any_of(busy_.begin(), busy_.end(), [](bool v) { return !v; });
        });
        if (rap_cuda_crash_guard::poisoned()) {
            throw_java(env, "java/lang/IllegalStateException",
                    "CUDA STC stream wait aborted: CUDA context is poisoned");
            return Lease();
        }
        for (std::size_t i = 0; i < busy_.size(); ++i) {
            if (!busy_[i]) {
                busy_[i] = true;
                return Lease(this, i);
            }
        }
        throw_java(env, "java/lang/IllegalStateException", "CUDA STC stream-pool invariant failed");
        return Lease();
    }

    ~StcStreamPool() {
        if (rap_cuda_crash_guard::poisoned()) return;
        for (cudaStream_t stream : streams_) if (stream) cudaStreamDestroy(stream);
    }

private:
    void initialize_locked(JNIEnv* env) {
        if (initialized_) return;
        int fallback = parse_env_int("RAP_CUDA_STREAMS", 8, 1, 64);
        int count = parse_env_int("RAP_CUDA_STC_STREAMS", fallback, 1, 64);
        streams_.resize(static_cast<std::size_t>(count), nullptr);
        busy_.assign(static_cast<std::size_t>(count), false);
        for (int i = 0; i < count; ++i) {
            cudaError_t status = cudaStreamCreateWithFlags(
                    &streams_[static_cast<std::size_t>(i)], cudaStreamNonBlocking);
            if (status != cudaSuccess) {
                throw_java(env, "java/lang/IllegalStateException",
                        cuda_error("cudaStreamCreateWithFlags(STC)", status));
                return;
            }
        }
        initialized_ = true;
    }

    void release(std::size_t index) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (index < busy_.size()) busy_[index] = false;
        }
        if (rap_cuda_crash_guard::poisoned()) condition_.notify_all();
        else condition_.notify_one();
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    bool initialized_ = false;
    std::vector<cudaStream_t> streams_;
    std::vector<bool> busy_;
    friend class Lease;
};

StcStreamPool& stc_stream_pool() {
    static StcStreamPool pool;
    return pool;
}

struct FreeDeleter {
    void operator()(u32* pointer) const noexcept { std::free(pointer); }
};
using U32Ptr = std::unique_ptr<u32, FreeDeleter>;

struct PlanData {
    std::vector<u32> carrier_columns;
    std::vector<int> message_offsets;
};

PlanData make_plan(int cover_length, int message_length, int height) {
    const stc_width_plan plan = stc_make_width_plan(cover_length, message_length);
    U32Ptr short_columns(getMatrix(plan.shorter, height));
    U32Ptr long_columns(getMatrix(plan.longer, height));
    if (!short_columns || !long_columns) {
        throw stc_exception("Could not construct CUDA STC parity-check columns.", 2);
    }

    PlanData out;
    out.carrier_columns.resize(static_cast<std::size_t>(cover_length));
    out.message_offsets.resize(static_cast<std::size_t>(message_length) + 1u);
    int carrier = 0;
    out.message_offsets[0] = 0;
    for (int j = 0; j < message_length; ++j) {
        const int choice = plan.matrix_choice[static_cast<std::size_t>(j)];
        const int width = plan.widths[static_cast<std::size_t>(j)];
        const u32* columns = choice == 0 ? short_columns.get() : long_columns.get();
        for (int k = 0; k < width; ++k) {
            if (carrier >= cover_length) {
                throw stc_exception("CUDA STC width plan exceeded cover length.", 1);
            }
            out.carrier_columns[static_cast<std::size_t>(carrier++)] = columns[k];
        }
        out.message_offsets[static_cast<std::size_t>(j) + 1u] = carrier;
    }
    if (carrier != cover_length) {
        throw stc_exception("CUDA STC width plan did not consume every carrier.", 1);
    }
    return out;
}

__global__ void stc_embed_kernel(
        const std::uint8_t* cover,
        const double* costs,
        int cover_length,
        const std::uint8_t* message,
        int message_length,
        int height,
        const std::uint32_t* columns,
        const int* message_offsets,
        std::uint32_t* path,
        std::uint8_t* stego,
        int* status) {
    extern __shared__ double shared[];
    const std::uint32_t state_count = 1u << height;
    double* prices_a = shared;
    double* prices_b = shared + state_count;
    double* prices = prices_a;
    double* next_prices = prices_b;
    const double infinity = CUDART_INF;
    const std::size_t path_words_per_carrier =
            (static_cast<std::size_t>(state_count) + 31u) / 32u;

    for (std::uint32_t state = threadIdx.x;
            state < state_count;
            state += blockDim.x) {
        prices[state] = state == 0u ? 0.0 : infinity;
        next_prices[state] = infinity;
    }
    __syncthreads();

    std::uint32_t column_mask = state_count - 1u;
    for (int j = 0; j < message_length; ++j) {
        const int begin = message_offsets[j];
        const int end = message_offsets[j + 1];
        for (int carrier = begin; carrier < end; ++carrier) {
            const std::uint32_t column = columns[carrier] & column_mask;
            const double output_zero_cost = cover[carrier] == 0u
                    ? 0.0 : costs[carrier];
            const double output_one_cost = cover[carrier] == 1u
                    ? 0.0 : costs[carrier];
            const std::size_t path_base = static_cast<std::size_t>(carrier)
                    * path_words_per_carrier;

            for (std::uint32_t state = threadIdx.x;
                    state < state_count;
                    state += blockDim.x) {
                const double candidate_zero = prices[state] + output_zero_cost;
                const double candidate_one = prices[state ^ column] + output_one_cost;
                const bool choose_one = candidate_one <= candidate_zero;
                next_prices[state] = choose_one ? candidate_one : candidate_zero;
                if (choose_one) {
                    atomicOr(path + path_base + (state >> 5u),
                            1u << (state & 31u));
                }
            }
            __syncthreads();
            double* swap = prices;
            prices = next_prices;
            next_prices = swap;
            __syncthreads();
        }

        const std::uint32_t constrained_bit = message[j];
        // Use the inactive price vector as a constraint scratch buffer.  An
        // in-place lower-half copy would race with threads clearing the upper
        // half before those values were read.
        for (std::uint32_t state = threadIdx.x;
                state < state_count;
                state += blockDim.x) {
            if (state < state_count / 2u) {
                next_prices[state] = prices[(state << 1u) | constrained_bit];
            } else {
                next_prices[state] = infinity;
            }
        }
        __syncthreads();
        double* constraint_swap = prices;
        prices = next_prices;
        next_prices = constraint_swap;
        __syncthreads();
        if (message_length - j <= height) column_mask >>= 1u;
    }

    if (threadIdx.x == 0) {
        if (!isfinite(prices[0])) {
            *status = 4;
            return;
        }

        int reverse_carrier = cover_length - 1;
        std::uint32_t state = 0u;
        std::uint32_t reverse_mask = 0u;
        for (int j = message_length - 1; j >= 0; --j) {
            const int begin = message_offsets[j];
            const int end = message_offsets[j + 1];
            for (int carrier = end - 1; carrier >= begin; --carrier, --reverse_carrier) {
                if (carrier == end - 1) {
                    state = (state << 1u) | message[j];
                    if (message_length - j <= height) {
                        reverse_mask = (reverse_mask << 1u) | 1u;
                    }
                }
                const std::size_t path_base = static_cast<std::size_t>(carrier)
                        * path_words_per_carrier;
                const bool chose_one = (path[path_base + (state >> 5u)]
                        & (1u << (state & 31u))) != 0u;
                stego[carrier] = chose_one ? 1u : 0u;
                if (chose_one) state ^= columns[carrier] & reverse_mask;
            }
        }
        *status = reverse_carrier == -1 ? 0 : 1;
    }
}

__global__ void stc_extract_kernel(
        const std::uint8_t* stego,
        int cover_length,
        int message_length,
        int height,
        const std::uint32_t* columns,
        const int* carrier_message_index,
        std::uint32_t* syndrome) {
    int carrier = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (carrier >= cover_length || stego[carrier] == 0u) return;
    const int j = carrier_message_index[carrier];
    const int remaining = message_length - j;
    const int active_height = remaining < height ? remaining : height;
    const std::uint32_t column = columns[carrier];
    for (int bit = 0; bit < active_height; ++bit) {
        if ((column & (1u << bit)) != 0u) {
            atomicXor(syndrome + j + bit, 1u);
        }
    }
}


template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t count) { allocate(count); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept
            : pointer_(other.pointer_), count_(other.count_) {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            pointer_ = other.pointer_;
            count_ = other.count_;
            other.pointer_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~PinnedBuffer() { reset(); }

    void allocate(std::size_t count) {
        reset();
        count_ = count;
        if (count_ == 0) return;
        cudaError_t status = cudaHostAlloc(
                reinterpret_cast<void**>(&pointer_),
                count_ * sizeof(T), cudaHostAllocDefault);
        if (status != cudaSuccess) {
            pointer_ = nullptr;
            count_ = 0;
            throw std::runtime_error(cuda_error("cudaHostAlloc(STC)", status));
        }
    }
    void reset() noexcept {
        // Do not issue more CUDA runtime calls after a terminal device fault;
        // the process must be restarted before CUDA is reused.
        if (pointer_ && !rap_cuda_crash_guard::poisoned()) cudaFreeHost(pointer_);
        pointer_ = nullptr;
        count_ = 0;
    }
    T* data() noexcept { return pointer_; }
    const T* data() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return count_; }

private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
PinnedBuffer<T> pinned_copy(const std::vector<T>& values) {
    PinnedBuffer<T> out(values.size());
    if (!values.empty()) {
        std::copy(values.begin(), values.end(), out.data());
    }
    return out;
}

std::vector<std::uint8_t> read_bytes(JNIEnv* env, jbyteArray array) {
    if (!array) throw std::invalid_argument("byte array is null");
    const jsize n = env->GetArrayLength(array);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(n));
    if (n > 0) {
        env->GetByteArrayRegion(array, 0, n,
                reinterpret_cast<jbyte*>(out.data()));
    }
    if (env->ExceptionCheck()) throw std::runtime_error("JNI byte copy failed");
    return out;
}

std::vector<double> read_doubles(JNIEnv* env, jdoubleArray array) {
    if (!array) throw std::invalid_argument("double array is null");
    const jsize n = env->GetArrayLength(array);
    std::vector<double> out(static_cast<std::size_t>(n));
    if (n > 0) env->GetDoubleArrayRegion(array, 0, n, out.data());
    if (env->ExceptionCheck()) throw std::runtime_error("JNI double copy failed");
    return out;
}

jbyteArray to_java_bytes(JNIEnv* env, const std::vector<std::uint8_t>& values) {
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
        throw std::overflow_error("CUDA STC output exceeds JNI array range");
    }
    jbyteArray out = env->NewByteArray(static_cast<jsize>(values.size()));
    if (!out) return nullptr;
    if (!values.empty()) {
        env->SetByteArrayRegion(out, 0, static_cast<jsize>(values.size()),
                reinterpret_cast<const jbyte*>(values.data()));
    }
    return out;
}

void validate_stc_inputs(
        const std::vector<std::uint8_t>& bits,
        const std::vector<double>* costs,
        int message_length,
        int height) {
    if (height != 10 && height != 12) {
        throw std::invalid_argument("CUDA STC supports heights 10 and 12 only");
    }
    if (bits.empty() || message_length <= 0
            || message_length > static_cast<int>(bits.size())
            || message_length < height) {
        throw std::invalid_argument("Invalid CUDA STC cover/message dimensions");
    }
    for (std::uint8_t bit : bits) {
        if (bit != 0u && bit != 1u) {
            throw std::invalid_argument("CUDA STC input contains a non-binary bit");
        }
    }
    if (costs) {
        if (costs->size() != bits.size()) {
            throw std::invalid_argument("CUDA STC cost length mismatch");
        }
        for (double cost : *costs) {
            if (!std::isfinite(cost) || cost < 0.0 || cost > kMaximumCost) {
                throw std::invalid_argument("CUDA STC cost outside [0,1e13]");
            }
        }
    }
}

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_thesis_rap_gpu_CudaStcNative_nativeRuntimeAvailable(JNIEnv*, jclass) {
    return runtime_available() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_thesis_rap_gpu_CudaStcNative_nativeEmbedBinary(
        JNIEnv* env,
        jclass,
        jbyteArray cover_array,
        jdoubleArray costs_array,
        jbyteArray message_array,
        jint height) {
    try {
        auto cover = read_bytes(env, cover_array);
        auto costs = read_doubles(env, costs_array);
        auto message = read_bytes(env, message_array);
        validate_stc_inputs(cover, &costs, static_cast<int>(message.size()), height);
        for (std::uint8_t bit : message) {
            if (bit != 0u && bit != 1u) {
                throw std::invalid_argument("CUDA STC message contains non-binary bit");
            }
        }
        if (!runtime_available()) {
            throw std::runtime_error("CUDA runtime/device unavailable for STC");
        }
        const int device = selected_device();
        if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(device))) return nullptr;
        if (!preflight_cuda(env)) return nullptr;
        const PlanData plan = make_plan(
                static_cast<int>(cover.size()), static_cast<int>(message.size()), height);

        const std::uint32_t state_count = 1u << height;
        const std::size_t path_words_per_carrier =
                (static_cast<std::size_t>(state_count) + 31u) / 32u;
        if (cover.size() > std::numeric_limits<std::size_t>::max()
                / path_words_per_carrier) {
            throw std::overflow_error("CUDA STC path size overflow");
        }
        const std::size_t path_words = cover.size() * path_words_per_carrier;

        auto stream_lease = stc_stream_pool().acquire(env);
        if (env->ExceptionCheck()) return nullptr;
        cudaStream_t stream = stream_lease.stream();
        if (!stream) throw std::runtime_error("Could not acquire CUDA STC stream");

        // All asynchronous H2D/D2H transfers use page-locked host buffers.
        // This is required for true overlap with non-blocking streams on the
        // H100 path; the Java arrays themselves remain ordinary JVM memory.
        auto h_cover = pinned_copy(cover);
        auto h_costs = pinned_copy(costs);
        auto h_message = pinned_copy(message);
        auto h_columns = pinned_copy(plan.carrier_columns);
        auto h_offsets = pinned_copy(plan.message_offsets);
        PinnedBuffer<std::uint8_t> h_stego(cover.size());
        PinnedBuffer<int> h_status(1);
        h_status.data()[0] = 0;

        std::uint8_t *d_cover=nullptr, *d_message=nullptr, *d_stego=nullptr;
        double* d_costs=nullptr;
        std::uint32_t *d_columns=nullptr, *d_path=nullptr;
        int *d_offsets=nullptr, *d_status=nullptr;

        auto cleanup_async = [&]() {
            if (d_cover) cudaFreeAsync(d_cover, stream);
            if (d_costs) cudaFreeAsync(d_costs, stream);
            if (d_message) cudaFreeAsync(d_message, stream);
            if (d_columns) cudaFreeAsync(d_columns, stream);
            if (d_offsets) cudaFreeAsync(d_offsets, stream);
            if (d_path) cudaFreeAsync(d_path, stream);
            if (d_stego) cudaFreeAsync(d_stego, stream);
            if (d_status) cudaFreeAsync(d_status, stream);
        };
        auto fail_cleanup = [&](const char* where, cudaError_t status) -> jbyteArray {
            rap_cuda_crash_guard::observe(status);
            if (!rap_cuda_crash_guard::poisoned()) {
                cleanup_async();
                cudaStreamSynchronize(stream);
            }
            check_cuda(env, where, status);
            return nullptr;
        };

        cudaError_t st = cudaMallocAsync(reinterpret_cast<void**>(&d_cover), cover.size(), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_costs), costs.size()*sizeof(double), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_message), message.size(), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_columns), plan.carrier_columns.size()*sizeof(std::uint32_t), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_offsets), plan.message_offsets.size()*sizeof(int), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_path), path_words*sizeof(std::uint32_t), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_stego), cover.size(), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_status), sizeof(int), stream);
        if (st != cudaSuccess) return fail_cleanup("cudaMallocAsync CUDA STC", st);

        st = cudaMemcpyAsync(d_cover, h_cover.data(), cover.size(), cudaMemcpyHostToDevice, stream);
        if (st == cudaSuccess) st = cudaMemcpyAsync(d_costs, h_costs.data(), costs.size()*sizeof(double), cudaMemcpyHostToDevice, stream);
        if (st == cudaSuccess) st = cudaMemcpyAsync(d_message, h_message.data(), message.size(), cudaMemcpyHostToDevice, stream);
        if (st == cudaSuccess) st = cudaMemcpyAsync(d_columns, h_columns.data(), plan.carrier_columns.size()*sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream);
        if (st == cudaSuccess) st = cudaMemcpyAsync(d_offsets, h_offsets.data(), plan.message_offsets.size()*sizeof(int), cudaMemcpyHostToDevice, stream);
        if (st == cudaSuccess) st = cudaMemsetAsync(d_path, 0, path_words*sizeof(std::uint32_t), stream);
        if (st == cudaSuccess) st = cudaMemsetAsync(d_status, 0, sizeof(int), stream);
        if (st != cudaSuccess) return fail_cleanup("CUDA STC H2D", st);

        const std::size_t shared_bytes = 2u * static_cast<std::size_t>(state_count) * sizeof(double);
        st = cudaFuncSetAttribute(
                stc_embed_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_bytes));
        if (st != cudaSuccess) {
            return fail_cleanup("cudaFuncSetAttribute(STC shared memory)", st);
        }
        stc_embed_kernel<<<1, 256, shared_bytes, stream>>>(
                d_cover, d_costs, static_cast<int>(cover.size()),
                d_message, static_cast<int>(message.size()), height,
                d_columns, d_offsets, d_path, d_stego, d_status);
        st = cudaGetLastError();
        if (st != cudaSuccess) {
            return fail_cleanup("CUDA STC embed launch", st);
        }

        st = cudaMemcpyAsync(h_stego.data(), d_stego, cover.size(), cudaMemcpyDeviceToHost, stream);
        if (st == cudaSuccess) st = cudaMemcpyAsync(h_status.data(), d_status, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cleanup_async();
        if (st == cudaSuccess) st = cudaStreamSynchronize(stream);
        if (!check_cuda(env, "CUDA STC embed completion", st)) return nullptr;
        const int status = h_status.data()[0];
        if (status == 4) {
            throw_java(env, "thesis/rap/stc/NoFiniteStcSolutionException",
                    "No finite CUDA STC solution exists.");
            return nullptr;
        }
        if (status != 0) {
            throw std::runtime_error("CUDA STC backtracking invariant failed");
        }
        std::vector<std::uint8_t> stego(cover.size());
        std::copy(h_stego.data(), h_stego.data() + h_stego.size(), stego.begin());
        return to_java_bytes(env, stego);
    } catch (const stc_exception& e) {
        const char* klass = e.error_id == 4
                ? "thesis/rap/stc/NoFiniteStcSolutionException"
                : e.error_id == 2 ? "java/lang/OutOfMemoryError"
                : "java/lang/IllegalArgumentException";
        throw_java(env, klass, e.what());
        return nullptr;
    } catch (const std::bad_alloc& e) {
        throw_java(env, "java/lang/OutOfMemoryError", e.what());
        return nullptr;
    } catch (const std::exception& e) {
        throw_java(env, "java/lang/IllegalStateException", e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_thesis_rap_gpu_CudaStcNative_nativeExtractBinary(
        JNIEnv* env,
        jclass,
        jbyteArray stego_array,
        jint message_length,
        jint height) {
    try {
        auto stego = read_bytes(env, stego_array);
        validate_stc_inputs(stego, nullptr, message_length, height);
        if (!runtime_available()) {
            throw std::runtime_error("CUDA runtime/device unavailable for STC extraction");
        }
        const int device = selected_device();
        if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(device))) return nullptr;
        if (!preflight_cuda(env)) return nullptr;
        const PlanData plan = make_plan(
                static_cast<int>(stego.size()), message_length, height);

        std::vector<int> carrier_message_index(stego.size());
        for (int j = 0; j < message_length; ++j) {
            for (int i = plan.message_offsets[static_cast<std::size_t>(j)];
                    i < plan.message_offsets[static_cast<std::size_t>(j)+1u]; ++i) {
                carrier_message_index[static_cast<std::size_t>(i)] = j;
            }
        }

        auto stream_lease = stc_stream_pool().acquire(env);
        if (env->ExceptionCheck()) return nullptr;
        cudaStream_t stream = stream_lease.stream();
        if (!stream) throw std::runtime_error("Could not acquire CUDA STC stream");
        auto h_stego = pinned_copy(stego);
        auto h_columns = pinned_copy(plan.carrier_columns);
        auto h_carrier_message = pinned_copy(carrier_message_index);
        PinnedBuffer<std::uint32_t> h_syndrome(static_cast<std::size_t>(message_length));

        std::uint8_t* d_stego=nullptr;
        std::uint32_t *d_columns=nullptr, *d_syndrome=nullptr;
        int* d_carrier_message=nullptr;
        auto cleanup_async = [&]() {
            if (d_stego) cudaFreeAsync(d_stego, stream);
            if (d_columns) cudaFreeAsync(d_columns, stream);
            if (d_carrier_message) cudaFreeAsync(d_carrier_message, stream);
            if (d_syndrome) cudaFreeAsync(d_syndrome, stream);
        };
        auto fail_cleanup = [&](const char* where, cudaError_t status) -> jbyteArray {
            rap_cuda_crash_guard::observe(status);
            if (!rap_cuda_crash_guard::poisoned()) {
                cleanup_async();
                cudaStreamSynchronize(stream);
            }
            check_cuda(env, where, status);
            return nullptr;
        };

        const std::size_t syndrome_count = static_cast<std::size_t>(message_length) + static_cast<std::size_t>(height);
        cudaError_t st = cudaMallocAsync(reinterpret_cast<void**>(&d_stego), stego.size(), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_columns), plan.carrier_columns.size()*sizeof(std::uint32_t), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_carrier_message), carrier_message_index.size()*sizeof(int), stream);
        if (st == cudaSuccess) st = cudaMallocAsync(reinterpret_cast<void**>(&d_syndrome), syndrome_count*sizeof(std::uint32_t), stream);
        if (st != cudaSuccess) return fail_cleanup("cudaMallocAsync CUDA STC extract", st);
        st = cudaMemcpyAsync(d_stego, h_stego.data(), stego.size(), cudaMemcpyHostToDevice, stream);
        if (st == cudaSuccess) st = cudaMemcpyAsync(d_columns, h_columns.data(), plan.carrier_columns.size()*sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream);
        if (st == cudaSuccess) st = cudaMemcpyAsync(d_carrier_message, h_carrier_message.data(), carrier_message_index.size()*sizeof(int), cudaMemcpyHostToDevice, stream);
        if (st == cudaSuccess) st = cudaMemsetAsync(d_syndrome, 0, syndrome_count*sizeof(std::uint32_t), stream);
        if (st != cudaSuccess) return fail_cleanup("CUDA STC extract H2D", st);

        constexpr int threads = 256;
        const int blocks = (static_cast<int>(stego.size()) + threads - 1) / threads;
        stc_extract_kernel<<<blocks, threads, 0, stream>>>(
                d_stego, static_cast<int>(stego.size()), message_length, height,
                d_columns, d_carrier_message, d_syndrome);
        st = cudaGetLastError();
        if (st != cudaSuccess) {
            return fail_cleanup("CUDA STC extract launch", st);
        }
        st = cudaMemcpyAsync(h_syndrome.data(), d_syndrome,
                static_cast<std::size_t>(message_length)*sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost, stream);
        cleanup_async();
        if (st == cudaSuccess) st = cudaStreamSynchronize(stream);
        if (!check_cuda(env, "CUDA STC extract completion", st)) return nullptr;

        std::vector<std::uint8_t> message(static_cast<std::size_t>(message_length));
        for (int i = 0; i < message_length; ++i) {
            message[static_cast<std::size_t>(i)] =
                    static_cast<std::uint8_t>(h_syndrome.data()[static_cast<std::size_t>(i)] & 1u);
        }
        return to_java_bytes(env, message);
    } catch (const stc_exception& e) {
        const char* klass = e.error_id == 2
                ? "java/lang/OutOfMemoryError"
                : "java/lang/IllegalArgumentException";
        throw_java(env, klass, e.what());
        return nullptr;
    } catch (const std::bad_alloc& e) {
        throw_java(env, "java/lang/OutOfMemoryError", e.what());
        return nullptr;
    } catch (const std::exception& e) {
        throw_java(env, "java/lang/IllegalStateException", e.what());
        return nullptr;
    }
}
