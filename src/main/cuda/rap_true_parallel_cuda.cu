#include <jni.h>
#include <cuda_runtime.h>
#include "rap_cuda_crash_guard.cuh"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int kCandidateFields = 8;
constexpr int kDigestBytes = 32;

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
                "CUDA true-parallel stage refused because CUDA context is already poisoned (first code="
                + std::to_string(rap_cuda_crash_guard::first_error_code()) + ")");
        return false;
    }
    return check_cuda(env, "CUDA true-parallel preflight pending error", cudaGetLastError());
}

void safe_free_host(void* pointer) noexcept {
    if (pointer != nullptr && !rap_cuda_crash_guard::poisoned()) {
        (void)cudaFreeHost(pointer);
    }
}

int parse_env_int(const char* name, int fallback, int minimum, int maximum) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < minimum || value > maximum) return fallback;
    return static_cast<int>(value);
}

int selected_device() {
    return parse_env_int("RAP_CUDA_DEVICE", 0, 0, 1024);
}

class StreamPool {
public:
    class Lease {
    public:
        Lease() = default;
        Lease(StreamPool* owner, std::size_t index)
                : owner_(owner), index_(index) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept
                : owner_(other.owner_), index_(other.index_) { other.owner_ = nullptr; }
        ~Lease() { if (owner_) owner_->release(index_); }
        cudaStream_t stream() const { return owner_ ? owner_->streams_[index_] : nullptr; }
    private:
        StreamPool* owner_ = nullptr;
        std::size_t index_ = 0;
    };

    Lease acquire(JNIEnv* env) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (rap_cuda_crash_guard::poisoned()) {
            throw_java(env, "java/lang/IllegalStateException",
                    "CUDA stream acquisition refused: CUDA context is poisoned");
            return Lease();
        }
        initialize_locked(env);
        if (env->ExceptionCheck()) return Lease();
        condition_.wait(lock, [&]() {
            return rap_cuda_crash_guard::poisoned()
                    || std::any_of(busy_.begin(), busy_.end(), [](bool b) { return !b; });
        });
        if (rap_cuda_crash_guard::poisoned()) {
            throw_java(env, "java/lang/IllegalStateException",
                    "CUDA stream wait aborted: CUDA context is poisoned");
            return Lease();
        }
        for (std::size_t i = 0; i < busy_.size(); ++i) {
            if (!busy_[i]) {
                busy_[i] = true;
                return Lease(this, i);
            }
        }
        throw_java(env, "java/lang/IllegalStateException", "CUDA stream-pool invariant failed");
        return Lease();
    }

    int size(JNIEnv* env) {
        std::lock_guard<std::mutex> lock(mutex_);
        initialize_locked(env);
        return static_cast<int>(streams_.size());
    }

    ~StreamPool() {
        if (rap_cuda_crash_guard::poisoned()) return;
        for (cudaStream_t stream : streams_) {
            if (stream) cudaStreamDestroy(stream);
        }
    }

private:
    void initialize_locked(JNIEnv* env) {
        if (initialized_) return;
        int count = parse_env_int("RAP_CUDA_STREAMS", 8, 1, 64);
        streams_.resize(static_cast<std::size_t>(count), nullptr);
        busy_.assign(static_cast<std::size_t>(count), false);
        for (int i = 0; i < count; ++i) {
            cudaError_t status = cudaStreamCreateWithFlags(
                    &streams_[static_cast<std::size_t>(i)], cudaStreamNonBlocking);
            if (status != cudaSuccess) {
                throw_java(env, "java/lang/IllegalStateException",
                        cuda_error("cudaStreamCreateWithFlags", status));
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

StreamPool& stream_pool() {
    static StreamPool pool;
    return pool;
}

// ---- SHA-256 / HMAC-SHA-256 device implementation -------------------------
__device__ __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

struct Sha256Ctx {
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
};

__device__ void sha256_transform(Sha256Ctx* ctx, const uint8_t data[64]) {
    uint32_t m[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int j = i * 4;
        m[i] = (static_cast<uint32_t>(data[j]) << 24)
             | (static_cast<uint32_t>(data[j+1]) << 16)
             | (static_cast<uint32_t>(data[j+2]) << 8)
             | static_cast<uint32_t>(data[j+3]);
    }
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(m[i-15],7) ^ rotr(m[i-15],18) ^ (m[i-15] >> 3);
        uint32_t s1 = rotr(m[i-2],17) ^ rotr(m[i-2],19) ^ (m[i-2] >> 10);
        m[i] = m[i-16] + s0 + m[i-7] + s1;
    }
    uint32_t a=ctx->state[0], b=ctx->state[1], c=ctx->state[2], d=ctx->state[3];
    uint32_t e=ctx->state[4], f=ctx->state[5], g=ctx->state[6], h=ctx->state[7];
    #pragma unroll 8
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + SHA256_K[i] + m[i];
        uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h=g; g=f; f=e; e=d + temp1;
        d=c; c=b; b=a; a=temp1 + temp2;
    }
    ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d;
    ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}

__device__ void sha256_init(Sha256Ctx* ctx) {
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0]=0x6a09e667u; ctx->state[1]=0xbb67ae85u;
    ctx->state[2]=0x3c6ef372u; ctx->state[3]=0xa54ff53au;
    ctx->state[4]=0x510e527fu; ctx->state[5]=0x9b05688cu;
    ctx->state[6]=0x1f83d9abu; ctx->state[7]=0x5be0cd19u;
}

__device__ void sha256_update(Sha256Ctx* ctx, const uint8_t* data, int len) {
    for (int i = 0; i < len; ++i) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void sha256_final(Sha256Ctx* ctx, uint8_t hash[32]) {
    uint32_t i = ctx->datalen;
    ctx->data[i++] = 0x80;
    if (i > 56) {
        while (i < 64) ctx->data[i++] = 0;
        sha256_transform(ctx, ctx->data);
        i = 0;
    }
    while (i < 56) ctx->data[i++] = 0;
    ctx->bitlen += static_cast<uint64_t>(ctx->datalen) * 8u;
    for (int shift = 7; shift >= 0; --shift) {
        ctx->data[56 + (7-shift)] = static_cast<uint8_t>((ctx->bitlen >> (shift*8)) & 0xffu);
    }
    sha256_transform(ctx, ctx->data);
    for (int word = 0; word < 8; ++word) {
        hash[word*4]   = static_cast<uint8_t>((ctx->state[word] >> 24) & 0xffu);
        hash[word*4+1] = static_cast<uint8_t>((ctx->state[word] >> 16) & 0xffu);
        hash[word*4+2] = static_cast<uint8_t>((ctx->state[word] >> 8) & 0xffu);
        hash[word*4+3] = static_cast<uint8_t>(ctx->state[word] & 0xffu);
    }
}

__device__ void put_be32(uint8_t* out, uint32_t value) {
    out[0] = static_cast<uint8_t>((value >> 24) & 0xffu);
    out[1] = static_cast<uint8_t>((value >> 16) & 0xffu);
    out[2] = static_cast<uint8_t>((value >> 8) & 0xffu);
    out[3] = static_cast<uint8_t>(value & 0xffu);
}

__global__ void hmac_candidates_kernel(
        const uint8_t* key32,
        const uint8_t* domain,
        int domain_len,
        int32_t qf,
        int32_t geometry_signature,
        const int32_t* fields,
        int candidate_count,
        uint8_t* digests) {
    int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= candidate_count) return;

    uint8_t ipad[64];
    uint8_t opad[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint8_t key_byte = i < 32 ? key32[i] : 0;
        ipad[i] = static_cast<uint8_t>(key_byte ^ 0x36u);
        opad[i] = static_cast<uint8_t>(key_byte ^ 0x5cu);
    }

    uint8_t serialized[40];
    put_be32(serialized, static_cast<uint32_t>(qf));
    put_be32(serialized + 4, static_cast<uint32_t>(geometry_signature));
    const int32_t* source = fields + static_cast<std::size_t>(index) * kCandidateFields;
    #pragma unroll
    for (int f = 0; f < kCandidateFields; ++f) {
        put_be32(serialized + 8 + f*4, static_cast<uint32_t>(source[f]));
    }

    uint8_t inner_hash[32];
    Sha256Ctx inner;
    sha256_init(&inner);
    sha256_update(&inner, ipad, 64);
    sha256_update(&inner, domain, domain_len);
    sha256_update(&inner, serialized, 40);
    sha256_final(&inner, inner_hash);

    Sha256Ctx outer;
    sha256_init(&outer);
    sha256_update(&outer, opad, 64);
    sha256_update(&outer, inner_hash, 32);
    sha256_final(&outer, digests + static_cast<std::size_t>(index) * kDigestBytes);
}


// ---- QF-independent forward DCT + block variance --------------------------
__global__ void dct_variance_kernel(
        const double* plane,
        int width,
        int height,
        int width_blocks,
        int block_count,
        const double* cosine,
        const double* alpha,
        double* dct,
        double* variances) {
    int block = static_cast<int>(blockIdx.x);
    int k = static_cast<int>(threadIdx.x);
    if (block >= block_count || k >= 64) return;
    int bx = block % width_blocks;
    int by = block / width_blocks;
    int u = k & 7;
    int v = k >> 3;

    double sum = 0.0;
    for (int y = 0; y < 8; ++y) {
        int py = by * 8 + y;
        if (py >= height) py = height - 1;
        for (int x = 0; x < 8; ++x) {
            int px = bx * 8 + x;
            if (px >= width) px = width - 1;
            double shifted = plane[static_cast<std::size_t>(py) * width + px] - 128.0;
            double term = shifted * cosine[x * 8 + u];
            term *= cosine[y * 8 + v];
            sum += term;
        }
    }
    double scale = 0.25 * alpha[u];
    scale *= alpha[v];
    dct[static_cast<std::size_t>(block) * 64u + static_cast<std::size_t>(k)] = sum * scale;

    if (k == 0) {
        double mean = 0.0;
        for (int y = 0; y < 8; ++y) {
            int py = by * 8 + y;
            if (py >= height) py = height - 1;
            for (int x = 0; x < 8; ++x) {
                int px = bx * 8 + x;
                if (px >= width) px = width - 1;
                mean += plane[static_cast<std::size_t>(py) * width + px];
            }
        }
        mean /= 64.0;
        double variance = 0.0;
        for (int y = 0; y < 8; ++y) {
            int py = by * 8 + y;
            if (py >= height) py = height - 1;
            for (int x = 0; x < 8; ++x) {
                int px = bx * 8 + x;
                if (px >= width) px = width - 1;
                double centered = plane[static_cast<std::size_t>(py) * width + px] - mean;
                variance += centered * centered;
            }
        }
        variances[block] = variance / 64.0;
    }
}



// ---- GPU-max image preprocessing -----------------------------------------
__global__ void rgb_to_ycbcr_kernel(
        const int32_t* rgb,
        int pixel_count,
        double* planes) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= pixel_count) return;
    uint32_t packed = static_cast<uint32_t>(rgb[i]);
    double r = static_cast<double>((packed >> 16) & 255u);
    double g = static_cast<double>((packed >> 8) & 255u);
    double b = static_cast<double>(packed & 255u);

    // CUDA build uses --fmad=false and precise division/sqrt flags.
    double y = 0.299 * r;
    y += 0.587 * g;
    y += 0.114 * b;
    double cb = 128.0;
    cb -= 0.168736 * r;
    cb -= 0.331264 * g;
    cb += 0.5 * b;
    double cr = 128.0;
    cr += 0.5 * r;
    cr -= 0.418688 * g;
    cr -= 0.081312 * b;

    planes[i] = y;
    planes[static_cast<std::size_t>(pixel_count) + i] = cb;
    planes[static_cast<std::size_t>(pixel_count) * 2u + i] = cr;
}

__global__ void resample_and_pad_kernel(
        const double* source,
        int source_width,
        int source_height,
        int ratio_x,
        int ratio_y,
        int output_width,
        int output_height,
        int padded_width,
        int padded_height,
        double* output) {
    int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int count = padded_width * padded_height;
    if (linear >= count) return;
    int x = linear % padded_width;
    int y = linear / padded_width;
    int ox = x < output_width ? x : output_width - 1;
    int oy = y < output_height ? y : output_height - 1;

    double sum = 0.0;
    for (int dy = 0; dy < ratio_y; ++dy) {
        int sy = oy * ratio_y + dy;
        if (sy >= source_height) sy = source_height - 1;
        for (int dx = 0; dx < ratio_x; ++dx) {
            int sx = ox * ratio_x + dx;
            if (sx >= source_width) sx = source_width - 1;
            sum += source[static_cast<std::size_t>(sy) * source_width + sx];
        }
    }
    output[linear] = sum / (static_cast<double>(ratio_x) * ratio_y);
}

struct CandidateOrderKey {
    double cost;
    int component;
    int block;
    int k;
    int index;
};

struct CandidateOrderLess {
    __host__ __device__ bool operator()(
            const CandidateOrderKey& a,
            const CandidateOrderKey& b) const {
        if (a.cost < b.cost) return true;
        if (a.cost > b.cost) return false;
        if (a.component < b.component) return true;
        if (a.component > b.component) return false;
        if (a.block < b.block) return true;
        if (a.block > b.block) return false;
        if (a.k < b.k) return true;
        if (a.k > b.k) return false;
        return a.index < b.index;
    }
};

__global__ void build_candidate_order_keys_kernel(
        const double* costs,
        const int32_t* fields,
        int count,
        CandidateOrderKey* keys) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    CandidateOrderKey key;
    key.cost = costs[i];
    key.component = fields[static_cast<std::size_t>(i) * 3u];
    key.block = fields[static_cast<std::size_t>(i) * 3u + 1u];
    key.k = fields[static_cast<std::size_t>(i) * 3u + 2u];
    key.index = i;
    keys[i] = key;
}



__device__ __forceinline__ double balance_adjusted(
        double base,
        double minimum,
        int zero_flow,
        double multiplier,
        double wet_cost) {
    if (!isfinite(base) || base >= wet_cost) return wet_cost;
    double adjusted = base + multiplier * static_cast<double>(zero_flow);
    if (!isfinite(adjusted)) return wet_cost;
    double legal_cap = wet_cost - 1.0;
    if (adjusted < minimum) adjusted = minimum;
    if (adjusted > legal_cap) adjusted = legal_cap;
    return adjusted;
}

__global__ void balance_plan_kernel(
        const double* rho_plus,
        const double* rho_minus,
        const double* minimum_plus,
        const double* minimum_minus,
        const int32_t* zero_flow_plus,
        const int32_t* zero_flow_minus,
        const int32_t* tie_directions,
        int count,
        double multiplier,
        double wet_cost,
        double* output) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    double plus = balance_adjusted(
            rho_plus[i], minimum_plus[i], zero_flow_plus[i],
            multiplier, wet_cost);
    double minus = balance_adjusted(
            rho_minus[i], minimum_minus[i], zero_flow_minus[i],
            multiplier, wet_cost);
    bool can_plus = isfinite(plus) && plus < wet_cost;
    bool can_minus = isfinite(minus) && minus < wet_cost;
    int direction;
    if (can_minus && can_plus) {
        if (minus < plus) direction = -1;
        else if (minus > plus) direction = 1;
        else direction = tie_directions[i];
    } else if (can_minus) {
        direction = -1;
    } else {
        direction = 1;
    }
    output[i] = plus;
    output[static_cast<std::size_t>(count) + i] = minus;
    output[static_cast<std::size_t>(count) * 2u + i] = plus < minus ? plus : minus;
    output[static_cast<std::size_t>(count) * 3u + i] = static_cast<double>(direction);
}



__global__ void canonical_luma_histogram_kernel(
        const int32_t* coefficients,
        const double* basis,
        int width_blocks,
        int width,
        int height,
        double* luminance,
        unsigned int* histogram) {
    int pixel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int pixel_count = width * height;
    if (pixel >= pixel_count) return;
    int x = pixel % width;
    int y = pixel / width;
    int bx = x >> 3;
    int by = y >> 3;
    int local_x = x & 7;
    int local_y = y & 7;
    int local = local_y * 8 + local_x;
    int block = by * width_blocks + bx;
    double sample = 128.0;
    const int32_t* block_coeff = coefficients + static_cast<std::size_t>(block) * 64u;
    for (int k = 0; k < 64; ++k) {
        double term = static_cast<double>(block_coeff[k])
                * basis[static_cast<std::size_t>(k) * 64u + local];
        sample += term;
    }
    if (sample < 0.0) sample = 0.0;
    if (sample > 255.0) sample = 255.0;
    luminance[pixel] = sample;

    // sum first, so 0.5 - 2^-54 lands in bin 1 on the GPU and bin 0 on the CPU,
    // which changes a histogram weight and therefore the frozen carrier order.
    // sample - whole is exact for sample in [0, 256).
    const double whole = floor(sample);
    int bin = static_cast<int>(whole) + ((sample - whole) >= 0.5 ? 1 : 0);
    if (bin < 0) bin = 0;
    if (bin > 255) bin = 255;
    atomicAdd(histogram + bin, 1u);
}

__global__ void normalize_histogram_kernel(
        const unsigned int* histogram,
        int pixel_count,
        double* normalized) {
    int bin = static_cast<int>(threadIdx.x);
    if (bin < 256) {
        normalized[bin] = static_cast<double>(histogram[bin])
                / static_cast<double>(pixel_count);
    }
}

bool runtime_available() {
    if (rap_cuda_crash_guard::poisoned()) return false;
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) rap_cuda_crash_guard::observe(status);
    return status == cudaSuccess && count > selected_device();
}

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_thesis_rap_gpu_CudaTrueParallelNative_nativeRuntimeAvailable(JNIEnv*, jclass) {
    return runtime_available() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jint JNICALL
Java_thesis_rap_gpu_CudaTrueParallelNative_nativeStreamPoolSize(JNIEnv* env, jclass) {
    if (!runtime_available()) return 0;
    cudaSetDevice(selected_device());
    return stream_pool().size(env);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_thesis_rap_gpu_CudaTrueParallelNative_nativeHmacCandidateDigests(
        JNIEnv* env, jclass,
        jbyteArray key_array,
        jbyteArray domain_array,
        jint qf,
        jint geometry_signature,
        jintArray fields_array) {
    if (!key_array || !domain_array || !fields_array) {
        throw_java(env, "java/lang/IllegalArgumentException", "HMAC CUDA inputs cannot be null");
        return nullptr;
    }
    jsize key_len = env->GetArrayLength(key_array);
    jsize domain_len = env->GetArrayLength(domain_array);
    jsize field_len = env->GetArrayLength(fields_array);
    if (key_len != 32 || domain_len <= 0 || domain_len > 64
            || field_len % kCandidateFields != 0) {
        throw_java(env, "java/lang/IllegalArgumentException", "Invalid CUDA HMAC input dimensions");
        return nullptr;
    }
    int candidate_count = field_len / kCandidateFields;
    if (candidate_count == 0) return env->NewByteArray(0);
    std::size_t digest_bytes = static_cast<std::size_t>(candidate_count) * kDigestBytes;
    if (digest_bytes > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
        throw_java(env, "java/lang/IllegalArgumentException", "HMAC digest output exceeds Java array limit");
        return nullptr;
    }

    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(selected_device()))) return nullptr;
    if (!preflight_cuda(env)) return nullptr;
    auto lease = stream_pool().acquire(env);
    if (env->ExceptionCheck()) return nullptr;
    cudaStream_t stream = lease.stream();

    uint8_t *h_key=nullptr, *h_domain=nullptr, *h_output=nullptr;
    int32_t* h_fields=nullptr;
    uint8_t *d_key=nullptr, *d_domain=nullptr, *d_digests=nullptr;
    int32_t* d_fields=nullptr;
    const std::size_t field_bytes = static_cast<std::size_t>(field_len) * sizeof(int32_t);
    cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&h_key), 32, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_domain), domain_len, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_fields), field_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_output), digest_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_key), 32, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_domain), domain_len, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_fields), field_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_digests), digest_bytes, stream);
    if (!check_cuda(env, "CUDA HMAC pinned/device allocation", status)) {
        if (h_key) safe_free_host(h_key); if (h_domain) safe_free_host(h_domain);
        if (h_fields) safe_free_host(h_fields); if (h_output) safe_free_host(h_output);
        return nullptr;
    }

    env->GetByteArrayRegion(key_array, 0, 32, reinterpret_cast<jbyte*>(h_key));
    env->GetByteArrayRegion(domain_array, 0, domain_len, reinterpret_cast<jbyte*>(h_domain));
    env->GetIntArrayRegion(fields_array, 0, field_len, reinterpret_cast<jint*>(h_fields));
    if (env->ExceptionCheck()) {
        cudaFreeAsync(d_key, stream); cudaFreeAsync(d_domain, stream);
        cudaFreeAsync(d_fields, stream); cudaFreeAsync(d_digests, stream);
        safe_free_host(h_key); safe_free_host(h_domain); safe_free_host(h_fields); safe_free_host(h_output);
        return nullptr;
    }

    status = cudaMemcpyAsync(d_key, h_key, 32, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_domain, h_domain, domain_len, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_fields, h_fields, field_bytes, cudaMemcpyHostToDevice, stream);
    if (!check_cuda(env, "cudaMemcpyAsync HMAC input", status)) {
        safe_free_host(h_key); safe_free_host(h_domain); safe_free_host(h_fields); safe_free_host(h_output);
        return nullptr;
    }

    int threads = 256;
    int blocks = (candidate_count + threads - 1) / threads;
    hmac_candidates_kernel<<<blocks, threads, 0, stream>>>(
            d_key, d_domain, domain_len, qf, geometry_signature,
            d_fields, candidate_count, d_digests);
    status = cudaGetLastError();
    if (!check_cuda(env, "CUDA HMAC kernel launch", status)) {
        safe_free_host(h_key); safe_free_host(h_domain); safe_free_host(h_fields); safe_free_host(h_output);
        return nullptr;
    }

    status = cudaMemcpyAsync(h_output, d_digests, digest_bytes,
            cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_key, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_domain, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_fields, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_digests, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (!check_cuda(env, "CUDA HMAC completion", status)) {
        safe_free_host(h_key); safe_free_host(h_domain); safe_free_host(h_fields); safe_free_host(h_output);
        return nullptr;
    }

    jbyteArray result = env->NewByteArray(static_cast<jsize>(digest_bytes));
    if (result) env->SetByteArrayRegion(result, 0, static_cast<jsize>(digest_bytes),
            reinterpret_cast<const jbyte*>(h_output));
    safe_free_host(h_key); safe_free_host(h_domain); safe_free_host(h_fields); safe_free_host(h_output);
    return env->ExceptionCheck() ? nullptr : result;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_thesis_rap_gpu_CudaTrueParallelNative_nativePercentile95(
        JNIEnv* env, jclass, jdoubleArray values_array) {
    if (!values_array) {
        throw_java(env, "java/lang/IllegalArgumentException", "P95 input cannot be null");
        return 0.0;
    }
    jsize count = env->GetArrayLength(values_array);
    if (count <= 0) return 0.0;
    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(selected_device()))) return 0.0;
    if (!preflight_cuda(env)) return 0.0;
    auto lease = stream_pool().acquire(env);
    if (env->ExceptionCheck()) return 0.0;
    cudaStream_t stream = lease.stream();
    std::size_t bytes = static_cast<std::size_t>(count) * sizeof(double);
    double *h_values=nullptr, *h_result=nullptr, *d_values=nullptr;
    cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&h_values), bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_result), sizeof(double), cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_values), bytes, stream);
    if (!check_cuda(env, "CUDA P95 allocation", status)) {
        if (h_values) safe_free_host(h_values); if (h_result) safe_free_host(h_result); return 0.0;
    }
    env->GetDoubleArrayRegion(values_array, 0, count, h_values);
    if (env->ExceptionCheck()) { cudaFreeAsync(d_values, stream); safe_free_host(h_values); safe_free_host(h_result); return 0.0; }
    for (jsize i = 0; i < count; ++i) {
        if (!std::isfinite(h_values[i]) || h_values[i] < 0.0) {
            cudaFreeAsync(d_values, stream); safe_free_host(h_values); safe_free_host(h_result);
            throw_java(env, "java/lang/IllegalArgumentException",
                       "nativePercentile95 requires prefiltered finite nonnegative values");
            return 0.0;
        }
    }
    status = cudaMemcpyAsync(d_values, h_values, bytes, cudaMemcpyHostToDevice, stream);
    if (!check_cuda(env, "cudaMemcpyAsync P95 input", status)) { safe_free_host(h_values); safe_free_host(h_result); return 0.0; }
    try {
        thrust::sort(thrust::cuda::par.on(stream), d_values, d_values + count);
        cudaError_t sort_status = cudaStreamSynchronize(stream);
        if (sort_status != cudaSuccess) {
            rap_cuda_crash_guard::observe(sort_status);
            throw std::runtime_error(cuda_error("CUDA Thrust sort synchronization", sort_status));
        }
    } catch (const std::exception& error) {
        if (!rap_cuda_crash_guard::poisoned()) cudaFreeAsync(d_values, stream);
        if (!rap_cuda_crash_guard::poisoned()) { safe_free_host(h_values); safe_free_host(h_result); }
        throw_java(env, "java/lang/IllegalStateException",
                   std::string("Thrust P95 sort failed: ") + error.what());
        return 0.0;
    } catch (...) {
        throw_java(env, "java/lang/IllegalStateException",
                   "Thrust P95 sort failed with a non-standard native exception");
        return 0.0;
    }
    int index = std::max(0, std::min(static_cast<int>(count) - 1,
            static_cast<int>(std::ceil(0.95 * static_cast<double>(count))) - 1));
    status = cudaMemcpyAsync(h_result, d_values + index, sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_values, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    double result = status == cudaSuccess ? *h_result : 0.0;
    safe_free_host(h_values); safe_free_host(h_result);
    if (!check_cuda(env, "CUDA P95 completion", status)) return 0.0;
    return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_thesis_rap_gpu_CudaDctNative_nativeRuntimeAvailable(JNIEnv*, jclass) {
    return runtime_available() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_thesis_rap_gpu_CudaDctNative_nativeDctAndVariance(
        JNIEnv* env, jclass,
        jdoubleArray plane_array,
        jint width,
        jint height,
        jint width_blocks,
        jint height_blocks,
        jdoubleArray cosine_array,
        jdoubleArray alpha_array) {
    if (!plane_array || !cosine_array || !alpha_array
            || width <= 0 || height <= 0
            || width_blocks <= 0 || height_blocks <= 0
            || width != width_blocks * 8
            || height != height_blocks * 8) {
        throw_java(env, "java/lang/IllegalArgumentException",
                "Invalid CUDA DCT dimensions");
        return nullptr;
    }
    const jsize plane_len = env->GetArrayLength(plane_array);
    const jsize cosine_len = env->GetArrayLength(cosine_array);
    const jsize alpha_len = env->GetArrayLength(alpha_array);
    const int block_count = width_blocks * height_blocks;
    const std::size_t dct_count = static_cast<std::size_t>(block_count) * 64u;
    if (plane_len != width * height || cosine_len != 64 || alpha_len != 8
            || dct_count + static_cast<std::size_t>(block_count)
                    > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
        throw_java(env, "java/lang/IllegalArgumentException",
                "Invalid CUDA DCT array dimensions");
        return nullptr;
    }
    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(selected_device()))) {
        return nullptr;
    }
    if (!preflight_cuda(env)) return nullptr;
    auto lease = stream_pool().acquire(env);
    if (env->ExceptionCheck()) return nullptr;
    cudaStream_t stream = lease.stream();

    const std::size_t plane_bytes = static_cast<std::size_t>(plane_len) * sizeof(double);
    const std::size_t dct_bytes = dct_count * sizeof(double);
    const std::size_t variance_bytes = static_cast<std::size_t>(block_count) * sizeof(double);
    const std::size_t combined_count = dct_count + static_cast<std::size_t>(block_count);
    const std::size_t combined_bytes = combined_count * sizeof(double);

    double *h_plane=nullptr,*h_cosine=nullptr,*h_alpha=nullptr,*h_combined=nullptr;
    double *d_plane=nullptr,*d_cosine=nullptr,*d_alpha=nullptr,*d_dct=nullptr,*d_variance=nullptr;
    cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&h_plane), plane_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_cosine), 64*sizeof(double), cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_alpha), 8*sizeof(double), cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_combined), combined_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_plane), plane_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_cosine), 64*sizeof(double), stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_alpha), 8*sizeof(double), stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_dct), dct_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_variance), variance_bytes, stream);
    if (!check_cuda(env, "CUDA DCT pinned/device allocation", status)) return nullptr;

    env->GetDoubleArrayRegion(plane_array, 0, plane_len, h_plane);
    env->GetDoubleArrayRegion(cosine_array, 0, 64, h_cosine);
    env->GetDoubleArrayRegion(alpha_array, 0, 8, h_alpha);
    if (env->ExceptionCheck()) return nullptr;

    status = cudaMemcpyAsync(d_plane, h_plane, plane_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_cosine, h_cosine, 64*sizeof(double), cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_alpha, h_alpha, 8*sizeof(double), cudaMemcpyHostToDevice, stream);
    if (!check_cuda(env, "cudaMemcpyAsync DCT input", status)) return nullptr;

    dct_variance_kernel<<<block_count, 64, 0, stream>>>(
            d_plane, width, height, width_blocks, block_count,
            d_cosine, d_alpha, d_dct, d_variance);
    status = cudaGetLastError();
    if (!check_cuda(env, "CUDA DCT kernel launch", status)) return nullptr;

    status = cudaMemcpyAsync(h_combined, d_dct, dct_bytes, cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(
            h_combined + dct_count, d_variance,
            variance_bytes, cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_plane, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_cosine, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_alpha, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_dct, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_variance, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (!check_cuda(env, "CUDA DCT completion", status)) return nullptr;

    jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(combined_count));
    if (result) env->SetDoubleArrayRegion(
            result, 0, static_cast<jsize>(combined_count), h_combined);
    safe_free_host(h_plane); safe_free_host(h_cosine); safe_free_host(h_alpha); safe_free_host(h_combined);
    return env->ExceptionCheck() ? nullptr : result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_thesis_rap_gpu_CudaImagePreprocessNative_nativeRuntimeAvailable(JNIEnv*, jclass) {
    return runtime_available() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_thesis_rap_gpu_CudaImagePreprocessNative_nativeRgbToYCbCr(
        JNIEnv* env, jclass, jintArray rgb_array, jint width, jint height) {
    if (!rgb_array || width <= 0 || height <= 0) {
        throw_java(env, "java/lang/IllegalArgumentException", "Invalid RGB->YCbCr inputs");
        return nullptr;
    }
    const jsize count = env->GetArrayLength(rgb_array);
    if (count != width * height) {
        throw_java(env, "java/lang/IllegalArgumentException", "RGB raster length mismatch");
        return nullptr;
    }
    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(selected_device()))) return nullptr;
    if (!preflight_cuda(env)) return nullptr;
    auto lease = stream_pool().acquire(env);
    if (env->ExceptionCheck()) return nullptr;
    cudaStream_t stream = lease.stream();

    int32_t* h_rgb = nullptr;
    double* h_planes = nullptr;
    int32_t* d_rgb = nullptr;
    double* d_planes = nullptr;
    const std::size_t rgb_bytes = static_cast<std::size_t>(count) * sizeof(int32_t);
    const std::size_t plane_count = static_cast<std::size_t>(count) * 3u;
    const std::size_t plane_bytes = plane_count * sizeof(double);
    cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&h_rgb), rgb_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_planes), plane_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_rgb), rgb_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_planes), plane_bytes, stream);
    if (!check_cuda(env, "CUDA RGB->YCbCr allocation", status)) {
        if (h_rgb) safe_free_host(h_rgb); if (h_planes) safe_free_host(h_planes);
        return nullptr;
    }
    env->GetIntArrayRegion(rgb_array, 0, count, reinterpret_cast<jint*>(h_rgb));
    if (env->ExceptionCheck()) {
        cudaFreeAsync(d_rgb, stream); cudaFreeAsync(d_planes, stream);
        safe_free_host(h_rgb); safe_free_host(h_planes); return nullptr;
    }
    status = cudaMemcpyAsync(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice, stream);
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    if (status == cudaSuccess) {
        rgb_to_ycbcr_kernel<<<blocks, threads, 0, stream>>>(d_rgb, count, d_planes);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaMemcpyAsync(h_planes, d_planes, plane_bytes, cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_rgb, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_planes, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (!check_cuda(env, "CUDA RGB->YCbCr completion", status)) {
        safe_free_host(h_rgb); safe_free_host(h_planes); return nullptr;
    }
    jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(plane_count));
    if (result) env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(plane_count), h_planes);
    safe_free_host(h_rgb); safe_free_host(h_planes);
    return env->ExceptionCheck() ? nullptr : result;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_thesis_rap_gpu_CudaImagePreprocessNative_nativeResampleAndPad(
        JNIEnv* env, jclass,
        jdoubleArray source_array,
        jint source_width,
        jint source_height,
        jint ratio_x,
        jint ratio_y,
        jint padded_width,
        jint padded_height) {
    if (!source_array || source_width <= 0 || source_height <= 0
            || ratio_x <= 0 || ratio_y <= 0 || padded_width <= 0 || padded_height <= 0) {
        throw_java(env, "java/lang/IllegalArgumentException", "Invalid resample/pad inputs");
        return nullptr;
    }
    const jsize source_count = env->GetArrayLength(source_array);
    if (source_count != source_width * source_height) {
        throw_java(env, "java/lang/IllegalArgumentException", "Source plane length mismatch");
        return nullptr;
    }
    const int output_width = (source_width + ratio_x - 1) / ratio_x;
    const int output_height = (source_height + ratio_y - 1) / ratio_y;
    if (padded_width < output_width || padded_height < output_height) {
        throw_java(env, "java/lang/IllegalArgumentException", "Padded grid smaller than resampled plane");
        return nullptr;
    }
    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(selected_device()))) return nullptr;
    if (!preflight_cuda(env)) return nullptr;
    auto lease = stream_pool().acquire(env);
    if (env->ExceptionCheck()) return nullptr;
    cudaStream_t stream = lease.stream();

    const std::size_t source_bytes = static_cast<std::size_t>(source_count) * sizeof(double);
    const std::size_t output_count = static_cast<std::size_t>(padded_width) * padded_height;
    const std::size_t output_bytes = output_count * sizeof(double);
    double *h_source=nullptr, *h_output=nullptr, *d_source=nullptr, *d_output=nullptr;
    cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&h_source), source_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_output), output_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_source), source_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_output), output_bytes, stream);
    if (!check_cuda(env, "CUDA resample/pad allocation", status)) {
        if (h_source) safe_free_host(h_source); if (h_output) safe_free_host(h_output);
        return nullptr;
    }
    env->GetDoubleArrayRegion(source_array, 0, source_count, h_source);
    if (env->ExceptionCheck()) {
        cudaFreeAsync(d_source, stream); cudaFreeAsync(d_output, stream);
        safe_free_host(h_source); safe_free_host(h_output); return nullptr;
    }
    status = cudaMemcpyAsync(d_source, h_source, source_bytes, cudaMemcpyHostToDevice, stream);
    int threads = 256;
    int blocks = (static_cast<int>(output_count) + threads - 1) / threads;
    if (status == cudaSuccess) {
        resample_and_pad_kernel<<<blocks, threads, 0, stream>>>(
                d_source, source_width, source_height,
                ratio_x, ratio_y, output_width, output_height,
                padded_width, padded_height, d_output);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaMemcpyAsync(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_source, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_output, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (!check_cuda(env, "CUDA resample/pad completion", status)) {
        safe_free_host(h_source); safe_free_host(h_output); return nullptr;
    }
    jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(output_count));
    if (result) env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output_count), h_output);
    safe_free_host(h_source); safe_free_host(h_output);
    return env->ExceptionCheck() ? nullptr : result;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_thesis_rap_gpu_CudaTrueParallelNative_nativeMedian(
        JNIEnv* env, jclass, jdoubleArray values_array) {
    if (!values_array) {
        throw_java(env, "java/lang/IllegalArgumentException", "Median input cannot be null");
        return 0.0;
    }
    jsize count = env->GetArrayLength(values_array);
    if (count <= 0) {
        throw_java(env, "java/lang/IllegalArgumentException", "Median input cannot be empty");
        return 0.0;
    }
    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(selected_device()))) return 0.0;
    if (!preflight_cuda(env)) return 0.0;
    auto lease = stream_pool().acquire(env);
    if (env->ExceptionCheck()) return 0.0;
    cudaStream_t stream = lease.stream();
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(double);
    double *h_values=nullptr, *d_values=nullptr;
    cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&h_values), bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_values), bytes, stream);
    if (!check_cuda(env, "CUDA median allocation", status)) {
        if (h_values) safe_free_host(h_values); return 0.0;
    }
    env->GetDoubleArrayRegion(values_array, 0, count, h_values);
    if (env->ExceptionCheck()) { cudaFreeAsync(d_values, stream); safe_free_host(h_values); return 0.0; }
    for (jsize i = 0; i < count; ++i) {
        if (!std::isfinite(h_values[i]) || h_values[i] < 0.0) {
            cudaFreeAsync(d_values, stream); safe_free_host(h_values);
            throw_java(env, "java/lang/IllegalArgumentException", "Median requires finite nonnegative values");
            return 0.0;
        }
    }
    status = cudaMemcpyAsync(d_values, h_values, bytes, cudaMemcpyHostToDevice, stream);
    if (!check_cuda(env, "CUDA median H2D", status)) { safe_free_host(h_values); return 0.0; }
    try {
        thrust::sort(thrust::cuda::par.on(stream), d_values, d_values + count);
        cudaError_t sort_status = cudaStreamSynchronize(stream);
        if (sort_status != cudaSuccess) {
            rap_cuda_crash_guard::observe(sort_status);
            throw std::runtime_error(cuda_error("CUDA Thrust sort synchronization", sort_status));
        }
    } catch (const std::exception& error) {
        if (!rap_cuda_crash_guard::poisoned()) cudaFreeAsync(d_values, stream);
        if (!rap_cuda_crash_guard::poisoned()) safe_free_host(h_values);
        throw_java(env, "java/lang/IllegalStateException", std::string("Thrust median sort failed: ") + error.what());
        return 0.0;
    } catch (...) {
        throw_java(env, "java/lang/IllegalStateException",
                   "Thrust median sort failed with a non-standard native exception");
        return 0.0;
    }
    double* h_pair = nullptr;
    status = cudaHostAlloc(reinterpret_cast<void**>(&h_pair), 2*sizeof(double), cudaHostAllocPortable);
    if (!check_cuda(env, "CUDA median pinned output allocation", status)) {
        cudaFreeAsync(d_values, stream); safe_free_host(h_values); return 0.0;
    }
    h_pair[0] = 0.0; h_pair[1] = 0.0;
    if ((count & 1) != 0) {
        status = cudaMemcpyAsync(h_pair, d_values + count/2, sizeof(double), cudaMemcpyDeviceToHost, stream);
        if (status == cudaSuccess) status = cudaFreeAsync(d_values, stream);
        if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
        h_pair[1] = h_pair[0];
    } else {
        status = cudaMemcpyAsync(h_pair, d_values + count/2 - 1, 2*sizeof(double), cudaMemcpyDeviceToHost, stream);
        if (status == cudaSuccess) status = cudaFreeAsync(d_values, stream);
        if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    }
    safe_free_host(h_values);
    if (!check_cuda(env, "CUDA median completion", status)) { safe_free_host(h_pair); return 0.0; }
    double result = 0.5 * (h_pair[0] + h_pair[1]);
    safe_free_host(h_pair);
    return result;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_thesis_rap_gpu_CudaTrueParallelNative_nativeSortCandidateIndices(
        JNIEnv* env, jclass, jdoubleArray costs_array, jintArray fields_array) {
    if (!costs_array || !fields_array) {
        throw_java(env, "java/lang/IllegalArgumentException", "Candidate-sort inputs cannot be null");
        return nullptr;
    }
    const jsize count = env->GetArrayLength(costs_array);
    const jsize field_count = env->GetArrayLength(fields_array);
    if (count < 0 || field_count != count * 3) {
        throw_java(env, "java/lang/IllegalArgumentException", "Candidate-sort field count mismatch");
        return nullptr;
    }
    if (count == 0) return env->NewIntArray(0);
    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(selected_device()))) return nullptr;
    if (!preflight_cuda(env)) return nullptr;
    auto lease = stream_pool().acquire(env);
    if (env->ExceptionCheck()) return nullptr;
    cudaStream_t stream = lease.stream();
    const std::size_t cost_bytes = static_cast<std::size_t>(count) * sizeof(double);
    const std::size_t field_bytes = static_cast<std::size_t>(field_count) * sizeof(int32_t);
    const std::size_t key_bytes = static_cast<std::size_t>(count) * sizeof(CandidateOrderKey);
    double *h_costs=nullptr, *d_costs=nullptr;
    int32_t *h_fields=nullptr, *d_fields=nullptr;
    CandidateOrderKey* d_keys=nullptr;
    cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&h_costs), cost_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_fields), field_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_costs), cost_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_fields), field_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_keys), key_bytes, stream);
    if (!check_cuda(env, "CUDA candidate-sort allocation", status)) {
        if (h_costs) safe_free_host(h_costs); if (h_fields) safe_free_host(h_fields); return nullptr;
    }
    env->GetDoubleArrayRegion(costs_array, 0, count, h_costs);
    env->GetIntArrayRegion(fields_array, 0, field_count, reinterpret_cast<jint*>(h_fields));
    if (env->ExceptionCheck()) {
        cudaFreeAsync(d_costs, stream); cudaFreeAsync(d_fields, stream); cudaFreeAsync(d_keys, stream);
        safe_free_host(h_costs); safe_free_host(h_fields); return nullptr;
    }
    status = cudaMemcpyAsync(d_costs, h_costs, cost_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_fields, h_fields, field_bytes, cudaMemcpyHostToDevice, stream);
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    if (status == cudaSuccess) {
        build_candidate_order_keys_kernel<<<blocks, threads, 0, stream>>>(d_costs, d_fields, count, d_keys);
        status = cudaGetLastError();
    }
    if (!check_cuda(env, "CUDA candidate-sort setup", status)) {
        safe_free_host(h_costs); safe_free_host(h_fields); return nullptr;
    }
    try {
        thrust::sort(thrust::cuda::par.on(stream), d_keys, d_keys + count, CandidateOrderLess());
        cudaError_t sort_status = cudaStreamSynchronize(stream);
        if (sort_status != cudaSuccess) {
            rap_cuda_crash_guard::observe(sort_status);
            throw std::runtime_error(cuda_error("CUDA Thrust candidate sort synchronization", sort_status));
        }
    } catch (const std::exception& error) {
        if (!rap_cuda_crash_guard::poisoned()) {
            cudaFreeAsync(d_costs, stream); cudaFreeAsync(d_fields, stream); cudaFreeAsync(d_keys, stream);
            safe_free_host(h_costs); safe_free_host(h_fields);
        }
        throw_java(env, "java/lang/IllegalStateException", std::string("Thrust candidate sort failed: ") + error.what());
        return nullptr;
    } catch (...) {
        throw_java(env, "java/lang/IllegalStateException",
                   "Thrust candidate sort failed with a non-standard native exception");
        return nullptr;
    }
    CandidateOrderKey* h_keys = nullptr;
    status = cudaHostAlloc(reinterpret_cast<void**>(&h_keys), key_bytes, cudaHostAllocPortable);
    if (!check_cuda(env, "CUDA candidate-sort pinned output allocation", status)) {
        cudaFreeAsync(d_costs, stream); cudaFreeAsync(d_fields, stream); cudaFreeAsync(d_keys, stream);
        safe_free_host(h_costs); safe_free_host(h_fields); return nullptr;
    }
    status = cudaMemcpyAsync(h_keys, d_keys, key_bytes, cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_costs, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_fields, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_keys, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (!check_cuda(env, "CUDA candidate-sort completion", status)) {
        safe_free_host(h_costs); safe_free_host(h_fields); safe_free_host(h_keys); return nullptr;
    }
    std::vector<jint> order(static_cast<std::size_t>(count));
    for (jsize i = 0; i < count; ++i) order[static_cast<std::size_t>(i)] = h_keys[static_cast<std::size_t>(i)].index;
    safe_free_host(h_costs); safe_free_host(h_fields); safe_free_host(h_keys);
    jintArray result = env->NewIntArray(count);
    if (result) env->SetIntArrayRegion(result, 0, count, order.data());
    return env->ExceptionCheck() ? nullptr : result;
}


extern "C" JNIEXPORT jboolean JNICALL
Java_thesis_rap_gpu_CudaBalanceNative_nativeRuntimeAvailable(JNIEnv*, jclass) {
    return runtime_available() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_thesis_rap_gpu_CudaBalanceNative_nativeCompute(
        JNIEnv* env, jclass,
        jdoubleArray rho_plus_array,
        jdoubleArray rho_minus_array,
        jdoubleArray minimum_plus_array,
        jdoubleArray minimum_minus_array,
        jintArray zero_flow_plus_array,
        jintArray zero_flow_minus_array,
        jintArray tie_directions_array,
        jdouble multiplier,
        jdouble wet_cost) {
    if (!rho_plus_array || !rho_minus_array || !minimum_plus_array
            || !minimum_minus_array || !zero_flow_plus_array
            || !zero_flow_minus_array || !tie_directions_array) {
        throw_java(env, "java/lang/IllegalArgumentException", "CUDA balance inputs cannot be null");
        return nullptr;
    }
    jsize count = env->GetArrayLength(rho_plus_array);
    if (count < 0
            || env->GetArrayLength(rho_minus_array) != count
            || env->GetArrayLength(minimum_plus_array) != count
            || env->GetArrayLength(minimum_minus_array) != count
            || env->GetArrayLength(zero_flow_plus_array) != count
            || env->GetArrayLength(zero_flow_minus_array) != count
            || env->GetArrayLength(tie_directions_array) != count) {
        throw_java(env, "java/lang/IllegalArgumentException", "CUDA balance dimensions mismatch");
        return nullptr;
    }
    if (count == 0) return env->NewDoubleArray(0);
    if (!(wet_cost > 0.0) || !std::isfinite(wet_cost)) {
        throw_java(env, "java/lang/IllegalArgumentException", "Invalid wet cost");
        return nullptr;
    }
    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(selected_device()))) return nullptr;
    if (!preflight_cuda(env)) return nullptr;
    auto lease = stream_pool().acquire(env);
    if (env->ExceptionCheck()) return nullptr;
    cudaStream_t stream = lease.stream();

    const std::size_t double_bytes = static_cast<std::size_t>(count) * sizeof(double);
    const std::size_t int_bytes = static_cast<std::size_t>(count) * sizeof(int32_t);
    const std::size_t output_count = static_cast<std::size_t>(count) * 4u;
    const std::size_t output_bytes = output_count * sizeof(double);

    double *h_rp=nullptr,*h_rm=nullptr,*h_mp=nullptr,*h_mm=nullptr,*h_out=nullptr;
    int32_t *h_zp=nullptr,*h_zm=nullptr,*h_tie=nullptr;
    double *d_rp=nullptr,*d_rm=nullptr,*d_mp=nullptr,*d_mm=nullptr,*d_out=nullptr;
    int32_t *d_zp=nullptr,*d_zm=nullptr,*d_tie=nullptr;
    cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&h_rp), double_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_rm), double_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_mp), double_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_mm), double_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_zp), int_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_zm), int_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_tie), int_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_out), output_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_rp), double_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_rm), double_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_mp), double_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_mm), double_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_zp), int_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_zm), int_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_tie), int_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_out), output_bytes, stream);
    if (!check_cuda(env, "CUDA balance allocation", status)) return nullptr;

    env->GetDoubleArrayRegion(rho_plus_array, 0, count, h_rp);
    env->GetDoubleArrayRegion(rho_minus_array, 0, count, h_rm);
    env->GetDoubleArrayRegion(minimum_plus_array, 0, count, h_mp);
    env->GetDoubleArrayRegion(minimum_minus_array, 0, count, h_mm);
    env->GetIntArrayRegion(zero_flow_plus_array, 0, count, reinterpret_cast<jint*>(h_zp));
    env->GetIntArrayRegion(zero_flow_minus_array, 0, count, reinterpret_cast<jint*>(h_zm));
    env->GetIntArrayRegion(tie_directions_array, 0, count, reinterpret_cast<jint*>(h_tie));
    if (env->ExceptionCheck()) return nullptr;

    status = cudaMemcpyAsync(d_rp, h_rp, double_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_rm, h_rm, double_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_mp, h_mp, double_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_mm, h_mm, double_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_zp, h_zp, int_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_zm, h_zm, int_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_tie, h_tie, int_bytes, cudaMemcpyHostToDevice, stream);
    if (!check_cuda(env, "CUDA balance H2D", status)) return nullptr;

    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    balance_plan_kernel<<<blocks, threads, 0, stream>>>(
            d_rp,d_rm,d_mp,d_mm,d_zp,d_zm,d_tie,count,
            static_cast<double>(multiplier),static_cast<double>(wet_cost),d_out);
    status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaMemcpyAsync(h_out, d_out, output_bytes, cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_rp, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_rm, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_mp, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_mm, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_zp, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_zm, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_tie, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_out, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (!check_cuda(env, "CUDA balance completion", status)) return nullptr;

    jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(output_count));
    if (result) env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output_count), h_out);
    safe_free_host(h_rp); safe_free_host(h_rm); safe_free_host(h_mp); safe_free_host(h_mm);
    safe_free_host(h_zp); safe_free_host(h_zm); safe_free_host(h_tie); safe_free_host(h_out);
    return env->ExceptionCheck() ? nullptr : result;
}


extern "C" JNIEXPORT jboolean JNICALL
Java_thesis_rap_gpu_CudaCanonicalLumaNative_nativeRuntimeAvailable(JNIEnv*, jclass) {
    return runtime_available() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_thesis_rap_gpu_CudaCanonicalLumaNative_nativeReconstructAndHistogram(
        JNIEnv* env, jclass,
        jintArray coefficients_array,
        jint width_blocks,
        jint height_blocks,
        jint width,
        jint height,
        jdoubleArray basis_array) {
    if (!coefficients_array || !basis_array
            || width_blocks <= 0 || height_blocks <= 0
            || width <= 0 || height <= 0) {
        throw_java(env, "java/lang/IllegalArgumentException", "Invalid canonical-luminance inputs");
        return nullptr;
    }
    jsize coefficient_count = env->GetArrayLength(coefficients_array);
    jsize basis_count = env->GetArrayLength(basis_array);
    int block_count = width_blocks * height_blocks;
    int pixel_count = width * height;
    if (coefficient_count != block_count * 64 || basis_count != 4096
            || width_blocks * 8 < width || height_blocks * 8 < height) {
        throw_java(env, "java/lang/IllegalArgumentException", "Canonical-luminance dimensions mismatch");
        return nullptr;
    }
    if (!check_cuda(env, "cudaSetDevice", cudaSetDevice(selected_device()))) return nullptr;
    if (!preflight_cuda(env)) return nullptr;
    auto lease = stream_pool().acquire(env);
    if (env->ExceptionCheck()) return nullptr;
    cudaStream_t stream = lease.stream();

    const std::size_t coefficient_bytes = static_cast<std::size_t>(coefficient_count) * sizeof(int32_t);
    const std::size_t basis_bytes = 4096u * sizeof(double);
    const std::size_t luminance_bytes = static_cast<std::size_t>(pixel_count) * sizeof(double);
    const std::size_t hist_int_bytes = 256u * sizeof(unsigned int);
    const std::size_t output_count = static_cast<std::size_t>(pixel_count) + 256u;
    const std::size_t output_bytes = output_count * sizeof(double);

    int32_t* h_coeff=nullptr;
    double *h_basis=nullptr,*h_output=nullptr;
    int32_t* d_coeff=nullptr;
    double *d_basis=nullptr,*d_luminance=nullptr,*d_hist_normalized=nullptr;
    unsigned int* d_hist=nullptr;
    cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&h_coeff), coefficient_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_basis), basis_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaHostAlloc(reinterpret_cast<void**>(&h_output), output_bytes, cudaHostAllocPortable);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_coeff), coefficient_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_basis), basis_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_luminance), luminance_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_hist), hist_int_bytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(reinterpret_cast<void**>(&d_hist_normalized), 256u*sizeof(double), stream);
    if (!check_cuda(env, "CUDA canonical-luminance allocation", status)) return nullptr;

    env->GetIntArrayRegion(coefficients_array, 0, coefficient_count, reinterpret_cast<jint*>(h_coeff));
    env->GetDoubleArrayRegion(basis_array, 0, 4096, h_basis);
    if (env->ExceptionCheck()) return nullptr;
    status = cudaMemcpyAsync(d_coeff, h_coeff, coefficient_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(d_basis, h_basis, basis_bytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaMemsetAsync(d_hist, 0, hist_int_bytes, stream);
    if (!check_cuda(env, "CUDA canonical-luminance H2D", status)) return nullptr;

    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    canonical_luma_histogram_kernel<<<blocks, threads, 0, stream>>>(
            d_coeff, d_basis, width_blocks, width, height,
            d_luminance, d_hist);
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        normalize_histogram_kernel<<<1,256,0,stream>>>(
                d_hist, pixel_count, d_hist_normalized);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaMemcpyAsync(
            h_output, d_luminance, luminance_bytes,
            cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaMemcpyAsync(
            h_output + pixel_count, d_hist_normalized, 256u*sizeof(double),
            cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_coeff, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_basis, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_luminance, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_hist, stream);
    if (status == cudaSuccess) status = cudaFreeAsync(d_hist_normalized, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (!check_cuda(env, "CUDA canonical-luminance completion", status)) return nullptr;

    jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(output_count));
    if (result) env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output_count), h_output);
    safe_free_host(h_coeff); safe_free_host(h_basis); safe_free_host(h_output);
    return env->ExceptionCheck() ? nullptr : result;
}
