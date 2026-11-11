#include <jni.h>
#include <vector>
#include <stdexcept>
#include <limits>
#include "original_stc_adapter.h"
#include "original_stc/common.h"

static void throw_java_class(JNIEnv* env, const char* class_name, const char* msg) {
    if (env->ExceptionCheck()) return;
    jclass ex = env->FindClass(class_name);
    if (ex != nullptr) env->ThrowNew(ex, msg);
}

static void throw_java(JNIEnv* env, const char* msg) {
    throw_java_class(env, "java/lang/RuntimeException", msg);
}

static std::vector<unsigned char> read_jbyte_array(JNIEnv* env, jbyteArray arr) {
    if (arr == nullptr) {
        throw std::runtime_error("byte-array JNI argument is null");
    }
    jsize n = env->GetArrayLength(arr);
    std::vector<unsigned char> out((size_t)n);
    if (n > 0) env->GetByteArrayRegion(arr, 0, n,
            reinterpret_cast<jbyte*>(out.data()));
    if (env->ExceptionCheck()) throw std::runtime_error("byte-array JNI copy failed");
    return out;
}

static std::vector<double> read_jdouble_array(JNIEnv* env, jdoubleArray arr) {
    if (arr == nullptr) {
        throw std::runtime_error("double-array JNI argument is null");
    }
    jsize n = env->GetArrayLength(arr);
    std::vector<double> out((size_t)n);
    if (n > 0) env->GetDoubleArrayRegion(arr, 0, n, out.data());
    if (env->ExceptionCheck()) throw std::runtime_error("double-array JNI copy failed");
    return out;
}

static jbyteArray make_jbyte_array(JNIEnv* env, const std::vector<unsigned char>& v) {
    if (v.size() > (size_t)std::numeric_limits<jsize>::max()) {
        throw std::runtime_error(
                "native STC output exceeds JNI array range");
    }
    jbyteArray arr = env->NewByteArray((jsize)v.size());
    if (!arr) return nullptr;
    std::vector<jbyte> tmp(v.begin(), v.end());
    env->SetByteArrayRegion(arr, 0, (jsize)tmp.size(), tmp.data());
    if (env->ExceptionCheck()) return nullptr;
    return arr;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_thesis_rap_stc_OriginalStcNative_embedBinary(
    JNIEnv* env,
    jclass,
    jbyteArray coverBits,
    jdoubleArray costs,
    jbyteArray messageBits,
    jint constraintHeight
) {
    try {
        if (constraintHeight < 4 || constraintHeight > 20) {
            throw std::runtime_error(
                    "constraintHeight must be in 4..20");
        }
        auto x = read_jbyte_array(env, coverBits);
        auto rho = read_jdouble_array(env, costs);
        auto m = read_jbyte_array(env, messageBits);
        if (x.size() != rho.size()) {
            throw std::runtime_error(
                    "coverBits and costs length mismatch");
        }
        auto y = original_stc_embed_adapter(x, rho, m, (int)constraintHeight);
        return make_jbyte_array(env, y);
    } catch (const stc_exception& e) {
        const char* type = e.error_id == 4
                ? "thesis/rap/stc/NoFiniteStcSolutionException"
                : e.error_id == 2 ? "java/lang/OutOfMemoryError"
                : "java/lang/RuntimeException";
        throw_java_class(env, type, e.what());
        return nullptr;
    } catch (const std::bad_alloc& e) {
        throw_java_class(env, "java/lang/OutOfMemoryError", e.what());
        return nullptr;
    } catch (const std::exception& e) {
        throw_java(env, e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_thesis_rap_stc_OriginalStcNative_extractBinary(
    JNIEnv* env,
    jclass,
    jbyteArray stegoBits,
    jint messageBitLength,
    jint constraintHeight
) {
    try {
        if (constraintHeight < 4 || constraintHeight > 20) {
            throw std::runtime_error(
                    "constraintHeight must be in 4..20");
        }
        if (messageBitLength <= 0) {
            throw std::runtime_error(
                    "messageBitLength must be positive");
        }
        auto y = read_jbyte_array(env, stegoBits);
        auto m = original_stc_extract_adapter(y, (int)messageBitLength, (int)constraintHeight);
        return make_jbyte_array(env, m);
    } catch (const stc_exception& e) {
        const char* type = e.error_id == 4
                ? "thesis/rap/stc/NoFiniteStcSolutionException"
                : e.error_id == 2 ? "java/lang/OutOfMemoryError"
                : "java/lang/RuntimeException";
        throw_java_class(env, type, e.what());
        return nullptr;
    } catch (const std::bad_alloc& e) {
        throw_java_class(env, "java/lang/OutOfMemoryError", e.what());
        return nullptr;
    } catch (const std::exception& e) {
        throw_java(env, e.what());
        return nullptr;
    }
}
