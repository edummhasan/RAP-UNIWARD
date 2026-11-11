#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <limits>

struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

typedef struct my_error_mgr* my_error_ptr;

static void my_error_exit(j_common_ptr cinfo) {
    my_error_ptr myerr = (my_error_ptr)cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

static void throw_java(JNIEnv* env, const char* msg) {
    if (env->ExceptionCheck()) return;
    jclass ex = env->FindClass("java/lang/RuntimeException");
    if (ex != NULL) {
        env->ThrowNew(ex, msg);
    }
}

extern "C" JNIEXPORT jobject JNICALL Java_thesis_rap_jni_JpegCore_readCoefficients
(JNIEnv* env, jclass, jstring pathStr) {

    if (pathStr == NULL) {
        throw_java(env, "JPEG input path cannot be null");
        return NULL;
    }
    const char* path = env->GetStringUTFChars(pathStr, nullptr);
    if (path == NULL) {
        return NULL;
    }
    FILE* infile = fopen(path, "rb");

    if (!infile) {
        env->ReleaseStringUTFChars(pathStr, path);
        throw_java(env, "Cannot open JPEG input");
        return NULL;
    }

    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    volatile bool decompressorCreated = false;
    // The pointer itself changes after setjmp and is read after longjmp.
    jint* volatile activeCoefficientBuffer = NULL;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        if (activeCoefficientBuffer != NULL) {
            free((void*)activeCoefficientBuffer);
            activeCoefficientBuffer = NULL;
        }
        if (decompressorCreated) {
            jpeg_destroy_decompress(&cinfo);
        }
        fclose(infile);
        env->ReleaseStringUTFChars(pathStr, path);
        throw_java(env, "JPEG read error");
        return NULL;
    }

    jpeg_create_decompress(&cinfo);
    decompressorCreated = true;
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    jvirt_barray_ptr* coef_arrays = jpeg_read_coefficients(&cinfo);

    jclass compCls = env->FindClass("thesis/rap/jni/JpegComponent");
    if (compCls == NULL) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        env->ReleaseStringUTFChars(pathStr, path);
        return NULL;
    }
    jmethodID compCtor = env->GetMethodID(compCls, "<init>", "(IIIIII[I[I)V");
    if (compCtor == NULL) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        env->ReleaseStringUTFChars(pathStr, path);
        return NULL;
    }
    jobjectArray comps = env->NewObjectArray(cinfo.num_components, compCls, NULL);
    if (comps == NULL) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        env->ReleaseStringUTFChars(pathStr, path);
        return NULL;
    }

    for (int comp = 0; comp < cinfo.num_components; comp++) {
        jpeg_component_info* ci = &cinfo.comp_info[comp];

        int wb = ci->width_in_blocks;
        int hb = ci->height_in_blocks;
        long long totalWide =
            (long long)wb * (long long)hb * (long long)DCTSIZE2;
        if (wb <= 0 || hb <= 0 || totalWide <= 0
                || totalWide > std::numeric_limits<jsize>::max()) {
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            env->ReleaseStringUTFChars(pathStr, path);
            throw_java(env, "JPEG component coefficient count is invalid or too large");
            return NULL;
        }
        jsize total = (jsize)totalWide;

        jintArray coeffArr = env->NewIntArray(total);
        if (coeffArr == NULL) {
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            env->ReleaseStringUTFChars(pathStr, path);
            return NULL;
        }
        jint* coeffs = (jint*)malloc(
            sizeof(jint) * (size_t)total);
        if (coeffs == NULL) {
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            env->ReleaseStringUTFChars(pathStr, path);
            throw_java(env, "Unable to allocate JPEG coefficient buffer");
            return NULL;
        }
        activeCoefficientBuffer = coeffs;

        int pos = 0;
        for (int by = 0; by < hb; by++) {
            JBLOCKARRAY row = (*cinfo.mem->access_virt_barray)(
                (j_common_ptr)&cinfo,
                                                               coef_arrays[comp],
                                                               by,
                                                               1,
                                                               FALSE
            );

            for (int bx = 0; bx < wb; bx++) {
                JCOEF* block = row[0][bx];
                for (int k = 0; k < 64; k++) {
                    coeffs[pos++] = (jint)block[k];
                }
            }
        }

        env->SetIntArrayRegion(coeffArr, 0, total, coeffs);
        free(coeffs);
        activeCoefficientBuffer = NULL;
        if (env->ExceptionCheck()) {
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            env->ReleaseStringUTFChars(pathStr, path);
            return NULL;
        }

        jintArray qArr = env->NewIntArray(DCTSIZE2);
        if (qArr == NULL) {
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            env->ReleaseStringUTFChars(pathStr, path);
            return NULL;
        }
        jint q[DCTSIZE2];

        if (ci->quant_table == NULL) {
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            env->ReleaseStringUTFChars(pathStr, path);
            throw_java(env, "JPEG component has no quantization table");
            return NULL;
        }
        for (int k = 0; k < DCTSIZE2; k++) {
            q[k] = (jint)ci->quant_table->quantval[k];
        }

        env->SetIntArrayRegion(qArr, 0, DCTSIZE2, q);
        if (env->ExceptionCheck()) {
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            env->ReleaseStringUTFChars(pathStr, path);
            return NULL;
        }

        jobject compObj = env->NewObject(
            compCls,
            compCtor,
            comp,
            wb,
            hb,
            ci->h_samp_factor,
            ci->v_samp_factor,
            ci->quant_tbl_no,
            coeffArr,
            qArr
        );

        if (compObj == NULL || env->ExceptionCheck()) {
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            env->ReleaseStringUTFChars(pathStr, path);
            return NULL;
        }
        env->SetObjectArrayElement(comps, comp, compObj);
        if (env->ExceptionCheck()) {
            jpeg_destroy_decompress(&cinfo);
            fclose(infile);
            env->ReleaseStringUTFChars(pathStr, path);
            return NULL;
        }
    }

    jclass imgCls = env->FindClass("thesis/rap/jni/JpegImage");
    if (imgCls == NULL) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        env->ReleaseStringUTFChars(pathStr, path);
        return NULL;
    }
    jmethodID imgCtor = env->GetMethodID(
        imgCls,
        "<init>",
        "(II[Lthesis/rap/jni/JpegComponent;)V"
    );
    if (imgCtor == NULL) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        env->ReleaseStringUTFChars(pathStr, path);
        return NULL;
    }

    jobject img = env->NewObject(
        imgCls,
        imgCtor,
        (jint)cinfo.image_width,
                                 (jint)cinfo.image_height,
                                 comps
    );
    if (img == NULL || env->ExceptionCheck()) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        env->ReleaseStringUTFChars(pathStr, path);
        return NULL;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    env->ReleaseStringUTFChars(pathStr, path);

    return img;
}

extern "C" JNIEXPORT void JNICALL Java_thesis_rap_jni_JpegCore_writeCoefficients
(JNIEnv* env, jclass, jstring inStr, jstring outStr, jobject imageObj) {

    if (inStr == NULL || outStr == NULL || imageObj == NULL) {
        throw_java(env, "JPEG input path, output path, and image cannot be null");
        return;
    }
    const char* inpath = env->GetStringUTFChars(inStr, nullptr);
    const char* outpath = env->GetStringUTFChars(outStr, nullptr);
    if (inpath == NULL || outpath == NULL) {
        if (inpath != NULL) {
            env->ReleaseStringUTFChars(inStr, inpath);
        }
        if (outpath != NULL) {
            env->ReleaseStringUTFChars(outStr, outpath);
        }
        return;
    }

    FILE* infile = NULL;
    FILE* outfile = NULL;

    struct jpeg_decompress_struct srcinfo;
    struct jpeg_compress_struct dstinfo;
    struct my_error_mgr jerr;

    volatile bool srcCreated = false;
    volatile bool dstCreated = false;

    infile = fopen(inpath, "rb");
    if (!infile) {
        throw_java(env, "Cannot open template JPEG");
        env->ReleaseStringUTFChars(inStr, inpath);
        env->ReleaseStringUTFChars(outStr, outpath);
        return;
    }

    outfile = fopen(outpath, "wb");
    if (!outfile) {
        fclose(infile);
        throw_java(env, "Cannot open output JPEG");
        env->ReleaseStringUTFChars(inStr, inpath);
        env->ReleaseStringUTFChars(outStr, outpath);
        return;
    }

    srcinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    dstinfo.err = &jerr.pub;

    // Both libjpeg objects use one recovery point. setjmp must be the entire
    // controlling expression, rather than an operand of logical OR.
    if (setjmp(jerr.setjmp_buffer)) {
        if (dstCreated) jpeg_destroy_compress(&dstinfo);
        if (srcCreated) jpeg_destroy_decompress(&srcinfo);
        if (infile) fclose(infile);
        if (outfile) fclose(outfile);

        env->ReleaseStringUTFChars(inStr, inpath);
        env->ReleaseStringUTFChars(outStr, outpath);

        throw_java(env, "JPEG write error");
        return;
    }

    jpeg_create_decompress(&srcinfo);
    srcCreated = true;

    jpeg_stdio_src(&srcinfo, infile);
    jpeg_read_header(&srcinfo, TRUE);

    jvirt_barray_ptr* coef_arrays = jpeg_read_coefficients(&srcinfo);

    jclass imgCls = env->GetObjectClass(imageObj);
    if (imgCls == NULL) {
        jpeg_destroy_decompress(&srcinfo);
        fclose(infile);
        fclose(outfile);
        env->ReleaseStringUTFChars(inStr, inpath);
        env->ReleaseStringUTFChars(outStr, outpath);
        return;
    }
    jfieldID widthF = env->GetFieldID(imgCls, "width", "I");
    jfieldID heightF = env->GetFieldID(imgCls, "height", "I");
    jfieldID compsF = env->GetFieldID(
        imgCls,
        "components",
        "[Lthesis/rap/jni/JpegComponent;"
    );
    if (widthF == NULL || heightF == NULL || compsF == NULL) {
        jpeg_destroy_decompress(&srcinfo);
        fclose(infile);
        fclose(outfile);
        env->ReleaseStringUTFChars(inStr, inpath);
        env->ReleaseStringUTFChars(outStr, outpath);
        return;
    }
    if (env->GetIntField(imageObj, widthF)
                != (jint)srcinfo.image_width
            || env->GetIntField(imageObj, heightF)
                != (jint)srcinfo.image_height) {
        throw_java(env, "JPEG dimensions do not match the template");
        jpeg_destroy_decompress(&srcinfo);
        fclose(infile);
        fclose(outfile);
        env->ReleaseStringUTFChars(inStr, inpath);
        env->ReleaseStringUTFChars(outStr, outpath);
        return;
    }

    jobjectArray comps = (jobjectArray)env->GetObjectField(imageObj, compsF);
    if (comps == NULL
            || env->GetArrayLength(comps) != srcinfo.num_components) {
        throw_java(env, "JPEG component count does not match the template");
        jpeg_destroy_decompress(&srcinfo);
        fclose(infile);
        fclose(outfile);
        env->ReleaseStringUTFChars(inStr, inpath);
        env->ReleaseStringUTFChars(outStr, outpath);
        return;
    }

    jclass compCls = env->FindClass("thesis/rap/jni/JpegComponent");
    if (compCls == NULL) {
        jpeg_destroy_decompress(&srcinfo);
        fclose(infile);
        fclose(outfile);
        env->ReleaseStringUTFChars(inStr, inpath);
        env->ReleaseStringUTFChars(outStr, outpath);
        return;
    }
    jfieldID componentIndexF =
        env->GetFieldID(compCls, "componentIndex", "I");
    jfieldID widthBlocksF =
        env->GetFieldID(compCls, "widthBlocks", "I");
    jfieldID heightBlocksF =
        env->GetFieldID(compCls, "heightBlocks", "I");
    jfieldID hSamplingF =
        env->GetFieldID(compCls, "hSamplingFactor", "I");
    jfieldID vSamplingF =
        env->GetFieldID(compCls, "vSamplingFactor", "I");
    jfieldID coeffF = env->GetFieldID(compCls, "coefficients", "[I");
    jfieldID quantF = env->GetFieldID(compCls, "quantTable", "[I");
    jfieldID quantIndexF = env->GetFieldID(compCls, "quantTableIndex", "I");
    if (componentIndexF == NULL
            || widthBlocksF == NULL
            || heightBlocksF == NULL
            || hSamplingF == NULL
            || vSamplingF == NULL
            || coeffF == NULL || quantF == NULL
            || quantIndexF == NULL) {
        jpeg_destroy_decompress(&srcinfo);
        fclose(infile);
        fclose(outfile);
        env->ReleaseStringUTFChars(inStr, inpath);
        env->ReleaseStringUTFChars(outStr, outpath);
        return;
    }

    bool quantTableSeen[NUM_QUANT_TBLS] = { false };

    for (int comp = 0; comp < srcinfo.num_components; comp++) {
        jobject compObj = env->GetObjectArrayElement(comps, comp);
        if (compObj == NULL) {
            throw_java(env, "JPEG component cannot be null");
            jpeg_destroy_decompress(&srcinfo);
            fclose(infile);
            fclose(outfile);
            env->ReleaseStringUTFChars(inStr, inpath);
            env->ReleaseStringUTFChars(outStr, outpath);
            return;
        }
        jintArray coeffArr = (jintArray)env->GetObjectField(compObj, coeffF);
        jintArray quantArr = (jintArray)env->GetObjectField(compObj, quantF);
        if (coeffArr == NULL || quantArr == NULL) {
            throw_java(env, "JPEG coefficient and quantization arrays cannot be null");
            jpeg_destroy_decompress(&srcinfo);
            fclose(infile);
            fclose(outfile);
            env->ReleaseStringUTFChars(inStr, inpath);
            env->ReleaseStringUTFChars(outStr, outpath);
            return;
        }
        jpeg_component_info* ci = &srcinfo.comp_info[comp];
        if (env->GetIntField(compObj, componentIndexF) != comp
                || env->GetIntField(compObj, widthBlocksF)
                    != static_cast<jint>(ci->width_in_blocks)
                || env->GetIntField(compObj, heightBlocksF)
                    != static_cast<jint>(ci->height_in_blocks)
                || env->GetIntField(compObj, hSamplingF)
                    != ci->h_samp_factor
                || env->GetIntField(compObj, vSamplingF)
                    != ci->v_samp_factor) {
            throw_java(env, "JPEG component layout does not match the template");
            jpeg_destroy_decompress(&srcinfo);
            fclose(infile);
            fclose(outfile);
            env->ReleaseStringUTFChars(inStr, inpath);
            env->ReleaseStringUTFChars(outStr, outpath);
            return;
        }
        long long expectedWide =
            (long long)ci->width_in_blocks
            * (long long)ci->height_in_blocks
            * (long long)DCTSIZE2;
        if (expectedWide <= 0
                || expectedWide > std::numeric_limits<jsize>::max()
                || env->GetArrayLength(coeffArr) != (jsize)expectedWide
                || env->GetArrayLength(quantArr) != DCTSIZE2) {
            throw_java(env, "JPEG coefficient or quantization array length mismatch");
            jpeg_destroy_decompress(&srcinfo);
            fclose(infile);
            fclose(outfile);
            env->ReleaseStringUTFChars(inStr, inpath);
            env->ReleaseStringUTFChars(outStr, outpath);
            return;
        }
        jint quantValues[DCTSIZE2];
        env->GetIntArrayRegion(quantArr, 0, DCTSIZE2, quantValues);
        if (env->ExceptionCheck()) {
            jpeg_destroy_decompress(&srcinfo);
            fclose(infile);
            fclose(outfile);
            env->ReleaseStringUTFChars(inStr, inpath);
            env->ReleaseStringUTFChars(outStr, outpath);
            return;
        }
        int quantIndex = env->GetIntField(compObj, quantIndexF);

        if (quantIndex < 0 || quantIndex >= NUM_QUANT_TBLS || ci->quant_tbl_no != quantIndex) {
            throw_java(env, "JPEG quantization-table index mismatch");
            jpeg_destroy_decompress(&srcinfo);
            fclose(infile);
            fclose(outfile);
            env->ReleaseStringUTFChars(inStr, inpath);
            env->ReleaseStringUTFChars(outStr, outpath);
            return;
        }
        if (srcinfo.quant_tbl_ptrs[quantIndex] == NULL) {
            srcinfo.quant_tbl_ptrs[quantIndex] = jpeg_alloc_quant_table((j_common_ptr)&srcinfo);
        }
        for (int k = 0; k < DCTSIZE2; k++) {
            int value = quantValues[k];
            if (value < 1 || value > 255) {
                throw_java(env, "Baseline JPEG quantization value outside 1..255");
                jpeg_destroy_decompress(&srcinfo);
                fclose(infile);
                fclose(outfile);
                env->ReleaseStringUTFChars(inStr, inpath);
                env->ReleaseStringUTFChars(outStr, outpath);
                return;
            }
            if (quantTableSeen[quantIndex]
                    && srcinfo.quant_tbl_ptrs[quantIndex]->quantval[k] != (UINT16)value) {
                throw_java(env, "Components sharing a JPEG quantization-table index disagree");
                jpeg_destroy_decompress(&srcinfo);
                fclose(infile);
                fclose(outfile);
                env->ReleaseStringUTFChars(inStr, inpath);
                env->ReleaseStringUTFChars(outStr, outpath);
                return;
            }
            srcinfo.quant_tbl_ptrs[quantIndex]->quantval[k] = (UINT16)value;
            if (ci->quant_table != NULL) {
                ci->quant_table->quantval[k] = (UINT16)value;
            }
        }
        quantTableSeen[quantIndex] = true;


        int wb = ci->width_in_blocks;
        int hb = ci->height_in_blocks;
        const jsize rowCount = (jsize)((long long)wb * DCTSIZE2);
        jint* rowValues = (jint*)(*srcinfo.mem->alloc_large)(
            (j_common_ptr)&srcinfo, JPOOL_IMAGE, (size_t)rowCount * sizeof(jint));

        for (int by = 0; by < hb; by++) {
            env->GetIntArrayRegion(coeffArr, by * rowCount, rowCount, rowValues);
            if (env->ExceptionCheck()) {
                jpeg_destroy_decompress(&srcinfo);
                fclose(infile);
                fclose(outfile);
                env->ReleaseStringUTFChars(inStr, inpath);
                env->ReleaseStringUTFChars(outStr, outpath);
                return;
            }
            int pos = 0;
            JBLOCKARRAY row = (*srcinfo.mem->access_virt_barray)(
                (j_common_ptr)&srcinfo,
                                                                 coef_arrays[comp],
                                                                 by,
                                                                 1,
                                                                 TRUE
            );

            for (int bx = 0; bx < wb; bx++) {
                JCOEF* block = row[0][bx];

                for (int k = 0; k < DCTSIZE2; k++) {
                    jint value = rowValues[pos++];
                    if (value < std::numeric_limits<JCOEF>::min()
                            || value > std::numeric_limits<JCOEF>::max()) {
                        throw_java(
                            env,
                            "JPEG coefficient is outside the native JCOEF range");
                        jpeg_destroy_decompress(&srcinfo);
                        fclose(infile);
                        fclose(outfile);
                        env->ReleaseStringUTFChars(inStr, inpath);
                        env->ReleaseStringUTFChars(outStr, outpath);
                        return;
                    }
                    block[k] = (JCOEF)value;
                }
            }
        }

    }

    jpeg_create_compress(&dstinfo);
    dstCreated = true;

    jpeg_stdio_dest(&dstinfo, outfile);
    jpeg_copy_critical_parameters(&srcinfo, &dstinfo);
    jpeg_write_coefficients(&dstinfo, coef_arrays);
    jpeg_finish_compress(&dstinfo);
    jpeg_finish_decompress(&srcinfo);

    jpeg_destroy_compress(&dstinfo);
    jpeg_destroy_decompress(&srcinfo);

    fclose(infile);
    fclose(outfile);

    env->ReleaseStringUTFChars(inStr, inpath);
    env->ReleaseStringUTFChars(outStr, outpath);
}

extern "C" JNIEXPORT jint JNICALL Java_thesis_rap_jni_JpegCore_libjpegVersion
(JNIEnv*, jclass) {
    return (jint)JPEG_LIB_VERSION;
}
