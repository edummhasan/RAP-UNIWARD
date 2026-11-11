package thesis.rap.gpu;

import java.util.Arrays;


public final class CudaTrueParallelNative {
    private static final boolean LIBRARY_LOADED;
    private static final String LOAD_DIAGNOSTIC;

    static {
        boolean loaded = false;
        String diagnostic;
        try {
            System.loadLibrary("rapcuda");
            loaded = true;
            diagnostic = "librapcuda loaded";
        } catch (Throwable t) {
            diagnostic = t.getClass().getSimpleName() + ": " + t.getMessage();
        }
        LIBRARY_LOADED = loaded;
        LOAD_DIAGNOSTIC = diagnostic;
    }

    private CudaTrueParallelNative() {}

    public static boolean isAvailable() {
        if (!LIBRARY_LOADED || CudaCrashGuard.isPoisoned()) return false;
        try { return CudaCrashGuard.call("CudaTrueParallelNative_RUNTIME_PROBE", CudaTrueParallelNative::nativeRuntimeAvailable); }
        catch (Throwable ignored) { return false; }
    }

    public static String diagnostic() {
        if (!LIBRARY_LOADED) return LOAD_DIAGNOSTIC;
        if (CudaCrashGuard.isPoisoned()) return CudaCrashGuard.diagnostic();
        try {
            return nativeRuntimeAvailable()
                    ? "available streams=" + nativeStreamPoolSize()
                    : "library loaded but CUDA runtime unavailable";
        } catch (Throwable t) {
            return t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    /**
     * Computes HMAC-SHA-256 for every candidate.  {@code fields} contains
     * exactly 8 int32 values per candidate: c,b,bx,by,u,v,k,qRef.  Native code
     * prepends qf and geometry signature after the supplied domain, matching
     * StcKeyedOrder's big-endian serialization exactly.
     */
    public static byte[] hmacCandidateDigests(
            byte[] key32, byte[] domain, int qf, int geometrySignature,
            int[] fields) {
        if (!isAvailable()) throw new IllegalStateException(
                "CUDA true-parallel runtime unavailable: " + diagnostic());
        if (key32 == null || key32.length != 32) {
            throw new IllegalArgumentException("HMAC ordering key must be 32 bytes");
        }
        if (domain == null || domain.length == 0 || domain.length > 64) {
            throw new IllegalArgumentException("HMAC domain length must be 1..64");
        }
        if (fields == null || fields.length % 8 != 0) {
            throw new IllegalArgumentException("candidate field vector must contain 8 ints per candidate");
        }
        return CudaCrashGuard.call("HMAC_CANDIDATE_DIGESTS", () ->
                nativeHmacCandidateDigests(
                        Arrays.copyOf(key32, key32.length),
                        Arrays.copyOf(domain, domain.length),
                        qf, geometrySignature, fields));
    }

    /** GPU sort-backed exact nearest-rank P95 for a prefiltered finite array. */
    public static double percentile95(double[] finiteNonnegative) {
        if (!isAvailable()) throw new IllegalStateException(
                "CUDA true-parallel runtime unavailable: " + diagnostic());
        if (finiteNonnegative == null || finiteNonnegative.length == 0) return 0.0;
        return CudaCrashGuard.call("PERCENTILE95",
                () -> nativePercentile95(finiteNonnegative));
    }

    public static int streamPoolSize() {
        if (!isAvailable()) return 0;
        return nativeStreamPoolSize();
    }


    /** GPU exact median for a prefiltered finite nonnegative array. */
    public static double median(double[] finiteNonnegative) {
        if (!isAvailable()) throw new IllegalStateException(
                "CUDA true-parallel runtime unavailable: " + diagnostic());
        if (finiteNonnegative == null || finiteNonnegative.length == 0) {
            throw new IllegalArgumentException("median input cannot be null/empty");
        }
        return CudaCrashGuard.call("MEDIAN",
                () -> nativeMedian(finiteNonnegative));
    }

    /**
     * Sorts candidate indices by the exact frozen base order:
     * cost, component, blockIndex, natural coefficient index.
     * {@code fields} contains exactly three int32 values per candidate.
     */
    public static int[] sortCandidateIndices(double[] costs, int[] fields) {
        if (!isAvailable()) throw new IllegalStateException(
                "CUDA true-parallel runtime unavailable: " + diagnostic());
        if (costs == null || fields == null || fields.length != costs.length * 3) {
            throw new IllegalArgumentException("candidate-sort field vector mismatch");
        }
        return CudaCrashGuard.call("CANDIDATE_SORT",
                () -> nativeSortCandidateIndices(costs, fields));
    }

    private static native boolean nativeRuntimeAvailable();
    private static native int nativeStreamPoolSize();
    private static native byte[] nativeHmacCandidateDigests(
            byte[] key32, byte[] domain, int qf, int geometrySignature,
            int[] fields);
    private static native double nativePercentile95(double[] finiteNonnegative);

    private static native double nativeMedian(double[] finiteNonnegative);
    private static native int[] nativeSortCandidateIndices(
            double[] costs, int[] fields);
}
