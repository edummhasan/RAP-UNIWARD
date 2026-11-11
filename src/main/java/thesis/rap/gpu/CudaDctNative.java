package thesis.rap.gpu;

import thesis.rap.alg.JpegMath;

import java.util.Arrays;

/**
 * Exact-binary64 CUDA helper for QF-independent 8x8 forward DCT coefficients
 * and two-pass block variance.  The device receives the exact StrictMath
 * cosine/alpha constants computed by the Java reference implementation.
 */
public final class CudaDctNative {
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

    private CudaDctNative() {}

    public static boolean isAvailable() {
        if (!LIBRARY_LOADED || CudaCrashGuard.isPoisoned()) return false;
        try { return CudaCrashGuard.call("CudaDctNative_RUNTIME_PROBE", CudaDctNative::nativeRuntimeAvailable); }
        catch (Throwable ignored) { return false; }
    }

    public static String diagnostic() {
        if (!LIBRARY_LOADED) return LOAD_DIAGNOSTIC;
        if (CudaCrashGuard.isPoisoned()) return CudaCrashGuard.diagnostic();
        try {
            return nativeRuntimeAvailable()
                    ? "CUDA DCT runtime available"
                    : "librapcuda loaded but CUDA DCT runtime unavailable";
        } catch (Throwable t) {
            return t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    public static Result compute(
            double[][] paddedPlane,
            int widthBlocks,
            int heightBlocks) {
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "CUDA DCT requested but unavailable: " + diagnostic());
        }
        if (paddedPlane == null || paddedPlane.length != heightBlocks * 8
                || widthBlocks <= 0 || heightBlocks <= 0) {
            throw new IllegalArgumentException(
                    "paddedPlane dimensions must exactly match the JPEG block grid");
        }
        int width = widthBlocks * 8;
        double[] flat = new double[Math.multiplyExact(width, paddedPlane.length)];
        for (int y = 0; y < paddedPlane.length; y++) {
            if (paddedPlane[y] == null || paddedPlane[y].length != width) {
                throw new IllegalArgumentException(
                        "paddedPlane row width does not match the JPEG block grid");
            }
            System.arraycopy(paddedPlane[y], 0, flat, y * width, width);
        }
        double[] combined = CudaCrashGuard.call("DCT_VARIANCE", () ->
                nativeDctAndVariance(
                        flat, width, paddedPlane.length,
                        widthBlocks, heightBlocks,
                        JpegMath.dctCosineForAccelerator(),
                        JpegMath.dctAlphaForAccelerator()));
        int blocks = Math.multiplyExact(widthBlocks, heightBlocks);
        int dctLength = Math.multiplyExact(blocks, 64);
        if (combined == null || combined.length != dctLength + blocks) {
            throw new IllegalStateException(
                    "CUDA DCT returned an invalid result length");
        }
        return new Result(
                Arrays.copyOf(combined, dctLength),
                Arrays.copyOfRange(combined, dctLength, combined.length));
    }

    public record Result(double[] dct, double[] variances) {
        public Result {
            dct = Arrays.copyOf(dct, dct.length);
            variances = Arrays.copyOf(variances, variances.length);
        }
        @Override public double[] dct() { return Arrays.copyOf(dct, dct.length); }
        @Override public double[] variances() {
            return Arrays.copyOf(variances, variances.length);
        }
        /** Internal read-only access for SourceImageContext caching. */
        public double[] dctView() { return dct; }
        /** Internal read-only access for SourceImageContext caching. */
        public double[] variancesView() { return variances; }
    }

    private static native boolean nativeRuntimeAvailable();
    private static native double[] nativeDctAndVariance(
            double[] plane,
            int width,
            int height,
            int widthBlocks,
            int heightBlocks,
            double[] cosine64,
            double[] alpha8);
}
