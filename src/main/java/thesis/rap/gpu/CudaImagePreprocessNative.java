package thesis.rap.gpu;

import java.util.Arrays;

/**
 * Optional CUDA helpers for QF-independent image preprocessing.
 *
 * <p>The production callers always retain the exact Java implementation as a
 * fallback/parity oracle.  CUDA is used only when the process execution policy
 * permits it and the target host has successfully loaded {@code librapcuda}.</p>
 */
public final class CudaImagePreprocessNative {
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

    private CudaImagePreprocessNative() {}

    public static boolean isAvailable() {
        if (!LIBRARY_LOADED || CudaCrashGuard.isPoisoned()) return false;
        try { return CudaCrashGuard.call("CudaImagePreprocessNative_RUNTIME_PROBE", CudaImagePreprocessNative::nativeRuntimeAvailable); }
        catch (Throwable ignored) { return false; }
    }

    public static String diagnostic() {
        if (!LIBRARY_LOADED) return LOAD_DIAGNOSTIC;
        if (CudaCrashGuard.isPoisoned()) return CudaCrashGuard.diagnostic();
        try {
            return nativeRuntimeAvailable()
                    ? "CUDA image preprocessing available"
                    : "librapcuda loaded but CUDA image preprocessing unavailable";
        } catch (Throwable t) {
            return t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    /**
     * Returns three concatenated full-resolution planes: Y, Cb, Cr.
     */
    public static double[] rgbToYCbCr(int[] packedRgb, int width, int height) {
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "CUDA image preprocessing unavailable: " + diagnostic());
        }
        if (packedRgb == null || width <= 0 || height <= 0
                || packedRgb.length != Math.multiplyExact(width, height)) {
            throw new IllegalArgumentException("Invalid RGB raster dimensions");
        }
        double[] result = CudaCrashGuard.call("IMAGE_RGB_TO_YCBCR", () ->
                nativeRgbToYCbCr(
                        Arrays.copyOf(packedRgb, packedRgb.length), width, height));
        int expected = Math.multiplyExact(packedRgb.length, 3);
        if (result == null || result.length != expected) {
            throw new IllegalStateException(
                    "CUDA RGB->YCbCr returned an invalid result length");
        }
        return result;
    }

    /**
     * Box-resamples one full-resolution component and edge-pads it directly to
     * the requested JPEG block grid.  This fuses the two CPU passes used by
     * ImageUtil.jpegComponentPlane() and ImageUtil.padToBlockGrid().
     */
    public static double[] resampleAndPad(
            double[] source,
            int sourceWidth,
            int sourceHeight,
            int ratioX,
            int ratioY,
            int paddedWidth,
            int paddedHeight
    ) {
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "CUDA image preprocessing unavailable: " + diagnostic());
        }
        if (source == null || sourceWidth <= 0 || sourceHeight <= 0
                || source.length != Math.multiplyExact(sourceWidth, sourceHeight)
                || ratioX <= 0 || ratioY <= 0
                || paddedWidth <= 0 || paddedHeight <= 0
                || (paddedWidth & 7) != 0 || (paddedHeight & 7) != 0) {
            throw new IllegalArgumentException("Invalid CUDA resample/pad dimensions");
        }
        int outputWidth = (sourceWidth + ratioX - 1) / ratioX;
        int outputHeight = (sourceHeight + ratioY - 1) / ratioY;
        if (paddedWidth < outputWidth || paddedHeight < outputHeight) {
            throw new IllegalArgumentException(
                    "Padded dimensions are smaller than resampled component");
        }
        double[] result = CudaCrashGuard.call("IMAGE_RESAMPLE_PAD", () ->
                nativeResampleAndPad(
                        source, sourceWidth, sourceHeight,
                        ratioX, ratioY, paddedWidth, paddedHeight));
        int expected = Math.multiplyExact(paddedWidth, paddedHeight);
        if (result == null || result.length != expected) {
            throw new IllegalStateException(
                    "CUDA resample/pad returned an invalid result length");
        }
        return result;
    }

    private static native boolean nativeRuntimeAvailable();
    private static native double[] nativeRgbToYCbCr(
            int[] packedRgb, int width, int height);
    private static native double[] nativeResampleAndPad(
            double[] source,
            int sourceWidth,
            int sourceHeight,
            int ratioX,
            int ratioY,
            int paddedWidth,
            int paddedHeight);
}
