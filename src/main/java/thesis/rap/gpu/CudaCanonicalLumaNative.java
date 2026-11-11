package thesis.rap.gpu;

/** CUDA reconstruction of canonical Y luminance plus its exact 256-bin histogram. */
public final class CudaCanonicalLumaNative {
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

    private CudaCanonicalLumaNative() {}

    public static boolean isAvailable() {
        if (!LIBRARY_LOADED || CudaCrashGuard.isPoisoned()) return false;
        try { return CudaCrashGuard.call("CudaCanonicalLumaNative_RUNTIME_PROBE", CudaCanonicalLumaNative::nativeRuntimeAvailable); }
        catch (Throwable ignored) { return false; }
    }

    public static String diagnostic() {
        if (!LIBRARY_LOADED) return LOAD_DIAGNOSTIC;
        if (CudaCrashGuard.isPoisoned()) return CudaCrashGuard.diagnostic();
        try {
            return nativeRuntimeAvailable()
                    ? "CUDA canonical-luminance runtime available"
                    : "librapcuda loaded but CUDA canonical-luminance runtime unavailable";
        } catch (Throwable t) {
            return t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    /**
     * Returns [width*height luminance samples][256 normalized histogram bins].
     * Basis is 64 coefficient bases x 64 local pixels in natural-index order.
     */
    public static double[] reconstructAndHistogram(
            int[] coefficients,
            int widthBlocks,
            int heightBlocks,
            int width,
            int height,
            double[] basis
    ) {
        int blockCount = Math.multiplyExact(widthBlocks, heightBlocks);
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "CUDA canonical luminance unavailable: " + diagnostic());
        }
        if (coefficients == null || basis == null
                || widthBlocks <= 0 || heightBlocks <= 0
                || width <= 0 || height <= 0
                || coefficients.length != Math.multiplyExact(blockCount, 64)
                || basis.length != 64 * 64
                || widthBlocks * 8 < width || heightBlocks * 8 < height) {
            throw new IllegalArgumentException(
                    "Invalid canonical-luminance dimensions");
        }
        double[] result = CudaCrashGuard.call("CANONICAL_LUMA", () ->
                nativeReconstructAndHistogram(
                        coefficients, widthBlocks, heightBlocks,
                        width, height, basis));
        int expected = Math.addExact(Math.multiplyExact(width, height), 256);
        if (result == null || result.length != expected) {
            throw new IllegalStateException(
                    "CUDA canonical-luminance output length mismatch");
        }
        return result;
    }

    private static native boolean nativeRuntimeAvailable();
    private static native double[] nativeReconstructAndHistogram(
            int[] coefficients,
            int widthBlocks,
            int heightBlocks,
            int width,
            int height,
            double[] basis);
}
