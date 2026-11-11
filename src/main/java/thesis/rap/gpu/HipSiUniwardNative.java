package thesis.rap.gpu;

/**
 * Thin JNI binding for the AMD HIP SI-UNIWARD batch-cost kernel.
 *
 * <p>The binding is deliberately optional. Merely loading the RAP Java code on
 * a CPU-only host does not require ROCm. Explicit HIP mode is fail-closed,
 * while AUTO mode may select the Java CPU reference backend when this library
 * or a HIP device is unavailable.</p>
 */
public final class HipSiUniwardNative {
    private static final boolean LIBRARY_LOADED;
    private static final String LOAD_DIAGNOSTIC;

    static {
        boolean loaded = false;
        String diagnostic = "not attempted";
        try {
            System.loadLibrary("raphip");
            loaded = true;
            diagnostic = "libraphip loaded";
        } catch (Throwable t) {
            diagnostic = t.getClass().getSimpleName() + ": " + String.valueOf(t.getMessage());
        }
        LIBRARY_LOADED = loaded;
        LOAD_DIAGNOSTIC = diagnostic;
    }

    private HipSiUniwardNative() {}

    public static boolean libraryLoaded() {
        return LIBRARY_LOADED;
    }

    public static boolean isAvailable() {
        if (!LIBRARY_LOADED) return false;
        try {
            return nativeRuntimeAvailable();
        } catch (Throwable ignored) {
            return false;
        }
    }

    public static String deviceName() {
        if (!LIBRARY_LOADED) return "unavailable (" + LOAD_DIAGNOSTIC + ")";
        try {
            String name = nativeDeviceName();
            return name == null || name.isBlank() ? "unknown HIP device" : name;
        } catch (Throwable t) {
            return "unavailable (" + t.getClass().getSimpleName() + ": "
                    + String.valueOf(t.getMessage()) + ")";
        }
    }

    public static String loadDiagnostic() {
        return LOAD_DIAGNOSTIC;
    }

    /** Human-readable load/runtime diagnostic for fail-closed startup errors. */
    public static String availabilityDiagnostic() {
        if (!LIBRARY_LOADED) return LOAD_DIAGNOSTIC;
        try {
            if (!nativeRuntimeAvailable()) {
                return "libraphip loaded, but no usable HIP device/runtime is available";
            }
            return "available on " + deviceName();
        } catch (Throwable t) {
            return "libraphip loaded, runtime probe failed: "
                    + t.getClass().getSimpleName() + ": " + String.valueOf(t.getMessage());
        }
    }

    /**
     * Computes one SI-UNIWARD base cost per JPEG block/band-position pair.
     *
     * <p>Output ordering is block-major and band-position-minor, identical to
     * {@code CandidateGenerator}: {@code out[block * bandCount + bandOffset]}.
     * {@code impactMagnitudesPositionMajor} is laid out as
     * {@code [filter][impactY][impactX][bandOffset]}. Only the requested band
     * is transferred, so a 32-lane RDNA wave reads adjacent impact values while
     * all lanes share the same denominator sample.</p>
     */
    public static double[] computeCosts(
            double[] denominators,
            int residualWidth,
            int residualHeight,
            double[] impactMagnitudesPositionMajor,
            int bandCount,
            int widthBlocks,
            int heightBlocks,
            double wetCost,
            int windowOffset
    ) {
        if (!LIBRARY_LOADED) {
            throw new IllegalStateException(
                    "HIP JNI library is not loaded: " + LOAD_DIAGNOSTIC);
        }
        return nativeComputeCosts(
                denominators,
                residualWidth,
                residualHeight,
                impactMagnitudesPositionMajor,
                bandCount,
                widthBlocks,
                heightBlocks,
                wetCost,
                windowOffset);
    }

    private static native boolean nativeRuntimeAvailable();

    private static native String nativeDeviceName();

    private static native double[] nativeComputeCosts(
            double[] denominators,
            int residualWidth,
            int residualHeight,
            double[] impactMagnitudesPositionMajor,
            int bandCount,
            int widthBlocks,
            int heightBlocks,
            double wetCost,
            int windowOffset);
}
