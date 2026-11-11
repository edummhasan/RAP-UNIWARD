package thesis.rap.gpu;

/** Optional JNI binding for the NVIDIA CUDA SI-UNIWARD binary64 batch kernel. */
public final class CudaSiUniwardNative {
    private static final boolean LIBRARY_LOADED;
    private static final String LOAD_DIAGNOSTIC;

    static {
        boolean loaded = false;
        String diagnostic = "not attempted";
        try {
            System.loadLibrary("rapcuda");
            loaded = true;
            diagnostic = "librapcuda loaded";
        } catch (Throwable t) {
            diagnostic = t.getClass().getSimpleName() + ": " + String.valueOf(t.getMessage());
        }
        LIBRARY_LOADED = loaded;
        LOAD_DIAGNOSTIC = diagnostic;
    }

    private CudaSiUniwardNative() {}

    public static boolean libraryLoaded() { return LIBRARY_LOADED; }

    public static boolean isAvailable() {
        if (!LIBRARY_LOADED || CudaCrashGuard.isPoisoned()) return false;
        try { return CudaCrashGuard.call("CudaSiUniwardNative_RUNTIME_PROBE", CudaSiUniwardNative::nativeRuntimeAvailable); }
        catch (Throwable ignored) { return false; }
    }

    public static String deviceName() {
        if (!LIBRARY_LOADED) return "unavailable (" + LOAD_DIAGNOSTIC + ")";
        if (CudaCrashGuard.isPoisoned()) return "unavailable (" + CudaCrashGuard.diagnostic() + ")";
        try {
            String name = nativeDeviceName();
            return name == null || name.isBlank() ? "unknown CUDA device" : name;
        } catch (Throwable t) {
            return "unavailable (" + t.getClass().getSimpleName() + ": "
                    + String.valueOf(t.getMessage()) + ")";
        }
    }

    public static String loadDiagnostic() { return LOAD_DIAGNOSTIC; }

    /** Number of independent non-blocking CUDA stream workspaces. */
    public static int streamPoolSize() {
        if (!LIBRARY_LOADED || CudaCrashGuard.isPoisoned()) return 0;
        try { return nativeStreamPoolSize(); }
        catch (Throwable ignored) { return 0; }
    }

    /** Human-readable load/runtime diagnostic for fail-closed startup errors. */
    public static String availabilityDiagnostic() {
        if (!LIBRARY_LOADED) return LOAD_DIAGNOSTIC;
        if (CudaCrashGuard.isPoisoned()) return CudaCrashGuard.diagnostic();
        try {
            if (!nativeRuntimeAvailable()) {
                return "librapcuda loaded, but no usable CUDA device/runtime is available";
            }
            return "available on " + deviceName();
        } catch (Throwable t) {
            return "librapcuda loaded, runtime probe failed: "
                    + t.getClass().getSimpleName() + ": " + String.valueOf(t.getMessage());
        }
    }

    public static double[] computeCosts(
            double[] denominators, int residualWidth, int residualHeight,
            double[] impactMagnitudesPositionMajor, int bandCount,
            int widthBlocks, int heightBlocks, double wetCost, int windowOffset) {
        if (!LIBRARY_LOADED) {
            throw new IllegalStateException(
                    "CUDA JNI library is not loaded: " + LOAD_DIAGNOSTIC);
        }
        return CudaCrashGuard.call("SI_UNIWARD", () -> nativeComputeCosts(
                denominators, residualWidth, residualHeight,
                impactMagnitudesPositionMajor, bandCount, widthBlocks,
                heightBlocks, wetCost, windowOffset));
    }

    private static native boolean nativeRuntimeAvailable();
    private static native String nativeDeviceName();
    private static native int nativeStreamPoolSize();
    private static native double[] nativeComputeCosts(
            double[] denominators, int residualWidth, int residualHeight,
            double[] impactMagnitudesPositionMajor, int bandCount,
            int widthBlocks, int heightBlocks, double wetCost, int windowOffset);
}
