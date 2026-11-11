package thesis.rap.gpu;

/** CUDA helper for the per-attempt three-state balance cost/direction plan. */
public final class CudaBalanceNative {
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

    private CudaBalanceNative() {}

    public static boolean isAvailable() {
        if (!LIBRARY_LOADED || CudaCrashGuard.isPoisoned()) return false;
        try { return CudaCrashGuard.call("CudaBalanceNative_RUNTIME_PROBE", CudaBalanceNative::nativeRuntimeAvailable); }
        catch (Throwable ignored) { return false; }
    }

    public static String diagnostic() {
        if (!LIBRARY_LOADED) return LOAD_DIAGNOSTIC;
        if (CudaCrashGuard.isPoisoned()) return CudaCrashGuard.diagnostic();
        try {
            return nativeRuntimeAvailable()
                    ? "CUDA balance-plan runtime available"
                    : "librapcuda loaded but CUDA balance-plan runtime unavailable";
        } catch (Throwable t) {
            return t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    /**
     * Returns four concatenated double vectors of length N:
     * adjustedPlus, adjustedMinus, scalarMin, direction(-1/+1 encoded exactly).
     */
    public static double[] compute(
            double[] rhoPlus,
            double[] rhoMinus,
            double[] minimumPlus,
            double[] minimumMinus,
            int[] zeroFlowPlus,
            int[] zeroFlowMinus,
            int[] tieDirections,
            double multiplier,
            double wetCost
    ) {
        int n = rhoPlus == null ? -1 : rhoPlus.length;
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "CUDA balance-plan runtime unavailable: " + diagnostic());
        }
        if (n < 0 || rhoMinus == null || minimumPlus == null
                || minimumMinus == null || zeroFlowPlus == null
                || zeroFlowMinus == null || tieDirections == null
                || rhoMinus.length != n || minimumPlus.length != n
                || minimumMinus.length != n || zeroFlowPlus.length != n
                || zeroFlowMinus.length != n || tieDirections.length != n) {
            throw new IllegalArgumentException("CUDA balance input dimensions mismatch");
        }
        double[] result = CudaCrashGuard.call("BALANCE_PLAN", () -> nativeCompute(
                rhoPlus, rhoMinus, minimumPlus, minimumMinus,
                zeroFlowPlus, zeroFlowMinus, tieDirections,
                multiplier, wetCost));
        if (result == null || result.length != Math.multiplyExact(n, 4)) {
            throw new IllegalStateException(
                    "CUDA balance-plan result dimensions mismatch");
        }
        return result;
    }

    private static native boolean nativeRuntimeAvailable();
    private static native double[] nativeCompute(
            double[] rhoPlus,
            double[] rhoMinus,
            double[] minimumPlus,
            double[] minimumMinus,
            int[] zeroFlowPlus,
            int[] zeroFlowMinus,
            int[] tieDirections,
            double multiplier,
            double wetCost);
}
