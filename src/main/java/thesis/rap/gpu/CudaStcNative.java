package thesis.rap.gpu;


public final class CudaStcNative {
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

    private CudaStcNative() {}

    public static boolean isAvailable() {
        if (!LIBRARY_LOADED || CudaCrashGuard.isPoisoned()) return false;
        try {
            return CudaCrashGuard.call("CudaStcNative_RUNTIME_PROBE", CudaStcNative::nativeRuntimeAvailable);
        } catch (Throwable ignored) {
            return false;
        }
    }

    public static String diagnostic() {
        if (!LIBRARY_LOADED) return LOAD_DIAGNOSTIC;
        if (CudaCrashGuard.isPoisoned()) return CudaCrashGuard.diagnostic();
        try {
            return nativeRuntimeAvailable()
                    ? "CUDA STC runtime available"
                    : "librapcuda loaded but CUDA STC runtime unavailable";
        } catch (Throwable t) {
            return t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }

    public static byte[] embedBinary(
            byte[] coverBits,
            double[] costs,
            byte[] messageBits,
            int constraintHeight) {
        return CudaCrashGuard.call("STC_EMBED",
                () -> nativeEmbedBinary(coverBits, costs, messageBits, constraintHeight));
    }

    public static byte[] extractBinary(
            byte[] stegoBits,
            int messageBitLength,
            int constraintHeight) {
        return CudaCrashGuard.call("STC_EXTRACT",
                () -> nativeExtractBinary(stegoBits, messageBitLength, constraintHeight));
    }

    private static native boolean nativeRuntimeAvailable();
    private static native byte[] nativeEmbedBinary(
            byte[] coverBits,
            double[] costs,
            byte[] messageBits,
            int constraintHeight);
    private static native byte[] nativeExtractBinary(
            byte[] stegoBits,
            int messageBitLength,
            int constraintHeight);
}
