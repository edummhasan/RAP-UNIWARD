package thesis.rap.runtime;

import java.util.Locale;


public final class ExecutionBackend {
    public enum Mode { AUTO, CPU, CUDA, HIP }

    private ExecutionBackend() {}

    /** True when the process-wide backend selector was explicitly configured. */
    public static boolean explicitlyConfigured() {
        String property = System.getProperty("rap.backend");
        String env = System.getenv("RAP_BACKEND");
        return (property != null && !property.isBlank())
                || (env != null && !env.isBlank());
    }

    public static Mode mode() {
        String value = System.getProperty("rap.backend");
        if (value == null || value.isBlank()) value = System.getenv("RAP_BACKEND");
        if (value == null || value.isBlank()) return Mode.AUTO;
        try {
            return Mode.valueOf(value.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException bad) {
            throw new IllegalArgumentException(
                    "RAP_BACKEND/rap.backend must be AUTO, CPU, CUDA, or HIP; got " + value,
                    bad);
        }
    }

    public static boolean cpuForced() { return explicitlyConfigured() && mode() == Mode.CPU; }
    public static boolean cudaPermitted() {
        if (!explicitlyConfigured()) return true; // per-stage selectors remain valid
        Mode mode = mode();
        return mode == Mode.AUTO || mode == Mode.CUDA;
    }
    public static boolean hipPermitted() {
        if (!explicitlyConfigured()) return true;
        Mode mode = mode();
        return mode == Mode.AUTO || mode == Mode.HIP;
    }

    /** Explicit CUDA/HIP is fail-closed unless this opt-in is true. AUTO always falls back. */
    public static boolean explicitAcceleratorFallbackEnabled() {
        return Boolean.parseBoolean(System.getProperty(
                "rap.accelerator.fallback",
                System.getenv().getOrDefault("RAP_ACCELERATOR_FALLBACK", "false")));
    }

    public static boolean cudaFailureIsFatal() {
        if (!explicitlyConfigured()) return stageStrictGpu();
        return mode() == Mode.CUDA && !explicitAcceleratorFallbackEnabled();
    }

    public static boolean hipFailureIsFatal() {
        if (!explicitlyConfigured()) return stageStrictGpu();
        return mode() == Mode.HIP && !explicitAcceleratorFallbackEnabled();
    }

    public static boolean anyAcceleratorFailureIsFatal() {
        return cudaFailureIsFatal() || hipFailureIsFatal();
    }

    public static String description() {
        return "mode=" + mode()
                + ",explicit=" + explicitlyConfigured()
                + ",explicitFallback=" + explicitAcceleratorFallbackEnabled();
    }

    private static boolean stageStrictGpu() {
        return Boolean.parseBoolean(System.getProperty(
                "rap.true.gpu.strict",
                System.getenv().getOrDefault("RAP_TRUE_GPU_STRICT", "false")));
    }
}
