package thesis.rap.runtime;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.LongAdder;


public final class GpuStageStats {
    public enum Stage {
        RGB_TO_YCBCR, RESAMPLE, DCT_VARIANCE, CANONICAL_LUMA, HMAC_ORDER,
        P95, CANDIDATE_SORT, BALANCE_PLAN, MEDIAN, STC
    }

    private static final Map<Stage, LongAdder> GPU = counters();
    private static final Map<Stage, LongAdder> FALLBACK = counters();
    private static final Map<Stage, AtomicBoolean> WARNED = new EnumMap<>(Stage.class);

    static {
        for (Stage stage : Stage.values()) WARNED.put(stage, new AtomicBoolean());
    }

    private GpuStageStats() {}

    public static void gpu(Stage stage) {
        GPU.get(stage).increment();
    }

    /** Records a CPU fallback and reports the first one per stage on stderr. */
    public static void fallback(Stage stage, Throwable cause) {
        FALLBACK.get(stage).increment();
        if (WARNED.get(stage).compareAndSet(false, true)) {
            System.err.println("WARN: GPU stage " + stage + " fell back to the CPU reference"
                    + " (further fallbacks of this stage are only counted): "
                    + (cause == null ? "unknown" : cause.getClass().getSimpleName()
                    + ": " + cause.getMessage()));
        }
    }

    public static long gpuCount(Stage stage) {
        return GPU.get(stage).sum();
    }

    public static long fallbackCount(Stage stage) {
        return FALLBACK.get(stage).sum();
    }

    public static long totalFallbacks() {
        long total = 0;
        for (Stage stage : Stage.values()) total += fallbackCount(stage);
        return total;
    }

    /** One line, e.g. {@code DCT_VARIANCE=gpu:14/cpu_fallback:0 ...}; stages never reached are omitted. */
    public static String summary() {
        StringBuilder out = new StringBuilder();
        for (Stage stage : Stage.values()) {
            long gpu = gpuCount(stage);
            long fallback = fallbackCount(stage);
            if (gpu == 0 && fallback == 0) continue;
            if (out.length() > 0) out.append(' ');
            out.append(stage).append("=gpu:").append(gpu)
                    .append("/cpu_fallback:").append(fallback);
        }
        return out.length() == 0 ? "no_gpu_stage_invoked" : out.toString();
    }

    private static Map<Stage, LongAdder> counters() {
        Map<Stage, LongAdder> map = new EnumMap<>(Stage.class);
        for (Stage stage : Stage.values()) map.put(stage, new LongAdder());
        return map;
    }
}
