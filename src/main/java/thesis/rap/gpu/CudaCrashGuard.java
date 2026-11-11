package thesis.rap.gpu;

import java.util.Locale;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/**
 * Process-wide CUDA admission/circuit-breaker guard.
 *
 * <p>This class changes only CUDA execution scheduling and failure containment.
 * It does not alter any RAP, SI-UNIWARD, STC, JPEG geometry, QF, payload,
 * statistical, cryptographic or carrier-order mathematics.</p>
 *
 * <p>The conservative default is one in-flight JNI CUDA call across all CUDA
 * stages.  A target host may raise {@code RAP_CUDA_GLOBAL_INFLIGHT} only after
 * concurrency qualification.  A terminal CUDA execution fault permanently
 * opens the circuit for the lifetime of the JVM so later stages cannot reuse a
 * poisoned CUDA context.</p>
 */
final class CudaCrashGuard {
    private static final int GLOBAL_INFLIGHT = readPositiveInt(
            "RAP_CUDA_GLOBAL_INFLIGHT", 1, 64);
    private static final Semaphore GATE = new Semaphore(GLOBAL_INFLIGHT, true);
    private static final AtomicBoolean POISONED = new AtomicBoolean(false);
    private static final AtomicReference<String> FIRST_FATAL = new AtomicReference<>();

    private CudaCrashGuard() {}

    static boolean isPoisoned() {
        return POISONED.get();
    }

    static String diagnostic() {
        String first = FIRST_FATAL.get();
        return first == null
                ? "healthy globalInflight=" + GLOBAL_INFLIGHT
                : "CUDA context disabled after fatal device error: " + first;
    }

    static <T> T call(String stage, Supplier<T> action) {
        if (POISONED.get()) throw poisonedException(stage);
        GATE.acquireUninterruptibly();
        try {
            if (POISONED.get()) throw poisonedException(stage);
            try {
                return action.get();
            } catch (RuntimeException | Error failure) {
                if (isTerminalCudaFailure(failure)) {
                    String message = stage + ": " + failure.getClass().getSimpleName()
                            + ": " + String.valueOf(failure.getMessage());
                    FIRST_FATAL.compareAndSet(null, message);
                    POISONED.set(true);
                }
                throw failure;
            }
        } finally {
            GATE.release();
        }
    }

    private static IllegalStateException poisonedException(String stage) {
        return new IllegalStateException(
                stage + " refused because the CUDA context is already poisoned; "
                        + "restart the JVM/process before using CUDA again. "
                        + diagnostic());
    }

    private static boolean isTerminalCudaFailure(Throwable failure) {
        Throwable current = failure;
        while (current != null) {
            String text = (current.getClass().getName() + " "
                    + String.valueOf(current.getMessage())).toLowerCase(Locale.ROOT);
            if (text.contains("illegal instruction")
                    || text.contains("cudaerrorillegalinstruction")
                    || text.contains("illegal address")
                    || text.contains("cudaerrorillegaladdress")
                    || text.contains("device-side assert")
                    || text.contains("cudaerrorassert")
                    || text.contains("hardware stack")
                    || text.contains("misaligned address")
                    || text.contains("invalid address space")
                    || text.contains("invalid pc")
                    || text.contains("launch failure")
                    || text.contains("unspecified launch failure")
                    || text.matches("(?s).*\\(7(00|02|10|14|15|16|17|18|19)\\).*")) {
                return true;
            }
            current = current.getCause();
        }
        return false;
    }

    private static int readPositiveInt(String environment, int fallback, int maximum) {
        String raw = System.getenv(environment);
        if (raw == null || raw.isBlank()) return fallback;
        try {
            int parsed = Integer.parseInt(raw.trim());
            return parsed >= 1 && parsed <= maximum ? parsed : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }
}
