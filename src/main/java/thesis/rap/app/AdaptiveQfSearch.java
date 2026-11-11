package thesis.rap.app;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Algorithm-4 control flow: capacity is only a prefilter and every failed real
 * kernel advances to the next public QF.
 */
public final class AdaptiveQfSearch {
    private AdaptiveQfSearch() {}

    public enum Status {
        CAPACITY_REJECTED,
        TRIAL_REJECTED,
        VERIFIED
    }

    public record Trial<T>(int qf, Status status, T value, String reason) {
        public Trial {
            if (status == Status.VERIFIED && value == null) {
                throw new IllegalArgumentException("A verified trial requires a value");
            }
            if (status != Status.VERIFIED && value != null) {
                throw new IllegalArgumentException("A failed trial cannot carry a result value");
            }
            reason = reason == null ? "" : reason;
        }

        public static <T> Trial<T> capacityRejected(
                int qf, String reason
        ) {
            return new Trial<>(
                    qf, Status.CAPACITY_REJECTED, null, reason);
        }

        public static <T> Trial<T> trialRejected(
                int qf, String reason
        ) {
            return new Trial<>(
                    qf, Status.TRIAL_REJECTED, null, reason);
        }

        public static <T> Trial<T> verified(int qf, T value) {
            return new Trial<>(qf, Status.VERIFIED, value, "");
        }
    }

    @FunctionalInterface
    public interface TrialExecutor<T> {
        Trial<T> execute(int qf) throws Exception;
    }

    public record Result<T>(T value, Integer selectedQf, List<Trial<T>> trials) {
        public Result {
            trials = Collections.unmodifiableList(new ArrayList<>(trials));
        }

        public boolean verified() {
            return value != null && selectedQf != null;
        }
    }

    public static <T> Result<T> firstVerified(
            int minQf, int maxQf, TrialExecutor<T> executor
    ) throws Exception {
        if (minQf > maxQf) throw new IllegalArgumentException("minQf exceeds maxQf");
        if (executor == null) throw new IllegalArgumentException("executor cannot be null");
        List<Trial<T>> trials = new ArrayList<>();
        for (int qf = minQf; qf <= maxQf; qf++) {
            Trial<T> trial = executor.execute(qf);
            if (trial == null || trial.qf() != qf) {
                throw new IllegalStateException("QF trial executor returned an invalid trial");
            }
            trials.add(trial);
            if (trial.status() == Status.VERIFIED) {
                return new Result<>(trial.value(), qf, trials);
            }
        }
        return new Result<>(null, null, trials);
    }
}
