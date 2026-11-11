package thesis.rap.app;

import java.util.List;

/**
 * Immutable records shared by the six-algorithm execution and evaluation logs.
 * Final Algorithm-4 selection is implemented only by {@link AdaptiveQfSearch}.
 */
public final class AdaptiveQFSelector {
    private AdaptiveQFSelector() {}

    public record SelectionConstraints(String policyName, double maxCandidateLoading, double maxStcUtilization) {
        public SelectionConstraints {
            if (policyName == null || policyName.isBlank()) policyName = "capacity";
            if (!(maxCandidateLoading > 0.0)) maxCandidateLoading = Double.POSITIVE_INFINITY;
            if (!(maxStcUtilization > 0.0)) maxStcUtilization = Double.POSITIVE_INFINITY;
        }

        public static SelectionConstraints capacityOnly() {
            return new SelectionConstraints("capacity", Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        }
    }

    public record QFDecision(int qf, int candidates, int engineeringCapacityBits, boolean fits,
                             List<QfBandCoefficientReport.Row> qfBandRows) {}
}
