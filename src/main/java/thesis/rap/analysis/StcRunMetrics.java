package thesis.rap.analysis;

import thesis.rap.app.StcCarrierPlan;
import thesis.rap.app.StcEmbeddingTrace;

/** Actual run-level STC dimensions and observed parity-change statistics. */
public record StcRunMetrics(
        int constraintHeight,
        StcCarrierPlan plan,
        StcEmbeddingTrace header,
        StcEmbeddingTrace body
) {
    public int totalChangedElements() {
        return header.changedElementCount() + body.changedElementCount();
    }

    public double totalWeightedDistortion() {
        return header.weightedDistortion() + body.weightedDistortion();
    }

    public double overallRelativePayload() {
        return plan.overallRelativePayload();
    }

    public double overallChangeRate() {
        return plan.totalCarrierCount() == 0 ? Double.NaN : totalChangedElements() / (double) plan.totalCarrierCount();
    }

    public double embeddingEfficiencyBitsPerChange() {
        return totalChangedElements() == 0 ? Double.POSITIVE_INFINITY : plan.totalMessageBits() / (double) totalChangedElements();
    }
}
