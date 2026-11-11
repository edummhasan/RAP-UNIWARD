package thesis.rap.app;

/** Actual measured result of one native STC embedding call. */
public record StcEmbeddingTrace(
        int coverElementCount,
        int messageBitCount,
        int changedElementCount,
        double weightedDistortion
) {
    public double relativePayload() {
        return coverElementCount == 0 ? Double.NaN : messageBitCount / (double) coverElementCount;
    }

    public double changeRate() {
        return coverElementCount == 0 ? Double.NaN : changedElementCount / (double) coverElementCount;
    }

    public double embeddingEfficiencyBitsPerChange() {
        return changedElementCount == 0 ? Double.POSITIVE_INFINITY : messageBitCount / (double) changedElementCount;
    }
}
