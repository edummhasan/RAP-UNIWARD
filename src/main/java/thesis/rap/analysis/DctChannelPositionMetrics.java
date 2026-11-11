package thesis.rap.analysis;

/** Detector-facing statistics for one channel and one zero-based target-band zig-zag rank. */
public final class DctChannelPositionMetrics {
    public int componentIndex;
    public String channel;
    public int zzRank;
    public int thesisZzRankOneBased;
    public int u;
    public int v;
    public int naturalIndex;
    public long totalCoefficients;
    public long coverNonzero;
    public long stegoNonzero;
    public long eligibleCandidates;
    public long activeCarriers;
    public long modifiedCoefficients;
    public long plusOneChanges;
    public long minusOneChanges;
    public long otherDeltaChanges;
    public long zeroToNonzero;
    public long nonzeroToZero;
    public double candidateSelectionRate;
    public double modificationRateEligible;
    public double modificationRateActive;
    public double modificationShareAllChanges;
    public double plusMinusImbalance;
    public double meanAbsoluteDelta;
    public double histogramL1;
    public double totalVariation;
    public double klCoverToStego;
    public double symmetricKl;
    public double jensenShannon;
    public double hellinger;
    public double horizontalCorrelationCover;
    public double horizontalCorrelationStego;
    public double horizontalCorrelationDelta;
    public double verticalCorrelationCover;
    public double verticalCorrelationStego;
    public double verticalCorrelationDelta;
}
