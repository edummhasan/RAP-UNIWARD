package thesis.rap.analysis;

/** Target-band detector-facing metrics for one chrominance component. */
public final class ChannelMetrics {
    public int componentIndex;
    public String channel;
    public long totalCoefficients;
    public long coverNonzero;
    public long stegoNonzero;
    public long coverZero;
    public long stegoZero;
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
}
