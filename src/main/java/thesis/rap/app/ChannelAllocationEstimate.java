package thesis.rap.app;

import thesis.rap.alg.Candidate;
import thesis.rap.jni.JpegImage;

import java.util.List;

/**
 * Idealized and observed Cb/Cr allocation evidence for the final STC active
 * set.
 *
 * <p>The prediction uses the binary payload-limited Gibbs model because this
 * implementation supplies STC with one unchanged/flip bit per coefficient;
 * the physical +1/-1 flip direction has already been selected by its minimum
 * adjusted directional cost. Header and body have separate entropy constraints
 * because they are separate STC calls. Per-channel entropy contributions are
 * descriptive rate--distortion proxies: an STC syndrome bit is not literally
 * assigned to one channel. Actual counts come from the verified coefficient
 * state.</p>
 */
public record ChannelAllocationEstimate(
        String model,
        double headerGibbsLambda,
        double bodyGibbsLambda,
        int requestedPayloadBits,
        double solvedEntropyBits,
        int activeCb,
        int activeCr,
        double activeCbShare,
        double activeCrShare,
        double cbEntropyContributionBits,
        double crEntropyContributionBits,
        double cbEntropyContributionShare,
        double crEntropyContributionShare,
        double expectedCbChanges,
        double expectedCrChanges,
        double expectedCbChangeShare,
        double expectedCrChangeShare,
        double expectedCbDistortion,
        double expectedCrDistortion,
        int actualModifiedCb,
        int actualModifiedCr,
        double actualModifiedCbShare,
        double actualModifiedCrShare
) {
    public static final String MODEL =
            "SEGMENTED-BINARY-GIBBS-ENTROPY-PROXY-ON-FINAL-STC-COSTS-V3";

    /**
     * Production entry point. Header and body are separate STC problems, so
     * their descriptive Gibbs multipliers must be solved separately as well.
     */
    public static ChannelAllocationEstimate analyze(
            List<Candidate> active,
            double[] adjustedCosts,
            JpegImage intendedStego,
            StcCarrierPlan plan
    ) {
        if (plan == null) {
            throw new IllegalArgumentException("carrier plan cannot be null");
        }
        return analyzeSegments(
                active, adjustedCosts, intendedStego,
                plan.headerCarrierCount(), plan.headerMessageBits(),
                plan.bodyCarrierCount(), plan.bodyMessageBits());
    }

    /**
     * Single-segment convenience entry point retained for focused model tests.
     * Production embedding uses {@link #analyze(List, double[], JpegImage,
     * StcCarrierPlan)} so the header/body boundary is not discarded.
     */
    public static ChannelAllocationEstimate analyze(
            List<Candidate> active,
            double[] adjustedCosts,
            JpegImage intendedStego,
            int payloadBits
    ) {
        int carrierCount = active == null ? 0 : active.size();
        return analyzeSegments(
                active, adjustedCosts, intendedStego,
                carrierCount, payloadBits, 0, 0);
    }

    private static ChannelAllocationEstimate analyzeSegments(
            List<Candidate> active,
            double[] adjustedCosts,
            JpegImage intendedStego,
            int headerCarrierCount,
            int headerPayloadBits,
            int bodyCarrierCount,
            int bodyPayloadBits
    ) {
        if (active == null || adjustedCosts == null
                || intendedStego == null) {
            throw new IllegalArgumentException(
                    "Allocation inputs cannot be null");
        }
        if (headerCarrierCount < 0 || bodyCarrierCount < 0
                || headerPayloadBits < 0 || bodyPayloadBits < 0
                || (long) headerCarrierCount + bodyCarrierCount
                        != active.size()
                || active.size() != adjustedCosts.length
                || headerPayloadBits > headerCarrierCount
                || bodyPayloadBits > bodyCarrierCount
                || (headerCarrierCount == 0) != (headerPayloadBits == 0)
                || (bodyCarrierCount == 0) != (bodyPayloadBits == 0)) {
            throw new IllegalArgumentException(
                    "Invalid segmented active-cost/payload dimensions");
        }
        for (double cost : adjustedCosts) {
            if (!Double.isFinite(cost) || cost <= 0.0) {
                throw new IllegalArgumentException(
                        "Final STC scalar costs must be finite and positive");
            }
        }

        int requestedPayloadBits = Math.addExact(
                headerPayloadBits, bodyPayloadBits);
        if (requestedPayloadBits <= 0) {
            throw new IllegalArgumentException(
                    "At least one payload bit is required");
        }
        double headerLambda = solveLambda(
                adjustedCosts, 0, headerCarrierCount,
                headerPayloadBits);
        double bodyLambda = bodyCarrierCount == 0
                ? Double.NaN
                : solveLambda(
                        adjustedCosts, headerCarrierCount,
                        bodyCarrierCount, bodyPayloadBits);
        double totalBits = 0.0;
        double cbBits = 0.0;
        double crBits = 0.0;
        double cbChanges = 0.0;
        double crChanges = 0.0;
        double cbDistortion = 0.0;
        double crDistortion = 0.0;
        int activeCb = 0;
        int activeCr = 0;
        int modifiedCb = 0;
        int modifiedCr = 0;

        for (int i = 0; i < active.size(); i++) {
            Candidate candidate = active.get(i);
            double lambda = i < headerCarrierCount
                    ? headerLambda : bodyLambda;
            double probability = flipProbability(
                    lambda, adjustedCosts[i]);
            double entropy = binaryEntropy(probability);
            totalBits += entropy;
            int actual = intendedStego.component(candidate.component)
                    .getCoeff(candidate.blockIndex, candidate.k);
            boolean modified = actual != candidate.qRef;
            if (candidate.component == 1) {
                activeCb++;
                cbBits += entropy;
                cbChanges += probability;
                cbDistortion += probability * adjustedCosts[i];
                if (modified) modifiedCb++;
            } else if (candidate.component == 2) {
                activeCr++;
                crBits += entropy;
                crChanges += probability;
                crDistortion += probability * adjustedCosts[i];
                if (modified) modifiedCr++;
            } else {
                throw new IllegalStateException(
                        "Active carrier is outside Cb/Cr");
            }
        }

        int activeTotal = activeCb + activeCr;
        double expectedChanges = cbChanges + crChanges;
        int modifiedTotal = modifiedCb + modifiedCr;
        return new ChannelAllocationEstimate(
                MODEL, headerLambda, bodyLambda,
                requestedPayloadBits, totalBits,
                activeCb, activeCr,
                ratio(activeCb, activeTotal), ratio(activeCr, activeTotal),
                cbBits, crBits,
                ratio(cbBits, totalBits), ratio(crBits, totalBits),
                cbChanges, crChanges,
                ratio(cbChanges, expectedChanges),
                ratio(crChanges, expectedChanges),
                cbDistortion, crDistortion,
                modifiedCb, modifiedCr,
                ratio(modifiedCb, modifiedTotal),
                ratio(modifiedCr, modifiedTotal));
    }

    private static double solveLambda(
            double[] costs,
            int from,
            int count,
            int payloadBits
    ) {
        if (from < 0 || count <= 0 || from + count > costs.length
                || payloadBits <= 0 || payloadBits > count) {
            throw new IllegalArgumentException(
                    "Invalid Gibbs segment dimensions");
        }
        if (payloadBits == count) return 0.0;
        double low = 0.0;
        double high = 1.0;
        int expansions = 0;
        while (totalEntropy(costs, from, count, high) > payloadBits) {
            high *= 2.0;
            expansions++;
            if (!Double.isFinite(high) || expansions > 1_024) {
                throw new IllegalStateException(
                        "Could not bracket Gibbs payload multiplier");
            }
        }
        for (int iteration = 0; iteration < 160; iteration++) {
            double middle = low + 0.5 * (high - low);
            if (totalEntropy(costs, from, count, middle) > payloadBits) {
                low = middle;
            }
            else high = middle;
        }
        return low + 0.5 * (high - low);
    }

    private static double totalEntropy(
            double[] costs, int from, int count, double lambda
    ) {
        double total = 0.0;
        int to = from + count;
        for (int i = from; i < to; i++) {
            total += binaryEntropy(
                    flipProbability(lambda, costs[i]));
        }
        return total;
    }

    private static double flipProbability(double lambda, double cost) {
        double exponent = lambda * cost;
        if (exponent > 40.0) {
            double small = StrictMath.exp(-exponent);
            return small / (1.0 + small);
        }
        return 1.0 / (1.0 + StrictMath.exp(exponent));
    }

    private static double binaryEntropy(double probability) {
        if (probability <= 0.0 || probability >= 1.0) return 0.0;
        return -(probability * log2(probability)
                + (1.0 - probability) * log2(1.0 - probability));
    }

    private static double log2(double value) {
        return StrictMath.log(value) / StrictMath.log(2.0);
    }

    private static double ratio(double numerator, double denominator) {
        return denominator == 0.0 ? Double.NaN : numerator / denominator;
    }
}
