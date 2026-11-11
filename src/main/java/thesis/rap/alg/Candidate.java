package thesis.rap.alg;

/** One ordered RAP-UNIWARD carrier candidate and its two parity-flip costs. */
public final class Candidate implements Comparable<Candidate> {
    public final int component; // 1=Cb, 2=Cr
    public final int blockIndex;
    public final int blockX;
    public final int blockY;
    public final int u;
    public final int v;
    public final int k;
    public final int qRef;
    public final double rhoPlus;
    public final double rhoMinus;
    public final double cost; // minimum legal cost to flip coefficient parity
    public final double roundingError;
    /**
     * Literature-comparison flag for the former four-mode near-half condition.
     * It is diagnostic only and never changes legality or costs.
     */
    public final boolean formerFourModeNearHalf;
    /** Unquantized forward-DCT value before division by the quantization step. */
    public final double rawDct;
    /** Positive canonical quantization step at this natural coefficient index. */
    public final int quantStep;
    /** Symmetric SI-UNIWARD unit-impact cost before RAP directional modulation. */
    public final double baseUniwardCost;
    /** Multipliers applied to the SI cost before the positive pre-weight offset. */
    public final double plusModulationFactor;
    public final double minusModulationFactor;
    /** RAP context factors retained for reproducibility and candidate-level audit. */
    public final double histogramWeight;
    
    public final double channelWeight;
    public final double zeroCrossingMultiplier;
    /** Cover-derived marginal risks used by the statistical regularizer. */
    public final double rgbMseEnergyProxy;
    public final double ssimLossProxy;
    public final double klRiskPlus;
    public final double klRiskMinus;
    public final double entropyRiskPlus;
    public final double entropyRiskMinus;
    /** Dimensionless multipliers applied after robust cross-channel normalization. */
    public final double statisticalMultiplierPlus;
    public final double statisticalMultiplierMinus;
    /**
     * Lower clipping bound used by the additive B/S multiplier.  This is the
     * positive pre-weight offset after histogram and channel scaling, as required
     * by the normative methodology.
     *
     * <p>Retained as the smaller of the two directional floors so that existing
     * audit columns stay comparable; the balance loop must clip each direction
     * with {@link #minimumAdjustedCost(int)}, which additionally carries the
     * zero-crossing multiplier lambda_i(delta) required by the methodology's
     * rho_min^(0)(delta) = lambda_i(delta) * 1e-4 * w_h * w_c.</p>
     */
    public final double minimumAdjustedCost;

    /** Direction-specific balance floor for delta = +1. */
    public final double minimumAdjustedPlus;

    /** Direction-specific balance floor for delta = -1. */
    public final double minimumAdjustedMinus;

    /** Backward-compatible constructor for tests/helpers that do not attach SI metadata. */
    public Candidate(int component, int blockIndex, int blockX, int blockY,
                     int u, int v, int k, int qRef,
                     double rhoPlus, double rhoMinus) {
        this(component, blockIndex, blockX, blockY, u, v, k, qRef,
                rhoPlus, rhoMinus, 0.0, false,
                Double.NaN, 1, Double.NaN, 1.0, 1.0,
                1.0, 1.0, 1.0, JpegMath.SI_COST_FLOOR);
    }

    public Candidate(int component, int blockIndex, int blockX, int blockY,
                     int u, int v, int k, int qRef,
                     double rhoPlus, double rhoMinus,
                     double roundingError, boolean formerFourModeNearHalf) {
        this(component, blockIndex, blockX, blockY, u, v, k, qRef,
                rhoPlus, rhoMinus, roundingError, formerFourModeNearHalf,
                Double.NaN, 1, Double.NaN,
                directionalFactor(roundingError, 1),
                directionalFactor(roundingError, -1),
                1.0, 1.0, 1.0, JpegMath.SI_COST_FLOOR);
    }

    public Candidate(int component, int blockIndex, int blockX, int blockY,
                     int u, int v, int k, int qRef,
                     double rhoPlus, double rhoMinus,
                     double roundingError, boolean formerFourModeNearHalf,
                     double rawDct, int quantStep, double baseUniwardCost,
                     double plusModulationFactor, double minusModulationFactor,
                     double histogramWeight, double channelWeight,
                     double zeroCrossingMultiplier, double minimumAdjustedCost) {
        this(component, blockIndex, blockX, blockY, u, v, k, qRef,
                rhoPlus, rhoMinus, roundingError, formerFourModeNearHalf,
                rawDct, quantStep, baseUniwardCost,
                plusModulationFactor, minusModulationFactor,
                histogramWeight, channelWeight, zeroCrossingMultiplier,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                minimumAdjustedCost,
                directionalFloor(minimumAdjustedCost,
                        zeroCrossingMultiplier, qRef, 1),
                directionalFloor(minimumAdjustedCost,
                        zeroCrossingMultiplier, qRef, -1));
    }

    public Candidate(int component, int blockIndex, int blockX, int blockY,
                     int u, int v, int k, int qRef,
                     double rhoPlus, double rhoMinus,
                     double roundingError, boolean formerFourModeNearHalf,
                     double rawDct, int quantStep, double baseUniwardCost,
                     double plusModulationFactor, double minusModulationFactor,
                     double histogramWeight, double channelWeight,
                     double zeroCrossingMultiplier,
                     double rgbMseEnergyProxy, double ssimLossProxy,
                     double klRiskPlus, double klRiskMinus,
                     double entropyRiskPlus, double entropyRiskMinus,
                     double statisticalMultiplierPlus,
                     double statisticalMultiplierMinus,
                     double minimumAdjustedCost,
                     double minimumAdjustedPlus,
                     double minimumAdjustedMinus) {
        if (quantStep <= 0) throw new IllegalArgumentException("quantStep must be positive");
        if (!positiveFinite(histogramWeight)
                || !positiveFinite(channelWeight)
                || !positiveFinite(zeroCrossingMultiplier)
                || !positiveFinite(statisticalMultiplierPlus)
                || !positiveFinite(statisticalMultiplierMinus)
                || !positiveFinite(minimumAdjustedCost)
                || !positiveFinite(minimumAdjustedPlus)
                || !positiveFinite(minimumAdjustedMinus)) {
            throw new IllegalArgumentException("Candidate scaling terms must be finite and positive");
        }
        if (!nonnegativeFinite(rgbMseEnergyProxy)
                || !nonnegativeFinite(ssimLossProxy)
                || !nonnegativeFinite(klRiskPlus)
                || !nonnegativeFinite(klRiskMinus)
                || !nonnegativeFinite(entropyRiskPlus)
                || !nonnegativeFinite(entropyRiskMinus)) {
            throw new IllegalArgumentException(
                    "Candidate statistical risks must be finite and nonnegative");
        }
        this.component = component;
        this.blockIndex = blockIndex;
        this.blockX = blockX;
        this.blockY = blockY;
        this.u = u;
        this.v = v;
        this.k = k;
        this.qRef = qRef;
        this.rhoPlus = sanitize(rhoPlus);
        this.rhoMinus = sanitize(rhoMinus);
        this.cost = Math.min(this.rhoPlus, this.rhoMinus);
        this.roundingError = roundingError;
        this.formerFourModeNearHalf = formerFourModeNearHalf;
        this.rawDct = rawDct;
        this.quantStep = quantStep;
        this.baseUniwardCost = baseUniwardCost;
        this.plusModulationFactor = plusModulationFactor;
        this.minusModulationFactor = minusModulationFactor;
        this.histogramWeight = histogramWeight;
        this.channelWeight = channelWeight;
        this.zeroCrossingMultiplier = zeroCrossingMultiplier;
        this.rgbMseEnergyProxy = rgbMseEnergyProxy;
        this.ssimLossProxy = ssimLossProxy;
        this.klRiskPlus = klRiskPlus;
        this.klRiskMinus = klRiskMinus;
        this.entropyRiskPlus = entropyRiskPlus;
        this.entropyRiskMinus = entropyRiskMinus;
        this.statisticalMultiplierPlus = statisticalMultiplierPlus;
        this.statisticalMultiplierMinus = statisticalMultiplierMinus;
        this.minimumAdjustedCost = minimumAdjustedCost;
        this.minimumAdjustedPlus = minimumAdjustedPlus;
        this.minimumAdjustedMinus = minimumAdjustedMinus;
    }

    /**
     * Returns the methodology's rho_min^(0)(delta) for one direction, that is
     * the positive pre-weight offset after histogram and channel scaling and
     * after the zero-crossing multiplier where that multiplier legally applies.
     */
    public double minimumAdjustedCost(int delta) {
        return switch (delta) {
            case 1 -> minimumAdjustedPlus;
            case -1 -> minimumAdjustedMinus;
            default -> throw new IllegalArgumentException("delta must be +1 or -1");
        };
    }

    /** True when {@code delta} moves this coefficient across the zero bin. */
    public boolean crossesZero(int delta) {
        if (delta != 1 && delta != -1) {
            throw new IllegalArgumentException("delta must be +1 or -1");
        }
        return qRef == 0 || qRef + delta == 0;
    }

    private static double directionalFloor(
            double preWeightFloor, double zeroCrossingMultiplier, int qRef, int delta
    ) {
        boolean crossesZero = qRef == 0 || qRef + delta == 0;
        double floor = crossesZero
                ? preWeightFloor * zeroCrossingMultiplier
                : preWeightFloor;
        if (!Double.isFinite(floor) || floor <= 0.0) {
            throw new IllegalArgumentException(
                    "Directional balance floor must be finite and positive");
        }
        return Math.min(JpegMath.WET_COST - 1.0, floor);
    }

    public boolean plusLegal() {
        return !JpegMath.isWetCost(rhoPlus);
    }

    public boolean minusLegal() {
        return !JpegMath.isWetCost(rhoMinus);
    }

    public boolean fullyWet() {
        return !plusLegal() && !minusLegal();
    }

    public int bestParityFlipValue() {
        return qRef + bestParityFlipDelta(rhoPlus, rhoMinus);
    }

    /**
     * Chooses the physical unit direction for one parity flip using the supplied
     * attempt-specific directional costs.  The deterministic tie rule is the same
     * for base and multiplier-adjusted costs.
     */
    public int bestParityFlipDelta(double adjustedPlus, double adjustedMinus) {
        return bestParityFlipDelta(
                adjustedPlus, adjustedMinus, defaultTieDirection());
    }

    /**
     * Chooses the physical unit direction and uses {@code tieDirection} only
     * when both legal adjusted costs are exactly equal.
     */
    public int bestParityFlipDelta(
            double adjustedPlus,
            double adjustedMinus,
            int tieDirection
    ) {
        if (tieDirection != -1 && tieDirection != 1) {
            throw new IllegalArgumentException(
                    "tieDirection must be -1 or +1");
        }
        boolean canMinus = !JpegMath.isWetCost(adjustedMinus);
        boolean canPlus = !JpegMath.isWetCost(adjustedPlus);
        if (!canMinus && !canPlus) {
            throw new IllegalStateException("No legal parity-flip direction for candidate " + describe());
        }
        if (canMinus && canPlus) {
            int comparison = Double.compare(adjustedMinus, adjustedPlus);
            if (comparison < 0) return -1;
            if (comparison > 0) return 1;
            return tieDirection;
        }
        return canMinus ? -1 : 1;
    }

    /** Compatibility tie rule for isolated tests and callers without key context. */
    private int defaultTieDirection() {
        // Every multiplier is odd, so this has the same parity as the former
        // 31-based address hash without depending on signed overflow.
        return Math.floorMod(component + blockIndex + k + qRef, 2) == 0
                ? 1 : -1;
    }

    public double directionalBaseCost(int delta) {
        return switch (delta) {
            case 1 -> rhoPlus;
            case -1 -> rhoMinus;
            default -> throw new IllegalArgumentException("delta must be +1 or -1");
        };
    }

    public int zeroFlowContribution(int delta) {
        int qNew = qRef + delta;
        if (qRef == 0 && qNew != 0) return -1;
        if ((qRef == 1 || qRef == -1) && qNew == 0) return 1;
        return 0;
    }

    public String transitionClass(int delta) {
        int qNew = qRef + delta;
        if (qRef == 0 && qNew == 1) return "ZP";
        if (qRef == 0 && qNew == -1) return "ZN";
        if (qRef == 1 && qNew == 0) return "PZ";
        if (qRef == -1 && qNew == 0) return "NZ";
        if (qRef > 0 && qNew > 0) return "PP";
        if (qRef < 0 && qNew < 0) return "NN";
        throw new IllegalStateException("Unexpected unit transition " + qRef + "->" + qNew);
    }

    public String describe() {
        return "component=" + component + ", block=" + blockIndex + ", ZZ0="
                + JpegMath.zigzagRankFromNaturalIndex(k) + ", (u=" + u + ",v=" + v
                + "), qRef=" + qRef + ", e=" + roundingError;
    }

    private static double sanitize(double v) {
        if (!Double.isFinite(v) || v < 0) return JpegMath.WET_COST;
        return Math.min(v, JpegMath.WET_COST);
    }

    private static boolean positiveFinite(double value) {
        return Double.isFinite(value) && value > 0.0;
    }

    private static boolean nonnegativeFinite(double value) {
        return Double.isFinite(value) && value >= 0.0;
    }

    private static double directionalFactor(double roundingError, int delta) {
        if (!Double.isFinite(roundingError) || roundingError == 0.0) return 1.0;
        boolean preferred = (roundingError > 0.0 && delta > 0)
                || (roundingError < 0.0 && delta < 0);
        return preferred ? Math.max(0.0, 1.0 - 2.0 * Math.abs(roundingError)) : 1.0;
    }

    @Override
    public int compareTo(Candidate o) {
        int byCost = Double.compare(this.cost, o.cost);
        if (byCost != 0) return byCost;
        int byComponent = Integer.compare(this.component, o.component);
        if (byComponent != 0) return byComponent;
        int byBlock = Integer.compare(this.blockIndex, o.blockIndex);
        if (byBlock != 0) return byBlock;
        return Integer.compare(this.k, o.k);
    }
}
