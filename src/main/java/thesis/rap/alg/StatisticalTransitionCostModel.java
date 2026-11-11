package thesis.rap.alg;

import thesis.rap.gpu.CudaTrueParallelNative;
import thesis.rap.runtime.ExecutionBackend;
import thesis.rap.runtime.GpuStageStats;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Converts cover-derived marginal transition risks into a bounded,
 * dimensionless regularizer on top of the directional SI-UNIWARD cost.
 *
 * <p>The four risk families are normalized jointly across Cb and Cr at one
 * fixed QF using their empirical 95th percentiles.  This preserves genuine
 * cross-channel differences while preventing the units of MSE, SSIM-style
 * loss, KL, or entropy from dominating merely because of scale.  SI-UNIWARD
 * remains the multiplicative anchor; the bounded [1,2] multiplier may still
 * change pairwise candidate ordering and therefore requires held-out ablation.</p>
 */
public final class StatisticalTransitionCostModel {
    public static final String MODEL_VERSION =
            "CBCR-MARGINAL-RISK-V1";
    public static final double REGULARIZER_STRENGTH = 1.0;
    public static final double MSE_WEIGHT = 0.25;
    public static final double SSIM_WEIGHT = 0.25;
    public static final double KL_WEIGHT = 0.25;
    public static final double ENTROPY_WEIGHT = 0.25;
    private static final double SCALE_EPSILON = 1.0e-300;

    private StatisticalTransitionCostModel() {}

    public record Draft(
            int component,
            int blockIndex,
            int blockX,
            int blockY,
            int u,
            int v,
            int k,
            int qRef,
            double rhoPlusBeforeStatistics,
            double rhoMinusBeforeStatistics,
            double roundingError,
            boolean formerFourModeNearHalf,
            double rawDct,
            int quantStep,
            double baseUniwardCost,
            double plusModulationFactor,
            double minusModulationFactor,
            double histogramWeight,
            double zeroCrossingMultiplier,
            double rgbMseEnergyProxy,
            double ssimLossProxy,
            double klRiskPlus,
            double klRiskMinus,
            double entropyRiskPlus,
            double entropyRiskMinus
    ) {
        public Draft {
            if (component != 1 && component != 2) {
                throw new IllegalArgumentException(
                        "Statistical transition draft must be Cb or Cr");
            }
            if (quantStep <= 0) {
                throw new IllegalArgumentException(
                        "quantStep must be positive");
            }
            requireNonnegative(rgbMseEnergyProxy, "rgbMseEnergyProxy");
            requireNonnegative(ssimLossProxy, "ssimLossProxy");
            requireNonnegative(klRiskPlus, "klRiskPlus");
            requireNonnegative(klRiskMinus, "klRiskMinus");
            requireNonnegative(entropyRiskPlus, "entropyRiskPlus");
            requireNonnegative(entropyRiskMinus, "entropyRiskMinus");
        }
    }

    public record Result(
            List<Candidate> candidates,
            StatisticalCostSummary summary
    ) {
        public Result {
            candidates = Collections.unmodifiableList(
                    new ArrayList<>(candidates));
            if (summary == null) {
                throw new IllegalArgumentException("summary cannot be null");
            }
        }
    }

    public static Result finalizeCosts(List<Draft> drafts) {
        if (drafts == null || drafts.isEmpty()) {
            throw new IllegalArgumentException(
                    "Statistical transition drafts cannot be empty");
        }

        int riskCount = 0;
        for (Draft draft : drafts) {
            if (!JpegMath.isWetCost(draft.rhoPlusBeforeStatistics())) riskCount++;
            if (!JpegMath.isWetCost(draft.rhoMinusBeforeStatistics())) riskCount++;
        }
        double[] scratch = new double[riskCount];
        double mseScale = riskScale(drafts, scratch, 0);
        double ssimScale = riskScale(drafts, scratch, 1);
        double klScale = riskScale(drafts, scratch, 2);
        double entropyScale = riskScale(drafts, scratch, 3);

        List<Candidate> candidates = new ArrayList<>(drafts.size());
        for (Draft draft : drafts) {
            double plusMultiplier = multiplier(
                    draft.rgbMseEnergyProxy(), draft.ssimLossProxy(),
                    draft.klRiskPlus(), draft.entropyRiskPlus(),
                    mseScale, ssimScale, klScale, entropyScale);
            double minusMultiplier = multiplier(
                    draft.rgbMseEnergyProxy(), draft.ssimLossProxy(),
                    draft.klRiskMinus(), draft.entropyRiskMinus(),
                    mseScale, ssimScale, klScale, entropyScale);

            double rhoPlus = scaleCost(
                    draft.rhoPlusBeforeStatistics(), plusMultiplier);
            double rhoMinus = scaleCost(
                    draft.rhoMinusBeforeStatistics(), minusMultiplier);

            double preTransitionFloor = JpegMath.SI_COST_FLOOR
                    * draft.histogramWeight();
            double plusFloor = directionalFloor(
                    preTransitionFloor, draft.zeroCrossingMultiplier(),
                    draft.qRef(), 1, plusMultiplier);
            double minusFloor = directionalFloor(
                    preTransitionFloor, draft.zeroCrossingMultiplier(),
                    draft.qRef(), -1, minusMultiplier);
            double scalarFloor = Math.min(plusFloor, minusFloor);

            candidates.add(new Candidate(
                    draft.component(), draft.blockIndex(),
                    draft.blockX(), draft.blockY(), draft.u(), draft.v(),
                    draft.k(), draft.qRef(), rhoPlus, rhoMinus,
                    draft.roundingError(), draft.formerFourModeNearHalf(),
                    draft.rawDct(), draft.quantStep(),
                    draft.baseUniwardCost(),
                    draft.plusModulationFactor(),
                    draft.minusModulationFactor(),
                    draft.histogramWeight(), 1.0,
                    draft.zeroCrossingMultiplier(),
                    draft.rgbMseEnergyProxy(), draft.ssimLossProxy(),
                    draft.klRiskPlus(), draft.klRiskMinus(),
                    draft.entropyRiskPlus(), draft.entropyRiskMinus(),
                    plusMultiplier, minusMultiplier,
                    scalarFloor, plusFloor, minusFloor));
        }

        StatisticalCostSummary summary = new StatisticalCostSummary(
                MODEL_VERSION, REGULARIZER_STRENGTH,
                MSE_WEIGHT, SSIM_WEIGHT, KL_WEIGHT, ENTROPY_WEIGHT,
                mseScale, ssimScale, klScale, entropyScale,
                channelSummary(candidates, 1, "Cb"),
                channelSummary(candidates, 2, "Cr"));
        return new Result(candidates, summary);
    }

    private static double multiplier(
            double mse,
            double ssim,
            double kl,
            double entropy,
            double mseScale,
            double ssimScale,
            double klScale,
            double entropyScale
    ) {
        double risk = MSE_WEIGHT * normalize(mse, mseScale)
                + SSIM_WEIGHT * normalize(ssim, ssimScale)
                + KL_WEIGHT * normalize(kl, klScale)
                + ENTROPY_WEIGHT * normalize(entropy, entropyScale);
        return 1.0 + REGULARIZER_STRENGTH * risk;
    }

    private static double normalize(double value, double scale) {
        if (!(scale > SCALE_EPSILON)) return 0.0;
        return Math.max(0.0, Math.min(1.0, value / scale));
    }

    private static double scaleCost(double cost, double multiplier) {
        if (JpegMath.isWetCost(cost)) return JpegMath.WET_COST;
        double scaled = cost * multiplier;
        return JpegMath.capComputedLegalCost(scaled);
    }

    private static double directionalFloor(
            double baseFloor,
            double zeroMultiplier,
            int qRef,
            int delta,
            double statisticalMultiplier
    ) {
        boolean crossesZero = qRef == 0 || qRef + delta == 0;
        double floor = baseFloor
                * (crossesZero ? zeroMultiplier : 1.0)
                * statisticalMultiplier;
        if (!Double.isFinite(floor) || floor <= 0.0) {
            throw new IllegalArgumentException(
                    "Statistical directional floor must be finite and positive");
        }
        return Math.min(JpegMath.WET_COST - 1.0, floor);
    }

    private static double percentile95(double[] values, int count) {
        if (count <= 0) return 0.0;
        int valid = 0;
        for (int i = 0; i < count; i++) {
            double value = values[i];
            if (Double.isFinite(value) && value >= 0.0) values[valid++] = value;
        }
        if (valid == 0) return 0.0;

        if (gpuReductionConfigured() && valid >= 4096) {
            double[] compact = Arrays.copyOf(values, valid);
            try {
                double scale = CudaTrueParallelNative.percentile95(compact);
                GpuStageStats.gpu(GpuStageStats.Stage.P95);
                return scale;
            } catch (RuntimeException | LinkageError failure) {
                if (strictGpu()) throw failure;
                GpuStageStats.fallback(GpuStageStats.Stage.P95, failure);
                // Exact CPU nearest-rank fallback below.
            }
        }
        Arrays.sort(values, 0, valid);
        int index = Math.max(0, Math.min(valid - 1,
                (int) Math.ceil(0.95 * valid) - 1));
        return values[index];
    }

    private static boolean gpuReductionConfigured() {
        return ExecutionBackend.cudaPermitted()
                && Boolean.parseBoolean(System.getProperty(
                "rap.gpu.reductions",
                System.getenv().getOrDefault("RAP_GPU_REDUCTIONS", "true")))
                && CudaTrueParallelNative.isAvailable();
    }

    private static boolean strictGpu() {
        return ExecutionBackend.cudaFailureIsFatal();
    }

    private static double riskScale(List<Draft> drafts, double[] scratch, int metric) {
        int count = 0;
        for (Draft draft : drafts) {
            for (int direction = 0; direction < 2; direction++) {
                boolean plus = direction == 0;
                if (JpegMath.isWetCost(plus ? draft.rhoPlusBeforeStatistics()
                        : draft.rhoMinusBeforeStatistics())) continue;
                scratch[count++] = switch (metric) {
                    case 0 -> draft.rgbMseEnergyProxy();
                    case 1 -> draft.ssimLossProxy();
                    case 2 -> plus ? draft.klRiskPlus() : draft.klRiskMinus();
                    case 3 -> plus ? draft.entropyRiskPlus() : draft.entropyRiskMinus();
                    default -> throw new IllegalArgumentException("Unknown risk metric");
                };
            }
        }
        return percentile95(scratch, count);
    }

    private static StatisticalCostSummary.Channel channelSummary(
            List<Candidate> candidates,
            int component,
            String name
    ) {
        int universe = 0;
        int dry = 0;
        int plusLegal = 0;
        int minusLegal = 0;
        double baseSum = 0.0; int baseCount = 0;
        double mseSum = 0.0; int mseCount = 0;
        double ssimSum = 0.0; int ssimCount = 0;
        double klSum = 0.0; int klCount = 0;
        double entropySum = 0.0; int entropyCount = 0;
        double multiplierSum = 0.0; int multiplierCount = 0;
        int eligible = 0;
        for (Candidate c : candidates) {
            if (c.component == component && !c.fullyWet()) eligible++;
        }
        double[] finalCosts = new double[eligible];
        int finalCount = 0;

        for (Candidate candidate : candidates) {
            if (candidate.component != component) continue;
            universe++;
            if (!candidate.fullyWet()) {
                dry++;
                finalCosts[finalCount++] = candidate.cost;
            }
            if (candidate.plusLegal()) {
                plusLegal++;
                if (Double.isFinite(candidate.klRiskPlus)) {
                    klSum += candidate.klRiskPlus; klCount++;
                }
                if (Double.isFinite(candidate.entropyRiskPlus)) {
                    entropySum += candidate.entropyRiskPlus; entropyCount++;
                }
                if (Double.isFinite(candidate.statisticalMultiplierPlus)) {
                    multiplierSum += candidate.statisticalMultiplierPlus; multiplierCount++;
                }
            }
            if (candidate.minusLegal()) {
                minusLegal++;
                if (Double.isFinite(candidate.klRiskMinus)) {
                    klSum += candidate.klRiskMinus; klCount++;
                }
                if (Double.isFinite(candidate.entropyRiskMinus)) {
                    entropySum += candidate.entropyRiskMinus; entropyCount++;
                }
                if (Double.isFinite(candidate.statisticalMultiplierMinus)) {
                    multiplierSum += candidate.statisticalMultiplierMinus; multiplierCount++;
                }
            }
            if (Double.isFinite(candidate.baseUniwardCost)
                    && candidate.baseUniwardCost >= 0.0) {
                baseSum += candidate.baseUniwardCost; baseCount++;
            }
            if (Double.isFinite(candidate.rgbMseEnergyProxy)) {
                mseSum += candidate.rgbMseEnergyProxy; mseCount++;
            }
            if (Double.isFinite(candidate.ssimLossProxy)) {
                ssimSum += candidate.ssimLossProxy; ssimCount++;
            }
        }
        Arrays.sort(finalCosts, 0, finalCount);
        double median = finalCount == 0
                ? Double.NaN
                : ((finalCount & 1) == 1
                ? finalCosts[finalCount / 2]
                : 0.5 * (finalCosts[finalCount / 2 - 1]
                + finalCosts[finalCount / 2]));
        return new StatisticalCostSummary.Channel(
                component, name, universe, dry, plusLegal, minusLegal,
                mean(baseSum, baseCount), mean(mseSum, mseCount),
                mean(ssimSum, ssimCount), mean(klSum, klCount),
                mean(entropySum, entropyCount),
                mean(multiplierSum, multiplierCount), median);
    }

    private static double mean(double sum, int count) {
        return count == 0 ? Double.NaN : sum / count;
    }

    private static void requireNonnegative(double value, String label) {
        if (!Double.isFinite(value) || value < 0.0) {
            throw new IllegalArgumentException(
                    label + " must be finite and nonnegative");
        }
    }
}
