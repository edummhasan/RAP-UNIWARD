package thesis.rap.alg;

import thesis.rap.image.ImageUtil;
import thesis.rap.image.SourceImageContext;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.gpu.CudaTrueParallelNative;
import thesis.rap.runtime.ExecutionBackend;
import thesis.rap.runtime.GpuStageStats;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class CandidateGenerator {
    private static final double CB_RGB_ENERGY =
            square(0.344136) + square(1.772);
    private static final double CR_RGB_ENERGY =
            square(1.402) + square(0.714136);
    private static final double SSIM_C2 = square(0.03 * 255.0);

    private CandidateGenerator() {}

    /** Run-scoped candidate preparation state shared across all QF trials. */
    public static final class Context {
        private final SourceImageContext source;
        private final Map<ComponentLayoutKey, SiUniwardCostEngine.PreparedResiduals> residualCache =
                new HashMap<>();

        public Context(SourceImageContext source) {
            if (source == null) throw new IllegalArgumentException("source cannot be null");
            this.source = source;
        }

        public SourceImageContext source() {
            return source;
        }

        private synchronized SiUniwardCostEngine.PreparedResiduals residuals(
                JpegImage ref, int componentIndex, double[][] paddedPlane) {
            ComponentLayoutKey key = ComponentLayoutKey.of(ref, componentIndex);
            return residualCache.computeIfAbsent(
                    key, ignored -> SiUniwardCostEngine.prepareResiduals(paddedPlane));
        }
    }

    private record ComponentLayoutKey(
            int component, int widthBlocks, int heightBlocks,
            int hSampling, int vSampling, int maxH, int maxV
    ) {
        static ComponentLayoutKey of(JpegImage ref, int componentIndex) {
            JpegComponent c = ref.component(componentIndex);
            return new ComponentLayoutKey(
                    componentIndex, c.widthBlocks, c.heightBlocks,
                    c.hSamplingFactor, c.vSamplingFactor,
                    maximumHorizontalSampling(ref), maximumVerticalSampling(ref));
        }
    }

    public static List<Candidate> generate(Path pngPath, JpegImage ref) throws Exception {
        return new ArrayList<>(generateSet(pngPath, ref).dryBaseCostSorted());
    }

    public static CandidateSet generateSet(Path pngPath, JpegImage ref) throws Exception {
        SourceImageContext source = SourceImageContext.load(pngPath);
        return generateSet(new Context(source), ref);
    }

    public static CandidateSet generateSet(Context context, JpegImage ref) {
        if (context == null) throw new IllegalArgumentException("context cannot be null");
        if (ref == null || ref.components.length < 3) {
            throw new IllegalArgumentException(
                    "RAP candidate construction requires Y, Cb, and Cr JPEG components");
        }
        SamplingGeometry.requireSupported(ref);
        ImageUtil.CanonicalLuminanceStats luminanceStats =
                ImageUtil.canonicalReferenceLuminanceStats(ref);
        double[][] referenceLuminance = luminanceStats.luminance();

        StatisticalTransitionCostModel.Result modeled =
                collectCandidates(
                        context, referenceLuminance,
                        luminanceStats.histogram(), ref);
        List<Candidate> universe = modeled.candidates();
        long expectedUniverse = 0;
        for (int component = 1;
             component <= 2 && component < ref.components.length;
             component++) {
            JpegComponent c = ref.component(component);
            expectedUniverse = Math.addExact(
                    expectedUniverse,
                    Math.multiplyExact(
                            (long) c.widthBlocks * c.heightBlocks,
                            JpegMath.TARGET_ZZ_END_RANK
                                    - JpegMath.TARGET_ZZ_START_RANK
                                    + 1L));
        }
        if (universe.size() != expectedUniverse) {
            throw new IllegalStateException(
                    "Candidate universe does not exactly cover every "
                            + "Cb/Cr fixed-band address: expected="
                            + expectedUniverse + ", actual="
                            + universe.size());
        }
        List<Candidate> dry = new ArrayList<>(universe.size());
        for (Candidate candidate : universe) {
            if (!candidate.fullyWet()) dry.add(candidate);
        }
        sortDryCandidates(dry);
        return new CandidateSet(
                universe, dry, modeled.summary());
    }

    private static void sortDryCandidates(List<Candidate> dry) {
        if (dry.size() < 4096 || !gpuCandidateSortConfigured()) {
            Collections.sort(dry);
            return;
        }
        double[] costs = new double[dry.size()];
        int[] fields = new int[Math.multiplyExact(dry.size(), 3)];
        for (int i = 0; i < dry.size(); i++) {
            Candidate candidate = dry.get(i);
            costs[i] = candidate.cost;
            int base = i * 3;
            fields[base] = candidate.component;
            fields[base + 1] = candidate.blockIndex;
            fields[base + 2] = candidate.k;
        }
        try {
            int[] order = CudaTrueParallelNative.sortCandidateIndices(
                    costs, fields);
            if (order == null || order.length != dry.size()) {
                throw new IllegalStateException(
                        "CUDA candidate sort returned an invalid permutation");
            }
            ArrayList<Candidate> sorted = new ArrayList<>(dry.size());
            boolean[] seen = new boolean[dry.size()];
            for (int index : order) {
                if (index < 0 || index >= dry.size() || seen[index]) {
                    throw new IllegalStateException(
                            "CUDA candidate sort returned a non-permutation");
                }
                seen[index] = true;
                sorted.add(dry.get(index));
            }
            for (int i = 1; i < sorted.size(); i++) {
                if (sorted.get(i - 1).compareTo(sorted.get(i)) > 0) {
                    throw new IllegalStateException(
                            "CUDA candidate sort violates the frozen base order");
                }
            }
            dry.clear();
            dry.addAll(sorted);
            GpuStageStats.gpu(GpuStageStats.Stage.CANDIDATE_SORT);
        } catch (RuntimeException | LinkageError failure) {
            if (ExecutionBackend.cudaFailureIsFatal()) throw failure;
            GpuStageStats.fallback(GpuStageStats.Stage.CANDIDATE_SORT, failure);
            Collections.sort(dry);
        }
    }

    private static boolean gpuCandidateSortConfigured() {
        return ExecutionBackend.cudaPermitted()
                && Boolean.parseBoolean(System.getProperty(
                "rap.gpu.candidate.sort",
                System.getenv().getOrDefault(
                        "RAP_GPU_CANDIDATE_SORT", "true")))
                && CudaTrueParallelNative.isAvailable();
    }

    private static StatisticalTransitionCostModel.Result collectCandidates(
            Context context,
            double[][] referenceLuminance,
            double[] luminanceHistogram,
            JpegImage ref
    ) {
        int draftCount = 0;
        for (int component = 1; component <= 2; component++) {
            JpegComponent c = ref.component(component);
            draftCount = Math.addExact(draftCount, Math.multiplyExact(
                    Math.multiplyExact(c.widthBlocks, c.heightBlocks),
                    JpegMath.TARGET_ZZ_END_RANK - JpegMath.TARGET_ZZ_START_RANK + 1));
        }
        List<StatisticalTransitionCostModel.Draft> drafts = new ArrayList<>(draftCount);
        double[] directional = new double[2];
        // RAP's declared histogram association is reconstructed directly from
        // canonical quantized Y coefficients, not a runtime JPEG decoder.
        if (luminanceHistogram == null || luminanceHistogram.length != 256) {
            throw new IllegalArgumentException(
                    "Canonical luminance histogram must contain 256 bins");
        }
        int maxH = maximumHorizontalSampling(ref);
        int maxV = maximumVerticalSampling(ref);
        for (int comp = 1; comp <= 2 && comp < ref.components.length; comp++) { // Cb and Cr only
            JpegComponent jc = ref.component(comp);
            // The precover component plane supplies the side-informed UNIWARD residual denominator and DCT F_i.
            double[][] pngComponentSamples =
                    context.source.componentSamples(ref, comp);
            double[][] pngComponentPlane =
                    context.source.paddedComponentPlane(ref, comp);
            SiUniwardCostEngine.PreparedResiduals residuals =
                    context.residuals(ref, comp, pngComponentPlane);
            SiUniwardCostEngine.PreparedPlane siContext =
                    SiUniwardCostEngine.prepare(residuals, jc.quantTable);
            int[] targetNaturalIndices =
                    JpegMath.targetExpandedMidBandNaturalIndices();
            double[] baseSiCosts = SiUniwardCostEngine.residualCosts(
                    siContext, jc.widthBlocks, jc.heightBlocks, targetNaturalIndices);
            int bandCount = targetNaturalIndices.length;
            double[] rawDctGrid =
                    context.source.rawDctCoefficients(ref, comp);
            double[] blockVariances =
                    context.source.blockVariances(ref, comp);
            int componentWidth = pngComponentSamples[0].length;
            int componentHeight = pngComponentSamples.length;
            int luminanceWidth = referenceLuminance[0].length;
            int luminanceHeight = referenceLuminance.length;
            CoefficientStatistics coefficientStatistics =
                    new CoefficientStatistics(jc, targetNaturalIndices);
            double[] zeroCrossingPenalties =
                    coefficientStatistics.zeroCrossingPenalties();
            double samplingAreaScale =
                    (maxH / (double) jc.hSamplingFactor)
                            * (maxV / (double) jc.vSamplingFactor);
            double[] rgbMseByNaturalIndex = new double[64];
            double[] ssimNumeratorByNaturalIndex = new double[64];
            for (int k : targetNaturalIndices) {
                int quantStep = jc.quantTable[k];
                rgbMseByNaturalIndex[k] = rgbMseEnergyProxy(
                        comp, quantStep, samplingAreaScale);
                ssimNumeratorByNaturalIndex[k] =
                        square(quantStep) / 64.0;
            }

            for (int by = 0; by < jc.heightBlocks; by++) {
                for (int bx = 0; bx < jc.widthBlocks; bx++) {
                    int block = by * jc.widthBlocks + bx;
                    int px = bx * 8;
                    int py = by * 8;

                    // Compute block-invariant context once for all 32 target ranks.
                    int lumaX = Math.min(luminanceWidth - 1,
                            (int) Math.floor((px + 4)
                                    * (double) luminanceWidth / componentWidth));
                    int lumaY = Math.min(luminanceHeight - 1,
                            (int) Math.floor((py + 4)
                                    * (double) luminanceHeight / componentHeight));
                    int val = ImageUtil.clamp(
                            (int) Math.round(referenceLuminance[lumaY][lumaX]));
                    double histogramWeight =
                            1.0 / (1e-6 + luminanceHistogram[val]);
                    double localVariance = blockVariances[block];
                    int rawDctBase = block * 64;
                    int baseCostOffset = block * bandCount;

                    for (int zz = JpegMath.TARGET_ZZ_START_RANK;
                         zz <= JpegMath.TARGET_ZZ_END_RANK;
                         zz++) {
                        int bandOffset = zz - JpegMath.TARGET_ZZ_START_RANK;
                        int k = targetNaturalIndices[bandOffset];
                        int u = JpegMath.uFromNaturalIndex(k);
                        int v = JpegMath.vFromNaturalIndex(k);
                        int qRef = jc.getCoeff(block, k);

                        // Three-state carrier universe: every Cb/Cr address in
                        // the declared target band, including quantized zeros.

                        int quantStep = jc.quantTable[k];
                        if (quantStep <= 0) {
                            throw new IllegalStateException(
                                    "Canonical quantization step is not positive");
                        }

                        // One reference SI unit-impact cost; the complete
                        // component batch is evaluated by the selected CPU/HIP
                        // backend before RAP directional asymmetry is applied.
                        double baseSiCost = baseSiCosts[baseCostOffset + bandOffset];
                        double rhoPlus = baseSiCost;
                        double rhoMinus = baseSiCost;

                        double rawDct = rawDctGrid[rawDctBase + k];
                        double roundingError = JpegMath.normalizedRoundingError(rawDct, quantStep, qRef);
                        boolean formerFourModeNearHalf =
                                JpegMath.isFormerFourModeNearHalf(
                                        u, v, roundingError);
                        double preferredFactor = Math.max(0.0, 1.0 - 2.0 * Math.abs(roundingError));
                        double plusModulationFactor = roundingError > 0.0 ? preferredFactor : 1.0;
                        double minusModulationFactor = roundingError < 0.0 ? preferredFactor : 1.0;
                        JpegMath.applySideInformedModulation(
                                rhoPlus, rhoMinus, roundingError, directional);
                        rhoPlus = scaleFiniteCost(
                                directional[0], histogramWeight);
                        rhoMinus = scaleFiniteCost(
                                directional[1], histogramWeight);

                        // Explicit statistical penalty for crossing the zero
                        // boundary.  This does not forbid activation or
                        // shrinkage; it prevents either transition being
                        // treated as statistically free.
                        double zeroCrossingPenalty = zeroCrossingPenalties[k];
                        if (qRef == 0) {
                            rhoPlus = scaleTransition(rhoPlus, zeroCrossingPenalty);
                            rhoMinus = scaleTransition(rhoMinus, zeroCrossingPenalty);
                        } else if (qRef == 1) {
                            rhoMinus = scaleTransition(rhoMinus, zeroCrossingPenalty);
                        } else if (qRef == -1) {
                            rhoPlus = scaleTransition(rhoPlus, zeroCrossingPenalty);
                        }
                        // Baseline 8-bit JPEG cannot Huffman-code an AC
                        // magnitude above 1023, so a direction that leaves that
                        // range would make the stego file unwritable rather than
                        // merely expensive.  Wetting it here keeps the failure
                        // reproducible on both endpoints instead of surfacing as
                        // a libjpeg write error mid-trial.
                        if (qRef >= JpegMath.BASELINE_MAX_AC_MAGNITUDE) {
                            rhoPlus = JpegMath.WET_COST;
                        }
                        if (qRef <= -JpegMath.BASELINE_MAX_AC_MAGNITUDE) {
                            rhoMinus = JpegMath.WET_COST;
                        }

                        double rgbMseEnergyProxy =
                                rgbMseByNaturalIndex[k];
                        double ssimLossProxy = ssimLossProxy(
                                ssimNumeratorByNaturalIndex[k], localVariance);
                        TransitionRisk plusRisk =
                                coefficientStatistics.transition(k, qRef, 1);
                        TransitionRisk minusRisk =
                                coefficientStatistics.transition(k, qRef, -1);

                        drafts.add(new StatisticalTransitionCostModel.Draft(
                                comp, block, bx, by, u, v, k, qRef,
                                rhoPlus, rhoMinus, roundingError,
                                formerFourModeNearHalf,
                                rawDct, quantStep, baseSiCost,
                                plusModulationFactor, minusModulationFactor,
                                histogramWeight, zeroCrossingPenalty,
                                rgbMseEnergyProxy, ssimLossProxy,
                                plusRisk.symmetricKl(),
                                minusRisk.symmetricKl(),
                                plusRisk.absoluteEntropyDelta(),
                                minusRisk.absoluteEntropyDelta()));
                    }
                }
            }
        }
        return StatisticalTransitionCostModel.finalizeCosts(drafts);
    }

    private static double scaleTransition(double cost, double factor) {
        if (JpegMath.isWetCost(cost)) return JpegMath.WET_COST;
        double adjusted = cost * factor;
        return JpegMath.capComputedLegalCost(adjusted);
    }

    private static double scaleFiniteCost(
            double cost,
            double histogramWeight
    ) {
        if (JpegMath.isWetCost(cost)) return JpegMath.WET_COST;
        double scaled = cost * histogramWeight;
        return JpegMath.capComputedLegalCost(scaled);
    }

    /**
     * Orthonormal-IDCT unit-impact energy mapped through the declared RGB/YCbCr
     * matrix and component sampling footprint.  This is a deterministic
     * marginal MSE proxy; clipping and decoder-specific chroma interpolation
     * remain part of the post-embedding measured MSE.
     */
    private static double rgbMseEnergyProxy(
            int component,
            int quantStep,
            double samplingAreaScale
    ) {
        double colourEnergy = component == 1
                ? CB_RGB_ENERGY : CR_RGB_ENERGY;
        return square(quantStep) * samplingAreaScale
                * colourEnergy / (64.0 * 3.0);
    }

    /**
     * Local contrast-normalized unit-impact energy.  It follows the contrast
     * denominator used by SSIM but is explicitly a marginal proxy, not a claim
     * that non-additive full-image SSIM decomposes by coefficient.
     */
    private static double ssimLossProxy(
            double unitImpactEnergy,
            double localVariance
    ) {
        return unitImpactEnergy
                / (Math.max(0.0, localVariance) + SSIM_C2);
    }

    private static int maximumHorizontalSampling(JpegImage image) {
        int max = 1;
        for (JpegComponent component : image.components) {
            max = Math.max(max, component.hSamplingFactor);
        }
        return max;
    }

    private static int maximumVerticalSampling(JpegImage image) {
        int max = 1;
        for (JpegComponent component : image.components) {
            max = Math.max(max, component.vSamplingFactor);
        }
        return max;
    }

    private static double square(double value) {
        return value * value;
    }

    private record TransitionRisk(
            double symmetricKl,
            double absoluteEntropyDelta
    ) {}

    /** Exact one-count DCT-histogram perturbation statistics for one component. */
    private static final class CoefficientStatistics {
        private static final double KL_EPSILON = 1.0e-12;
        private final List<Map<Integer, Long>> histograms =
                new ArrayList<>(64);
        private final long total;
        private final List<Map<Integer, TransitionRiskPair>> cache =
                new ArrayList<>(64);

        private final int[] targetNaturalIndices;

        private CoefficientStatistics(
                JpegComponent component, int[] targetNaturalIndices) {
            total = component.blockCount();
            this.targetNaturalIndices = targetNaturalIndices.clone();
            for (int k = 0; k < 64; k++) {
                histograms.add(new HashMap<>());
                cache.add(new HashMap<>());
            }
            // Only the declared 32-coefficient carrier band participates in
            // RAP transitions. Scanning the other 32 coefficients is pure load.
            for (int block = 0; block < component.blockCount(); block++) {
                for (int k : this.targetNaturalIndices) {
                    histograms.get(k).merge(
                            component.getCoeff(block, k), 1L, Long::sum);
                }
            }
        }

        private double[] zeroCrossingPenalties() {
            double[] penalties = new double[64];
            java.util.Arrays.fill(penalties, 1.0);
            for (int k : targetNaturalIndices) {
                Map<Integer, Long> histogram = histograms.get(k);
                long zeros = histogram.getOrDefault(0, 0L);
                long adjacent = histogram.getOrDefault(1, 0L)
                        + histogram.getOrDefault(-1, 0L);
                penalties[k] = 1.0 + 0.05 * zeros / (adjacent + 1.0);
            }
            return penalties;
        }

        private TransitionRisk transition(int k, int q, int delta) {
            TransitionRiskPair pair = cache.get(k).computeIfAbsent(
                    q, ignored -> new TransitionRiskPair(
                            calculate(k, q, 1), calculate(k, q, -1)));
            return delta > 0 ? pair.plus() : pair.minus();
        }

        private record TransitionRiskPair(
                TransitionRisk plus, TransitionRisk minus
        ) {}

        private TransitionRisk calculate(int k, int q, int delta) {
            Map<Integer, Long> histogram = histograms.get(k);
            long from = histogram.getOrDefault(q, 0L);
            if (from <= 0L) {
                throw new IllegalStateException(
                        "Reference coefficient is absent from its histogram");
            }
            int target = Math.addExact(q, delta);
            long to = histogram.getOrDefault(target, 0L);
            int support = histogram.size() + (to == 0L ? 1 : 0);
            double denominator = total + KL_EPSILON * support;

            double pFrom = (from + KL_EPSILON) / denominator;
            double qFrom = (from - 1L + KL_EPSILON) / denominator;
            double pTo = (to + KL_EPSILON) / denominator;
            double qTo = (to + 1L + KL_EPSILON) / denominator;
            double forward = klTerm(pFrom, qFrom) + klTerm(pTo, qTo);
            double reverse = klTerm(qFrom, pFrom) + klTerm(qTo, pTo);
            double symmetric = Math.max(0.0, 0.5 * (forward + reverse));

            double before = entropyTerm(from, total)
                    + entropyTerm(to, total);
            double after = entropyTerm(from - 1L, total)
                    + entropyTerm(to + 1L, total);
            return new TransitionRisk(
                    symmetric, Math.abs(after - before));
        }

        private static double klTerm(double p, double q) {
            return p <= 0.0 ? 0.0 : p * StrictMath.log(p / q);
        }

        private static double entropyTerm(long count, long total) {
            if (count <= 0L || total <= 0L) return 0.0;
            double probability = count / (double) total;
            return -probability
                    * (StrictMath.log(probability) / StrictMath.log(2.0));
        }
    }
}
