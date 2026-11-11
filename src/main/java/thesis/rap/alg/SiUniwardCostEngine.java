package thesis.rap.alg;

import thesis.rap.gpu.HipSiUniwardNative;
import thesis.rap.gpu.CudaSiUniwardNative;
import thesis.rap.runtime.ExecutionBackend;

import java.util.Locale;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.LongAdder;

/**
 * Reference-aligned SI-UNIWARD base-cost engine used by RAP-UNIWARD before
 * the thesis-specific directional, histogram, and chroma weighting layers.
 *
 * <p>This class follows the public DDE/MATLAB construction: the precover is
 * padded by 16 samples using symmetric reflection, three Daubechies-8
 * residuals are calculated on that padded plane, and every coefficient cost
 * uses the complete 23x23 full-correlation response of its 8x8 spatial unit
 * impact.  In particular, boundary coefficients do not use a cropped impact
 * window.</p>
 *
 * <p>The impact window is aligned to the residual samples a unit coefficient
 * change actually disturbs; see {@link #IMPACT_WINDOW_OFFSET}.  This is the
 * corrected alignment declared by the RAP methodology, not the off-by-one
 * window of the published DDE MATLAB reference.</p>
 */
public final class SiUniwardCostEngine {
    /** Stabilizer specified by SI-UNIWARD: 2^-6. */
    public static final double SIGMA = 1.0 / 64.0;
    private static final double WET_COST = JpegMath.WET_COST;
    private static final int FILTER_N = 16;
    private static final int FILTER_CENTER = 7; // MATLAB floor((16+1)/2)-1
    private static final int PAD = 16;
    private static final int IMPACT_N = 8 + FILTER_N - 1; // 23
    private static final int FILTER_COUNT = 3;
    private static final int IMPACT_AREA = IMPACT_N * IMPACT_N;
    private static final int NATURAL_COEFFICIENT_COUNT = 64;

    private static final int HIP_MIN_CANDIDATES = readPositiveInt(
            "rap.si.hipMinCandidates", "RAP_SI_HIP_MIN_CANDIDATES", 8192);
    private static final int CUDA_MIN_CANDIDATES = readPositiveInt(
            "rap.si.cudaMinCandidates", "RAP_SI_CUDA_MIN_CANDIDATES", 8192);
    private static final int GPU_SLOT_COUNT = readPositiveInt(
            "rap.gpu.slots", "RAP_GPU_SLOTS", 1);
    
    private static final int CUDA_INFLIGHT = readPositiveInt(
            "rap.cuda.inflight", "RAP_CUDA_INFLIGHT", 16);
    private static final Semaphore HIP_SLOTS = new Semaphore(GPU_SLOT_COUNT, true);
    private static final Semaphore CUDA_SLOTS = new Semaphore(CUDA_INFLIGHT, true);
    private static final AtomicBoolean CUDA_AUTO_FALLBACK_WARNING_PRINTED =
            new AtomicBoolean(false);
    private static final AtomicBoolean HIP_AUTO_FALLBACK_WARNING_PRINTED =
            new AtomicBoolean(false);

    private static final LongAdder CPU_BATCHES = new LongAdder();
    private static final LongAdder CPU_CANDIDATES = new LongAdder();
    private static final LongAdder CPU_NANOS = new LongAdder();
    private static final LongAdder HIP_BATCHES = new LongAdder();
    private static final LongAdder HIP_CANDIDATES = new LongAdder();
    private static final LongAdder HIP_NANOS = new LongAdder();
    private static final LongAdder CUDA_BATCHES = new LongAdder();
    private static final LongAdder CUDA_CANDIDATES = new LongAdder();
    private static final LongAdder CUDA_NANOS = new LongAdder();
    private static final LongAdder VERIFY_BATCHES = new LongAdder();
    private static final LongAdder AUTO_FALLBACKS = new LongAdder();

    /*
     * A unit JPEG coefficient impact depends only on its natural index and
     * quantization step, never on image content.  Standard JPEG QF scans reuse
     * the same small set of (k,q) pairs across thousands of images, so this
     * immutable cache eliminates repeated 8x8 IDCT-basis filtering.
     */
    private static final ConcurrentMap<ImpactKey, double[][][]> IMPACT_CACHE =
            new ConcurrentHashMap<>();

    /**
     * Offset from an 8x8 block origin in the unpadded plane to the first
     * residual sample that a unit coefficient change can reach.
     *
     * <p>Derivation. Let the block occupy padded rows {@code [B, B+7]}. The
     * "same"-size correlation computes
     * {@code R[p] = sum_t X[p + t - FILTER_CENTER] * F[t]}, so {@code R[p]}
     * reads {@code X[p-7 .. p+8]} and is disturbed exactly for
     * {@code p} in {@code [B-8, B+14]}: 23 samples beginning at {@code B-8}.
     * With {@code B = blockOrigin + PAD}, the first disturbed residual sample
     * is at {@code blockOrigin + PAD - 8}.
     *
     * <p>The published DDE J-UNIWARD MATLAB reference uses
     * {@code row-modRow-6+padSize}, that is {@code blockOrigin + PAD - 7}: one
     * sample too far on both axes. This is the off-by-one reported by Lorch et
     * al. (2023). It pairs the numerator (the wavelet impact at position
     * {@code p}) with a denominator sampled at {@code p-1}.  It can change
     * individual costs and therefore the frozen carrier order.  This source
     * comment intentionally makes no corpus-level numerical claim; any effect
     * size belongs to a retained, reproducible experiment rather than code
     * documentation.
     *
     * <p>The RAP methodology declares the corrected window, so the corrected
     * value is normative here. {@link #DDE_COMPARISON_WINDOW_OFFSET} is retained
     * for documentation and for the declared alignment ablation.
     */
    private static final int WINDOW_OFFSET = PAD - 8;

    /** Uncorrected offset used by the published DDE J-UNIWARD MATLAB code. */
    public static final int DDE_COMPARISON_WINDOW_OFFSET = PAD - 7;

    /** Impact-window offset actually applied by this build. */
    public static final int IMPACT_WINDOW_OFFSET = WINDOW_OFFSET;

    /** Stable provenance label for the impact-window alignment in use. */
    public static final String IMPACT_WINDOW_ALIGNMENT =
            WINDOW_OFFSET == DDE_COMPARISON_WINDOW_OFFSET
                    ? "DDE_COMPARISON_OFF_BY_ONE"
                    : "CORRECTED_LORCH2023";

    private SiUniwardCostEngine() {}

    // Public DDE Daubechies-8 decomposition filters (full precision version).
    private static final double[] HPDF = {
            -0.054415842243081609,
             0.31287159091446592,
            -0.67563073629801285,
             0.58535468365486909,
             0.015829105256023893,
            -0.28401554296242809,
            -0.00047248457399797254,
             0.12874742662018601,
             0.017369301002022108,
            -0.044088253931064719,
            -0.013981027917015516,
             0.0087460940470156547,
             0.0048703529930106603,
            -0.00039174037299597711,
            -0.00067544940599855677,
            -0.00011747678400228192
    };

    private static final double[] LPDF = {
            -0.00011747678400228192,
             0.00067544940599855677,
            -0.00039174037299597711,
            -0.0048703529930106603,
             0.0087460940470156547,
             0.013981027917015516,
            -0.044088253931064719,
            -0.017369301002022108,
             0.12874742662018601,
             0.00047248457399797254,
            -0.28401554296242809,
            -0.015829105256023893,
             0.58535468365486909,
             0.67563073629801285,
             0.31287159091446592,
             0.054415842243081609
    };

    // F1=lpdf'*hpdf, F2=hpdf'*lpdf, F3=hpdf'*hpdf.
    private static final double[][] Y_FILTERS = { LPDF, HPDF, HPDF };
    private static final double[][] X_FILTERS = { HPDF, LPDF, HPDF };

    /**
     * QF-independent wavelet residual context for one precover component plane.
     * A multi-QF scan computes this convolution once and reuses it while only
     * the quantization-dependent unit impacts change.
     */
    public static final class PreparedResiduals {
        private final int width;
        private final int height;
        private final double[][][] residuals;
        private volatile double[] packedDenominators;

        private PreparedResiduals(double[][] plane) {
            validatePlane(plane);
            this.width = plane[0].length;
            this.height = plane.length;
            double[][] padded = symmetricPad(plane, PAD);
            this.residuals = new double[FILTER_COUNT][][];
            for (int f = 0; f < FILTER_COUNT; f++) {
                this.residuals[f] = correlateSameZero(
                        padded, Y_FILTERS[f], X_FILTERS[f]);
            }
        }

        private double[] packedDenominators() {
            double[] local = packedDenominators;
            if (local != null) return local;
            synchronized (this) {
                local = packedDenominators;
                if (local != null) return local;
                int residualHeight = residuals[0].length;
                int residualWidth = residuals[0][0].length;
                int residualArea = Math.multiplyExact(residualWidth, residualHeight);
                local = new double[Math.multiplyExact(FILTER_COUNT, residualArea)];
                for (int f = 0; f < FILTER_COUNT; f++) {
                    int filterBase = f * residualArea;
                    for (int y = 0; y < residualHeight; y++) {
                        int rowBase = filterBase + y * residualWidth;
                        for (int x = 0; x < residualWidth; x++) {
                            local[rowBase + x] = SIGMA + Math.abs(residuals[f][y][x]);
                        }
                    }
                }
                packedDenominators = local;
                return local;
            }
        }
    }

    /**
     * Reusable context for one component plane at one quantization table.
     * QF-independent residuals can be shared across many instances.
     */
    public static final class PreparedPlane {
        private final int width;
        private final int height;
        private final PreparedResiduals preparedResiduals;
        private final int[] quantTable;
        private final double[][][][] impacts;

        private PreparedPlane(PreparedResiduals preparedResiduals, int[] quantTable) {
            if (preparedResiduals == null) {
                throw new IllegalArgumentException("preparedResiduals cannot be null");
            }
            if (quantTable == null || quantTable.length != NATURAL_COEFFICIENT_COUNT) {
                throw new IllegalArgumentException(
                        "quantTable must have 64 natural-order entries");
            }
            this.width = preparedResiduals.width;
            this.height = preparedResiduals.height;
            this.preparedResiduals = preparedResiduals;
            this.quantTable = quantTable.clone();
            this.impacts = new double[NATURAL_COEFFICIENT_COUNT][][][];
            for (int quantStep : this.quantTable) {
                if (quantStep <= 0) {
                    throw new IllegalArgumentException(
                            "quantization steps must be positive");
                }
            }
        }

        private double[][][] impactFor(int k) {
            double[][][] impact = impacts[k];
            if (impact != null) return impact;
            synchronized (impacts) {
                impact = impacts[k];
                if (impact == null) {
                    int quantStep = quantTable[k];
                    impact = IMPACT_CACHE.computeIfAbsent(
                            new ImpactKey(k, quantStep),
                            key -> buildImpact(
                                    key.naturalIndex(), key.quantStep()));
                    impacts[k] = impact;
                }
                return impact;
            }
        }

        private AcceleratorBuffers acceleratorBuffers(int[] naturalIndices) {
            int residualHeight = preparedResiduals.residuals[0].length;
            int residualWidth = preparedResiduals.residuals[0][0].length;
            int bandCount = naturalIndices.length;
            double[][][][] selectedImpacts =
                    new double[bandCount][][][];
            for (int band = 0; band < bandCount; band++) {
                selectedImpacts[band] = impactFor(naturalIndices[band]);
            }
            double[] packedImpacts = new double[Math.multiplyExact(
                    bandCount, FILTER_COUNT * IMPACT_AREA)];
            for (int f = 0; f < FILTER_COUNT; f++) {
                for (int iy = 0; iy < IMPACT_N; iy++) {
                    for (int ix = 0; ix < IMPACT_N; ix++) {
                        int position = iy * IMPACT_N + ix;
                        int base = (f * IMPACT_AREA + position) * bandCount;
                        for (int band = 0; band < bandCount; band++) {
                            int k = naturalIndices[band];
                            packedImpacts[base + band] =
                                    Math.abs(selectedImpacts[band][f][iy][ix]);
                        }
                    }
                }
            }
            return new AcceleratorBuffers(
                    residualWidth, residualHeight, bandCount,
                    preparedResiduals.packedDenominators(), packedImpacts);
        }
    }

    private record ImpactKey(int naturalIndex, int quantStep) {}

    private static double[][][] buildImpact(int k, int quantStep) {
        int u = JpegMath.uFromNaturalIndex(k);
        int v = JpegMath.vFromNaturalIndex(k);
        double[][] basis = JpegMath.idctBasis(u, v, quantStep);
        double[][][] impact = new double[FILTER_COUNT][][];
        for (int f = 0; f < FILTER_COUNT; f++) {
            impact[f] = correlateFull(basis, Y_FILTERS[f], X_FILTERS[f]);
        }
        return impact;
    }

    private record AcceleratorBuffers(
            int residualWidth,
            int residualHeight,
            int bandCount,
            double[] denominators,
            double[] impactMagnitudesPositionMajor
    ) {}

    public static PreparedResiduals prepareResiduals(double[][] plane) {
        return new PreparedResiduals(plane);
    }

    public static PreparedPlane prepare(PreparedResiduals residuals, int[] quantTable) {
        return new PreparedPlane(residuals, quantTable.clone());
    }

    public static PreparedPlane prepare(double[][] plane, int[] quantTable) {
        return prepare(prepareResiduals(plane), quantTable);
    }

    public static double residualCost(PreparedPlane prepared, int pixelX, int pixelY,
                                      int naturalCoefficientIndex) {
        if (prepared == null) throw new IllegalArgumentException("prepared plane is null");
        if (naturalCoefficientIndex < 0 || naturalCoefficientIndex >= 64) {
            throw new IllegalArgumentException("natural coefficient index out of range");
        }
        validateBlock(pixelX, pixelY, prepared.width, prepared.height);
        return sumCost(prepared.preparedResiduals.residuals, prepared.impactFor(naturalCoefficientIndex),
                pixelX, pixelY);
    }

    /**
     * Computes all SI base costs for a component in one batch.  Output ordering
     * is block-major, then the supplied natural-index order.
     *
     * <p>Backend selection is controlled by {@code -Drap.si.backend=} or
     * {@code RAP_SI_BACKEND}: CPU, HIP, CUDA, AUTO, VERIFY, VERIFY_HIP, or
     * VERIFY_CUDA. Explicit accelerators are fail-closed. AUTO prefers CUDA,
     * then HIP, then CPU. VERIFY uses the first available accelerator and
     * checks it against the CPU binary64 reference.</p>
     */
    public static double[] residualCosts(
            PreparedPlane prepared,
            int widthBlocks,
            int heightBlocks,
            int[] naturalIndices
    ) {
        validateBatch(prepared, widthBlocks, heightBlocks, naturalIndices);
        int candidateCount = Math.multiplyExact(
                Math.multiplyExact(widthBlocks, heightBlocks),
                naturalIndices.length);
        Backend backend = configuredBackend();
        return switch (backend) {
            case CPU -> residualCostsCpu(prepared, widthBlocks, heightBlocks, naturalIndices);
            case HIP -> {
                try {
                    yield residualCostsHip(prepared, widthBlocks, heightBlocks, naturalIndices);
                } catch (RuntimeException | LinkageError failure) {
                    if (!ExecutionBackend.explicitAcceleratorFallbackEnabled()) throw failure;
                    AUTO_FALLBACKS.increment();
                    yield residualCostsCpu(prepared, widthBlocks, heightBlocks, naturalIndices);
                }
            }
            case CUDA -> {
                try {
                    yield residualCostsCuda(prepared, widthBlocks, heightBlocks, naturalIndices);
                } catch (RuntimeException | LinkageError failure) {
                    if (!ExecutionBackend.explicitAcceleratorFallbackEnabled()) throw failure;
                    AUTO_FALLBACKS.increment();
                    yield residualCostsCpu(prepared, widthBlocks, heightBlocks, naturalIndices);
                }
            }
            case VERIFY_HIP -> verifyHip(prepared, widthBlocks, heightBlocks, naturalIndices);
            case VERIFY_CUDA -> verifyCuda(prepared, widthBlocks, heightBlocks, naturalIndices);
            case VERIFY -> {
                if (CudaSiUniwardNative.isAvailable()) {
                    yield verifyCuda(prepared, widthBlocks, heightBlocks, naturalIndices);
                }
                if (HipSiUniwardNative.isAvailable()) {
                    yield verifyHip(prepared, widthBlocks, heightBlocks, naturalIndices);
                }
                throw new IllegalStateException(
                        "RAP_SI_BACKEND=VERIFY requires CUDA or HIP; CUDA="
                                + CudaSiUniwardNative.availabilityDiagnostic() + ", HIP="
                                + HipSiUniwardNative.availabilityDiagnostic());
            }
            case AUTO -> {
                if (candidateCount >= CUDA_MIN_CANDIDATES
                        && CudaSiUniwardNative.isAvailable()) {
                    try {
                        yield residualCostsCuda(
                                prepared, widthBlocks, heightBlocks, naturalIndices);
                    } catch (RuntimeException | LinkageError gpuFailure) {
                        AUTO_FALLBACKS.increment();
                        if (CUDA_AUTO_FALLBACK_WARNING_PRINTED.compareAndSet(false, true)) {
                            System.err.println(
                                    "WARN: SI-UNIWARD CUDA AUTO fallback: " + gpuFailure);
                        }
                    }
                }
                if (candidateCount >= HIP_MIN_CANDIDATES
                        && HipSiUniwardNative.isAvailable()) {
                    try {
                        yield residualCostsHip(
                                prepared, widthBlocks, heightBlocks, naturalIndices);
                    } catch (RuntimeException | LinkageError gpuFailure) {
                        AUTO_FALLBACKS.increment();
                        if (HIP_AUTO_FALLBACK_WARNING_PRINTED.compareAndSet(false, true)) {
                            System.err.println(
                                    "WARN: SI-UNIWARD HIP AUTO fallback to CPU: "
                                            + gpuFailure);
                        }
                    }
                }
                yield residualCostsCpu(
                        prepared, widthBlocks, heightBlocks, naturalIndices);
            }
        };
    }

    private static double[] verifyHip(PreparedPlane prepared, int widthBlocks,
                                      int heightBlocks, int[] naturalIndices) {
        if (!HipSiUniwardNative.isAvailable()) {
            throw new IllegalStateException("VERIFY_HIP requires HIP: "
                    + HipSiUniwardNative.availabilityDiagnostic());
        }
        double[] cpu = residualCostsCpu(prepared, widthBlocks, heightBlocks, naturalIndices);
        double[] result = residualCostsHip(prepared, widthBlocks, heightBlocks, naturalIndices);
        verifyParity(cpu, result, "HIP");
        VERIFY_BATCHES.increment();
        return result;
    }

    private static double[] verifyCuda(PreparedPlane prepared, int widthBlocks,
                                       int heightBlocks, int[] naturalIndices) {
        if (!CudaSiUniwardNative.isAvailable()) {
            throw new IllegalStateException("VERIFY_CUDA requires CUDA: "
                    + CudaSiUniwardNative.availabilityDiagnostic());
        }
        double[] cpu = residualCostsCpu(prepared, widthBlocks, heightBlocks, naturalIndices);
        double[] result = residualCostsCuda(prepared, widthBlocks, heightBlocks, naturalIndices);
        verifyParity(cpu, result, "CUDA");
        VERIFY_BATCHES.increment();
        return result;
    }

    /**
     * Batched CPU reference used by production and parity tests.  It deliberately
     * keeps the original nested residual/impact representation so CPU-only runs
     * do not pay the GPU packing cost or change the proven summation order.
     */
    static double[] residualCostsCpu(
            PreparedPlane prepared,
            int widthBlocks,
            int heightBlocks,
            int[] naturalIndices
    ) {
        validateBatch(prepared, widthBlocks, heightBlocks, naturalIndices);
        int blockCount = Math.multiplyExact(widthBlocks, heightBlocks);
        int bandCount = naturalIndices.length;
        double[] out = new double[Math.multiplyExact(blockCount, bandCount)];
        double[][][][] selectedImpacts = new double[bandCount][][][];
        for (int band = 0; band < bandCount; band++) {
            selectedImpacts[band] =
                    prepared.impactFor(naturalIndices[band]);
        }
        long start = System.nanoTime();
        for (int block = 0; block < blockCount; block++) {
            int bx = block % widthBlocks;
            int by = block / widthBlocks;
            int pixelX = bx * 8;
            int pixelY = by * 8;
            int outBase = block * bandCount;
            for (int band = 0; band < bandCount; band++) {
                out[outBase + band] = sumCost(
                        prepared.preparedResiduals.residuals,
                        selectedImpacts[band], pixelX, pixelY);
            }
        }
        CPU_NANOS.add(System.nanoTime() - start);
        CPU_BATCHES.increment();
        CPU_CANDIDATES.add(out.length);
        return out;
    }

    /** AMD HIP batch path; one RDNA wave32 evaluates one JPEG block. */
    static double[] residualCostsHip(
            PreparedPlane prepared,
            int widthBlocks,
            int heightBlocks,
            int[] naturalIndices
    ) {
        validateBatch(prepared, widthBlocks, heightBlocks, naturalIndices);
        if (!HipSiUniwardNative.isAvailable()) {
            throw new IllegalStateException(
                    "HIP backend requested but no usable HIP device is available: "
                            + HipSiUniwardNative.availabilityDiagnostic());
        }
        HIP_SLOTS.acquireUninterruptibly();
        try {
            long start = System.nanoTime();
            AcceleratorBuffers buffers = prepared.acceleratorBuffers(naturalIndices);
            double[] out = HipSiUniwardNative.computeCosts(
                    buffers.denominators(),
                    buffers.residualWidth(),
                    buffers.residualHeight(),
                    buffers.impactMagnitudesPositionMajor(),
                    buffers.bandCount(),
                    widthBlocks,
                    heightBlocks,
                    WET_COST,
                    WINDOW_OFFSET);
            HIP_NANOS.add(System.nanoTime() - start);
            HIP_BATCHES.increment();
            HIP_CANDIDATES.add(out.length);
            return out;
        } finally {
            HIP_SLOTS.release();
        }
    }

    /** NVIDIA CUDA batch path; one CUDA block evaluates one JPEG block. */
    static double[] residualCostsCuda(
            PreparedPlane prepared, int widthBlocks, int heightBlocks,
            int[] naturalIndices) {
        validateBatch(prepared, widthBlocks, heightBlocks, naturalIndices);
        if (!CudaSiUniwardNative.isAvailable()) {
            throw new IllegalStateException(
                    "CUDA backend requested but no usable CUDA device is available: "
                            + CudaSiUniwardNative.availabilityDiagnostic());
        }
        CUDA_SLOTS.acquireUninterruptibly();
        try {
            long start = System.nanoTime();
            AcceleratorBuffers buffers = prepared.acceleratorBuffers(naturalIndices);
            double[] out = CudaSiUniwardNative.computeCosts(
                    buffers.denominators(), buffers.residualWidth(),
                    buffers.residualHeight(),
                    buffers.impactMagnitudesPositionMajor(), buffers.bandCount(),
                    widthBlocks, heightBlocks, WET_COST, WINDOW_OFFSET);
            CUDA_NANOS.add(System.nanoTime() - start);
            CUDA_BATCHES.increment();
            CUDA_CANDIDATES.add(out.length);
            return out;
        } finally {
            CUDA_SLOTS.release();
        }
    }

    /** Backend/timing counters for audit and benchmark reporting. */
    public record BackendPerformance(
            long cpuBatches,
            long cpuCandidates,
            long cpuNanos,
            long hipBatches,
            long hipCandidates,
            long hipNanos,
            long cudaBatches,
            long cudaCandidates,
            long cudaNanos,
            long verifyBatches,
            long autoFallbacks
    ) {
        public double cpuMilliseconds() { return cpuNanos / 1_000_000.0; }
        public double hipMilliseconds() { return hipNanos / 1_000_000.0; }
        public double cudaMilliseconds() { return cudaNanos / 1_000_000.0; }
    }

    public static BackendPerformance backendPerformance() {
        return new BackendPerformance(
                CPU_BATCHES.sum(), CPU_CANDIDATES.sum(), CPU_NANOS.sum(),
                HIP_BATCHES.sum(), HIP_CANDIDATES.sum(), HIP_NANOS.sum(),
                CUDA_BATCHES.sum(), CUDA_CANDIDATES.sum(), CUDA_NANOS.sum(),
                VERIFY_BATCHES.sum(), AUTO_FALLBACKS.sum());
    }

    public static void resetBackendPerformance() {
        CPU_BATCHES.reset();
        CPU_CANDIDATES.reset();
        CPU_NANOS.reset();
        HIP_BATCHES.reset();
        HIP_CANDIDATES.reset();
        HIP_NANOS.reset();
        CUDA_BATCHES.reset();
        CUDA_CANDIDATES.reset();
        CUDA_NANOS.reset();
        VERIFY_BATCHES.reset();
        AUTO_FALLBACKS.reset();
    }

    public static String configuredBackendName() {
        return configuredBackend().name();
    }

    public static boolean hipAvailable() {
        return HipSiUniwardNative.isAvailable();
    }

    public static String hipDeviceName() {
        return HipSiUniwardNative.deviceName();
    }

    public static boolean cudaAvailable() { return CudaSiUniwardNative.isAvailable(); }

    public static String cudaDeviceName() { return CudaSiUniwardNative.deviceName(); }

    /** Compatibility entry point for callers supplying an explicit 8x8 impact. */
    public static double rhoPlus(double[][] plane, int pixelX, int pixelY,
                                 double[][] impactBasis) {
        return residualCost(plane, pixelX, pixelY, impactBasis);
    }

    /** The unmodulated UNIWARD unit cost is symmetric in +/- one DCT unit. */
    public static double rhoMinus(double[][] plane, int pixelX, int pixelY,
                                  double[][] impactBasis) {
        return residualCost(plane, pixelX, pixelY, impactBasis);
    }

    /** Compatibility entry point used by focused tests and external integrations. */
    public static double residualCost(double[][] plane, int pixelX, int pixelY,
                                      double[][] impactBasis) {
        validatePlane(plane);
        validateBasis(impactBasis);
        validateBlock(pixelX, pixelY, plane[0].length, plane.length);
        double[][] padded = symmetricPad(plane, PAD);
        double[][][] residuals = new double[3][][];
        double[][][] impacts = new double[3][][];
        for (int f = 0; f < 3; f++) {
            residuals[f] = correlateSameZero(padded, Y_FILTERS[f], X_FILTERS[f]);
            impacts[f] = correlateFull(impactBasis, Y_FILTERS[f], X_FILTERS[f]);
        }
        return sumCost(residuals, impacts, pixelX, pixelY);
    }

    /**
     * Returns one symmetrically padded directional wavelet residual plane.
     *
     * <p>This is the exact denominator field used by {@link #residualCost}.
     * It is exposed so that the impact-window alignment can be checked against
     * the physically measured change of the residual instead of against a
     * restatement of the same offset constant, and so that the declared
     * alignment ablation can be reported.  Index {@code [y][x]} is in padded
     * coordinates: unpadded sample {@code (x, y)} is at {@code [y+PAD][x+PAD]}.
     *
     * @param filterIndex 0 = LH (lpdf x hpdf), 1 = HL, 2 = HH
     */
    public static double[][] directionalResidual(double[][] plane, int filterIndex) {
        validatePlane(plane);
        if (filterIndex < 0 || filterIndex > 2) {
            throw new IllegalArgumentException("filterIndex must be 0, 1, or 2");
        }
        return correlateSameZero(symmetricPad(plane, PAD),
                Y_FILTERS[filterIndex], X_FILTERS[filterIndex]);
    }

    /** Padding applied before the directional residuals are correlated. */
    public static int padding() {
        return PAD;
    }

    /** Side length of the full unit-impact response, in samples. */
    public static int impactWindowSize() {
        return IMPACT_N;
    }

    /**
     * Returns the full 23x23 wavelet response of a unit coefficient change,
     * i.e. the numerator field of {@link #residualCost} for one filter.
     */
    public static double[][] unitImpactResponse(double[][] impactBasis, int filterIndex) {
        validateBasis(impactBasis);
        if (filterIndex < 0 || filterIndex > 2) {
            throw new IllegalArgumentException("filterIndex must be 0, 1, or 2");
        }
        return correlateFull(impactBasis, Y_FILTERS[filterIndex], X_FILTERS[filterIndex]);
    }

    private static double sumCost(double[][][] residuals, double[][][] impacts,
                                  int pixelX, int pixelY) {
        int y0 = pixelY + WINDOW_OFFSET;
        int x0 = pixelX + WINDOW_OFFSET;
        double cost = 0.0;
        for (int f = 0; f < 3; f++) {
            for (int iy = 0; iy < IMPACT_N; iy++) {
                for (int ix = 0; ix < IMPACT_N; ix++) {
                    cost += Math.abs(impacts[f][iy][ix])
                            / (SIGMA + Math.abs(residuals[f][y0 + iy][x0 + ix]));
                }
            }
        }
        if (!Double.isFinite(cost) || cost < 0.0) return WET_COST;
        return JpegMath.capComputedLegalCost(cost);
    }

    /** MATLAB-compatible symmetric padding with repeated edge samples. */
    private static double[][] symmetricPad(double[][] input, int pad) {
        int height = input.length;
        int width = input[0].length;
        double[][] out = new double[height + 2 * pad][width + 2 * pad];
        for (int y = 0; y < out.length; y++) {
            int sy = symmetricIndex(y - pad, height);
            for (int x = 0; x < out[0].length; x++) {
                out[y][x] = input[sy][symmetricIndex(x - pad, width)];
            }
        }
        return out;
    }

    /** Separable 2-D correlation with MATLAB's even-filter center convention. */
    private static double[][] correlateSameZero(double[][] input,
                                                double[] yKernel, double[] xKernel) {
        int height = input.length;
        int width = input[0].length;
        double[][] horizontal = new double[height][width];
        double[][] output = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0.0;
                for (int t = 0; t < FILTER_N; t++) {
                    int sx = x + t - FILTER_CENTER;
                    if (sx >= 0 && sx < width) sum += input[y][sx] * xKernel[t];
                }
                horizontal[y][x] = sum;
            }
        }
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0.0;
                for (int t = 0; t < FILTER_N; t++) {
                    int sy = y + t - FILTER_CENTER;
                    if (sy >= 0 && sy < height) sum += horizontal[sy][x] * yKernel[t];
                }
                output[y][x] = sum;
            }
        }
        return output;
    }

    /** Full 2-D correlation of an 8x8 impact by a separable 16x16 filter. */
    private static double[][] correlateFull(double[][] input,
                                            double[] yKernel, double[] xKernel) {
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int fullWidth = inputWidth + FILTER_N - 1;
        int fullHeight = inputHeight + FILTER_N - 1;
        double[][] horizontal = new double[inputHeight][fullWidth];
        double[][] output = new double[fullHeight][fullWidth];
        for (int y = 0; y < inputHeight; y++) {
            for (int outX = 0; outX < fullWidth; outX++) {
                double sum = 0.0;
                for (int t = 0; t < FILTER_N; t++) {
                    int sx = outX + t - (FILTER_N - 1);
                    if (sx >= 0 && sx < inputWidth) sum += input[y][sx] * xKernel[t];
                }
                horizontal[y][outX] = sum;
            }
        }
        for (int outY = 0; outY < fullHeight; outY++) {
            for (int x = 0; x < fullWidth; x++) {
                double sum = 0.0;
                for (int t = 0; t < FILTER_N; t++) {
                    int sy = outY + t - (FILTER_N - 1);
                    if (sy >= 0 && sy < inputHeight) sum += horizontal[sy][x] * yKernel[t];
                }
                output[outY][x] = sum;
            }
        }
        return output;
    }

    /** Symmetric reflection: -1->0, -2->1, n->n-1, n+1->n-2. */
    static int symmetricIndex(int index, int length) {
        if (length <= 1) return 0;
        int i = index;
        while (i < 0 || i >= length) {
            if (i < 0) i = -i - 1;
            else i = 2 * length - i - 1;
        }
        return i;
    }

    private enum Backend { CPU, HIP, CUDA, AUTO, VERIFY, VERIFY_HIP, VERIFY_CUDA }

    private static Backend configuredBackend() {
        if (ExecutionBackend.explicitlyConfigured()) {
            return switch (ExecutionBackend.mode()) {
                case CPU -> Backend.CPU;
                case CUDA -> Backend.CUDA;
                case HIP -> Backend.HIP;
                case AUTO -> Backend.AUTO;
            };
        }
        String value = System.getProperty("rap.si.backend");
        if (value == null || value.isBlank()) value = System.getenv("RAP_SI_BACKEND");
        if (value == null || value.isBlank()) value = "AUTO";
        try {
            return Backend.valueOf(value.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Unknown SI backend '" + value
                            + "'; expected CPU, HIP, CUDA, AUTO, VERIFY, "
                            + "VERIFY_HIP, or VERIFY_CUDA", e);
        }
    }

    private static void validateBatch(
            PreparedPlane prepared,
            int widthBlocks,
            int heightBlocks,
            int[] naturalIndices
    ) {
        if (prepared == null) throw new IllegalArgumentException("prepared plane is null");
        if (widthBlocks <= 0 || heightBlocks <= 0) {
            throw new IllegalArgumentException("block dimensions must be positive");
        }
        if (Math.multiplyExact(widthBlocks, 8) != prepared.width
                || Math.multiplyExact(heightBlocks, 8) != prepared.height) {
            throw new IllegalArgumentException(
                    "block dimensions do not match the prepared component plane");
        }
        if (naturalIndices == null || naturalIndices.length == 0
                || naturalIndices.length > NATURAL_COEFFICIENT_COUNT) {
            throw new IllegalArgumentException(
                    "naturalIndices must contain between 1 and 64 entries");
        }
        for (int k : naturalIndices) {
            if (k < 0 || k >= NATURAL_COEFFICIENT_COUNT) {
                throw new IllegalArgumentException(
                        "natural coefficient index out of range: " + k);
            }
        }
        Math.multiplyExact(
                Math.multiplyExact(widthBlocks, heightBlocks),
                naturalIndices.length);
    }

    private static void verifyParity(double[] cpu, double[] accelerated, String label) {
        if (cpu.length != accelerated.length) {
            throw new IllegalStateException(
                    "CPU/" + label + " SI cost length mismatch: " + cpu.length
                            + " vs " + accelerated.length);
        }

        // frozen base-cost sort and its 64-candidate windows from these costs,
        // so a one-ulp GPU difference that a 5e-11 relative tolerance accepts
        // can reorder two carriers and make a GPU-embedded message
        // unrecoverable by a CPU extractor.  RAP_GPU_REL_TOL/RAP_GPU_ABS_TOL
        // are no longer consulted.
        for (int i = 0; i < cpu.length; i++) {
            double a = cpu[i];
            double b = accelerated[i];
            if (Double.doubleToLongBits(a) != Double.doubleToLongBits(b)) {
                double absolute = Math.abs(a - b);
                throw new IllegalStateException(
                        "CPU/" + label + " SI cost is not bit-identical at index " + i
                                + " of " + cpu.length + ": cpu=" + a
                                + ", accelerated=" + b + ", abs=" + absolute
                                + ", rel=" + absolute / Math.max(1.0, Math.abs(a))
                                + ". Check that the accelerator library was built with"
                                + " --fmad=false (CUDA) or -ffp-contract=off (HIP).");
            }
        }
        if (Boolean.parseBoolean(System.getProperty(
                "rap.gpu.verifyVerbose",
                System.getenv().getOrDefault("RAP_GPU_VERIFY_VERBOSE", "false")))) {
            System.out.printf(Locale.ROOT,
                    "SI CPU/%s parity PASS: n=%d bit-identical%n",
                    label, cpu.length);
        }
    }

    private static int readPositiveInt(
            String propertyName, String envName, int defaultValue) {
        String value = System.getProperty(propertyName);
        if (value == null || value.isBlank()) value = System.getenv(envName);
        if (value == null || value.isBlank()) return defaultValue;
        try {
            int parsed = Integer.parseInt(value.trim());
            return parsed > 0 ? parsed : defaultValue;
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static void validatePlane(double[][] plane) {
        if (plane == null || plane.length == 0 || plane[0] == null || plane[0].length == 0) {
            throw new IllegalArgumentException("plane must be nonempty");
        }
        int width = plane[0].length;
        for (double[] row : plane) {
            if (row == null || row.length != width) {
                throw new IllegalArgumentException("plane must be rectangular");
            }
        }
    }

    private static void validateBasis(double[][] basis) {
        if (basis == null || basis.length != 8) {
            throw new IllegalArgumentException("impactBasis must be 8x8");
        }
        for (double[] row : basis) {
            if (row == null || row.length != 8) {
                throw new IllegalArgumentException("impactBasis must be 8x8");
            }
        }
    }

    private static void validateBlock(int pixelX, int pixelY, int width, int height) {
        if (pixelX < 0 || pixelY < 0 || pixelX + 8 > width || pixelY + 8 > height) {
            throw new IllegalArgumentException("8x8 impact block lies outside the component grid");
        }
    }
}
