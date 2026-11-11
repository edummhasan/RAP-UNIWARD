package thesis.rap.alg;

/**
 * JPEG 8x8 DCT helper methods.
 *
 * Indexing convention used in this project:
 * - Libjpeg coefficient arrays are stored in natural row-major order: k = row * 8 + col.
 * - DCT coordinates are represented as (u, v), where u = column/frequency-x and v = row/frequency-y.
 * - The implementation uses zero-based JPEG zig-zag rank ZZ0(r).
 * - The statistical Cb/Cr research window is ZZ0(16:47), equivalent to one-based ZZ1(17:48).
 * - The avoided low/structural region is ZZ0(0:15), and the high-tail diagnostic region is ZZ0(48:63).
 */
public final class JpegMath {
    private JpegMath() {}

    /** Finite sentinel used by Java and the native STC adapter for forbidden changes. */
    public static final double WET_COST = 1e13;

    /**
     * Caps a cost that has already been established as a codec-legal
     * transition.  The value immediately below the wet sentinel is used so a
     * large-but-legal transition cannot be reclassified as prohibited merely
     * by numerical scaling.
     */
    static double capComputedLegalCost(double cost) {
        if (Double.isNaN(cost) || cost < 0.0) return WET_COST;
        return Math.min(cost, WET_COST - 1.0);
    }

    /**
     * Largest AC coefficient magnitude a baseline 8-bit JPEG entropy coder can
     * represent.  libjpeg's {@code MAX_COEF_BITS} is 10 for 8-bit samples, so a
     * magnitude of 1024 or more aborts Huffman encoding with
     * {@code JERR_BAD_DCT_COEF}.  A direction that would leave this range is
     * therefore not merely expensive, it is unwritable, and it is wet.
     */
    public static final int BASELINE_MAX_AC_MAGNITUDE = 1023;
    /** Additive floor used by the public SI-UNIWARD reference implementation. */
    public static final double SI_COST_FLOOR = 1e-4;
    /** Literature-comparison threshold; it is never an RAP wetting rule. */
    public static final double NEAR_HALF_DIAGNOSTIC_THRESHOLD = 0.4999;

    /** Expanded research candidate window used for embedding. */
    public static final int TARGET_ZZ_START_RANK = 16;
    public static final int TARGET_ZZ_END_RANK = 47;

    /** High-frequency tail used for diagnostics only, not for embedding. */
    public static final int HIGH_TAIL_ZZ_START_RANK = 48;
    public static final int HIGH_TAIL_ZZ_END_RANK = 63;

    /** Low/structural region avoided by the candidate policy. */
    public static final int LOW_REGION_ZZ_START_RANK = 0;
    public static final int LOW_REGION_ZZ_END_RANK = TARGET_ZZ_START_RANK - 1;

    /**
     * Standard zero-based JPEG zig-zag order.
     * Each entry is {row, col}. In the rest of the Java code, u = col and v = row.
     */
    private static final int[][] ZIGZAG_ROW_COL = {
            {0,0}, {0,1}, {1,0}, {2,0}, {1,1}, {0,2}, {0,3}, {1,2},
            {2,1}, {3,0}, {4,0}, {3,1}, {2,2}, {1,3}, {0,4}, {0,5},
            {1,4}, {2,3}, {3,2}, {4,1}, {5,0}, {6,0}, {5,1}, {4,2},
            {3,3}, {2,4}, {1,5}, {0,6}, {0,7}, {1,6}, {2,5}, {3,4},
            {4,3}, {5,2}, {6,1}, {7,0}, {7,1}, {6,2}, {5,3}, {4,4},
            {3,5}, {2,6}, {1,7}, {2,7}, {3,6}, {4,5}, {5,4}, {6,3},
            {7,2}, {7,3}, {6,4}, {5,5}, {4,6}, {3,7}, {4,7}, {5,6},
            {6,5}, {7,4}, {7,5}, {6,6}, {5,7}, {6,7}, {7,6}, {7,7}
    };

    private static final int[] NATURAL_TO_ZIGZAG = buildNaturalToZigzag();

    /** Cached StrictMath constants keep every DCT evaluation deterministic. */
    private static final double[] DCT_ALPHA = buildDctAlpha();
    private static final double[][] DCT_COSINE = buildDctCosine();

    /** Exact StrictMath-derived DCT alpha constants for accelerator parity. */
    public static double[] dctAlphaForAccelerator() {
        return java.util.Arrays.copyOf(DCT_ALPHA, DCT_ALPHA.length);
    }

    /**
     * Exact StrictMath-derived cosine table flattened as [sample*8+frequency].
     * Accelerators consume these host-computed binary64 constants rather than
     * recomputing transcendental values with a device math library.
     */
    public static double[] dctCosineForAccelerator() {
        double[] out = new double[64];
        for (int sample = 0; sample < 8; sample++) {
            System.arraycopy(DCT_COSINE[sample], 0, out, sample * 8, 8);
        }
        return out;
    }

    static {
        validateZigzagTableOrThrow();
    }

    /** Libjpeg natural coefficient index: row-major order, where u=column and v=row. */
    public static int naturalIndex(int u, int v) {
        if (u < 0 || u > 7 || v < 0 || v > 7) {
            throw new IllegalArgumentException("DCT coordinate out of range: u=" + u + ", v=" + v);
        }
        return v * 8 + u;
    }

    public static int uFromNaturalIndex(int k) {
        checkNaturalIndex(k);
        return k % 8;
    }

    public static int vFromNaturalIndex(int k) {
        checkNaturalIndex(k);
        return k / 8;
    }

    /** Returns the libjpeg natural index for a zero-based JPEG zig-zag rank. */
    public static int naturalIndexFromZigzagRank(int rank) {
        checkZigzagRank(rank);
        int row = ZIGZAG_ROW_COL[rank][0];
        int col = ZIGZAG_ROW_COL[rank][1];
        return naturalIndex(col, row);
    }

    /** Returns the zero-based JPEG zig-zag rank for a natural-order coefficient index. */
    public static int zigzagRankFromNaturalIndex(int k) {
        checkNaturalIndex(k);
        return NATURAL_TO_ZIGZAG[k];
    }

    /** Returns the zero-based JPEG zig-zag rank for DCT coordinate (u=col, v=row). */
    public static int zigzagIndex(int u, int v) {
        return zigzagRankFromNaturalIndex(naturalIndex(u, v));
    }

    /** Preferred name for the statistical research candidate window ZZ0(16:47). */
    public static boolean isTargetExpandedMidBandRank(int rank) {
        return rank >= TARGET_ZZ_START_RANK && rank <= TARGET_ZZ_END_RANK;
    }

    public static boolean isTargetExpandedMidBandNaturalIndex(int k) {
        return isTargetExpandedMidBandRank(zigzagRankFromNaturalIndex(k));
    }


    public static boolean isHighTailRank(int rank) {
        return rank >= HIGH_TAIL_ZZ_START_RANK && rank <= HIGH_TAIL_ZZ_END_RANK;
    }

    public static boolean isHighTailNaturalIndex(int k) {
        return isHighTailRank(zigzagRankFromNaturalIndex(k));
    }

    public static int[] targetExpandedMidBandNaturalIndices() {
        int[] out = new int[TARGET_ZZ_END_RANK - TARGET_ZZ_START_RANK + 1];
        for (int rank = TARGET_ZZ_START_RANK; rank <= TARGET_ZZ_END_RANK; rank++) {
            out[rank - TARGET_ZZ_START_RANK] = naturalIndexFromZigzagRank(rank);
        }
        return out;
    }


    public static int[] highTailNaturalIndices() {
        int[] out = new int[HIGH_TAIL_ZZ_END_RANK - HIGH_TAIL_ZZ_START_RANK + 1];
        for (int rank = HIGH_TAIL_ZZ_START_RANK; rank <= HIGH_TAIL_ZZ_END_RANK; rank++) {
            out[rank - HIGH_TAIL_ZZ_START_RANK] = naturalIndexFromZigzagRank(rank);
        }
        return out;
    }

    public static String targetBandLabel() {
        return "ZZ0(" + TARGET_ZZ_START_RANK + ":" + TARGET_ZZ_END_RANK + ")";
    }

    public static String targetBandOneBasedLabel() {
        return "ZZ1(" + (TARGET_ZZ_START_RANK + 1) + ":" + (TARGET_ZZ_END_RANK + 1) + ")";
    }

    public static String lowRegionLabel() {
        return "ZZ0(" + LOW_REGION_ZZ_START_RANK + ":" + LOW_REGION_ZZ_END_RANK + ")";
    }

    public static String highTailLabel() {
        return "ZZ0(" + HIGH_TAIL_ZZ_START_RANK + ":" + HIGH_TAIL_ZZ_END_RANK + ")";
    }

    public static String describeZigzagRank(int rank) {
        int k = naturalIndexFromZigzagRank(rank);
        return "ZZ0(" + rank + ")=(row=" + vFromNaturalIndex(k) + ", col=" + uFromNaturalIndex(k) + ")";
    }

    public static String targetBandCoordinateList() {
        StringBuilder sb = new StringBuilder();
        for (int rank = TARGET_ZZ_START_RANK; rank <= TARGET_ZZ_END_RANK; rank++) {
            if (rank > TARGET_ZZ_START_RANK) sb.append(", ");
            int k = naturalIndexFromZigzagRank(rank);
            sb.append(rank)
              .append("->(")
              .append(vFromNaturalIndex(k))
              .append(",")
              .append(uFromNaturalIndex(k))
              .append(")");
        }
        return sb.toString();
    }

    public static double[][] idctBasis(int u, int v, int q) {
        if (u < 0 || u > 7 || v < 0 || v > 7) {
            throw new IllegalArgumentException(
                    "DCT coordinate out of range: u=" + u + ", v=" + v);
        }
        double[][] b = new double[8][8];
        double au = DCT_ALPHA[u];
        double av = DCT_ALPHA[v];
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                b[y][x] = 0.25 * au * av * q
                        * DCT_COSINE[x][u]
                        * DCT_COSINE[y][v];
            }
        }
        return b;
    }

    /** JPEG-compatible orthonormal forward-DCT coefficient from an 8x8 block. */
    public static double forwardDctCoefficient(
            double[][] plane, int pixelX, int pixelY, int u, int v) {
        if (u < 0 || u > 7 || v < 0 || v > 7) {
            throw new IllegalArgumentException(
                    "DCT coordinate out of range: u=" + u + ", v=" + v);
        }
        double sum = 0.0;
        for (int y = 0; y < 8; y++) {
            int sy = clamp(pixelY + y, 0, plane.length - 1);
            for (int x = 0; x < 8; x++) {
                int sx = clamp(pixelX + x, 0, plane[0].length - 1);
                double shifted = plane[sy][sx] - 128.0;
                sum += shifted * DCT_COSINE[x][u] * DCT_COSINE[y][v];
            }
        }
        return 0.25 * DCT_ALPHA[u] * DCT_ALPHA[v] * sum;
    }

    /**
     * Computes a block's requested DCT coefficients together.  For every
     * coefficient the accumulation still visits pixels in the same y/x order
     * as {@link #forwardDctCoefficient}, but the source sample is loaded only
     * once and all cosine values come from the deterministic cache.
     */
    public static double[] forwardDctCoefficients(
            double[][] plane,
            int pixelX,
            int pixelY,
            int[] naturalIndices
    ) {
        if (plane == null || plane.length == 0 || plane[0] == null
                || plane[0].length == 0) {
            throw new IllegalArgumentException("plane must be nonempty");
        }
        if (naturalIndices == null || naturalIndices.length == 0) {
            throw new IllegalArgumentException("naturalIndices must be nonempty");
        }
        int n = naturalIndices.length;
        int[] us = new int[n];
        int[] vs = new int[n];
        double[] sums = new double[n];
        for (int i = 0; i < n; i++) {
            int k = naturalIndices[i];
            checkNaturalIndex(k);
            us[i] = uFromNaturalIndex(k);
            vs[i] = vFromNaturalIndex(k);
        }
        for (int y = 0; y < 8; y++) {
            int sy = clamp(pixelY + y, 0, plane.length - 1);
            for (int x = 0; x < 8; x++) {
                int sx = clamp(pixelX + x, 0, plane[0].length - 1);
                double shifted = plane[sy][sx] - 128.0;
                for (int i = 0; i < n; i++) {
                    sums[i] += shifted
                            * DCT_COSINE[x][us[i]]
                            * DCT_COSINE[y][vs[i]];
                }
            }
        }
        for (int i = 0; i < n; i++) {
            sums[i] *= 0.25 * DCT_ALPHA[us[i]] * DCT_ALPHA[vs[i]];
        }
        return sums;
    }

    /**
     * Computes e=F/Q-q using the actual quantized reference coefficient.  If the
     * reconstructed component samples do not agree with the encoder's bin, side
     * information is disabled instead of applying a false polarity.
     */
    public static double normalizedRoundingError(double rawDct, int quantStep, int quantizedReference) {
        if (quantStep <= 0) throw new IllegalArgumentException("quantStep must be positive");
        double error = rawDct / quantStep - quantizedReference;
        double tolerance = 1e-7;
        if (!Double.isFinite(error) || error < -0.5 - tolerance || error > 0.5 + tolerance) return 0.0;
        return Math.max(-0.5, Math.min(0.5, error));
    }

    /**
     * Returns {rho(+1), rho(-1)} after the RAP directional side-information modulation.
     * The direction matching sign(e) is discounted by 1-2|e|; the opposite direction remains
     * available unless another RAP constraint makes it wet. This is the declared RAP extension,
     * not a claim that both polarities are legal in the original binary SI-UNIWARD operation.
     */
    public static double[] applySideInformedModulation(double rhoPlus, double rhoMinus, double roundingError) {
        double[] result = new double[2];
        applySideInformedModulation(rhoPlus, rhoMinus, roundingError, result);
        return result;
    }

    /** Allocation-free overload for the sequential candidate construction loop. */
    static void applySideInformedModulation(double rhoPlus, double rhoMinus,
            double roundingError, double[] result) {
        if (roundingError < -0.5 || roundingError > 0.5 || !Double.isFinite(roundingError)) {
            throw new IllegalArgumentException("roundingError must be finite and within [-0.5,0.5]");
        }
        double preferredFactor = Math.max(0.0, 1.0 - 2.0 * Math.abs(roundingError));
        if (roundingError > 0.0) rhoPlus *= preferredFactor;
        else if (roundingError < 0.0) rhoMinus *= preferredFactor;
        rhoPlus = addSiCostFloor(rhoPlus);
        rhoMinus = addSiCostFloor(rhoMinus);
        result[0] = rhoPlus;
        result[1] = rhoMinus;
    }

    /**
     * Exact squared reconstruction-error increment for one quantized unit
     * transition. This is logged for the minimum-perturbation comparison; it
     * does not replace RAP's primary ternary-difference modulation.
     */
    public static double exactReconstructionErrorIncrement(
            double roundingError, int delta
    ) {
        if (!Double.isFinite(roundingError)
                || roundingError < -0.5
                || roundingError > 0.5) {
            throw new IllegalArgumentException(
                    "roundingError must be finite and within [-0.5,0.5]");
        }
        if (delta != -1 && delta != 1) {
            throw new IllegalArgumentException(
                    "delta must be -1 or +1");
        }
        return 1.0 - 2.0 * delta * roundingError;
    }

    private static double addSiCostFloor(double cost) {
        if (isWetCost(cost)) return WET_COST;
        double adjusted = cost + SI_COST_FLOOR;
        return capComputedLegalCost(adjusted);
    }

    /** True for the original four-mode near-half condition, for logging only. */
    public static boolean isFormerFourModeNearHalf(
            int u, int v, double roundingError
    ) {
        boolean formerFourMode =
                (u == 0 || u == 4) && (v == 0 || v == 4);
        return formerFourMode
                && Double.isFinite(roundingError)
                && Math.abs(roundingError)
                        > NEAR_HALF_DIAGNOSTIC_THRESHOLD;
    }

    public static boolean isWetCost(double cost) {
        return !Double.isFinite(cost) || cost >= WET_COST;
    }

    /** Deterministic half-away-from-zero quantization, including negative exact-half values. */
    public static int roundHalfAwayFromZero(double value) {
        if (!Double.isFinite(value)) throw new IllegalArgumentException("Non-finite quantization input");
        double rounded = value >= 0.0 ? Math.floor(value + 0.5) : Math.ceil(value - 0.5);
        if (rounded < Integer.MIN_VALUE || rounded > Integer.MAX_VALUE) {
            throw new ArithmeticException("Quantized DCT coefficient outside int range: " + rounded);
        }
        return (int) rounded;
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static double[] buildDctAlpha() {
        double[] out = new double[8];
        out[0] = 1.0 / StrictMath.sqrt(2.0);
        for (int i = 1; i < out.length; i++) out[i] = 1.0;
        return out;
    }

    private static double[][] buildDctCosine() {
        double[][] out = new double[8][8];
        for (int sample = 0; sample < 8; sample++) {
            for (int frequency = 0; frequency < 8; frequency++) {
                out[sample][frequency] = StrictMath.cos(
                        ((2 * sample + 1) * frequency * StrictMath.PI) / 16.0);
            }
        }
        return out;
    }

    private static int[] buildNaturalToZigzag() {
        int[] out = new int[64];
        boolean[] seen = new boolean[64];
        for (int rank = 0; rank < ZIGZAG_ROW_COL.length; rank++) {
            int row = ZIGZAG_ROW_COL[rank][0];
            int col = ZIGZAG_ROW_COL[rank][1];
            int natural = naturalIndex(col, row);
            if (seen[natural]) {
                throw new IllegalStateException("Duplicate zig-zag coordinate at natural index " + natural);
            }
            seen[natural] = true;
            out[natural] = rank;
        }
        for (int i = 0; i < seen.length; i++) {
            if (!seen[i]) throw new IllegalStateException("Missing zig-zag coordinate for natural index " + i);
        }
        return out;
    }

    private static void validateZigzagTableOrThrow() {
        // These checkpoints match the methodology figure for the expanded mid-band window.
        assertRankCoordinate(20, 5, 0);
        assertRankCoordinate(21, 6, 0);
        assertRankCoordinate(22, 5, 1);
        assertRankCoordinate(23, 4, 2);
        assertRankCoordinate(24, 3, 3);
        assertRankCoordinate(35, 7, 0);
        assertRankCoordinate(36, 7, 1);
        assertRankCoordinate(39, 4, 4);
        assertRankCoordinate(40, 3, 5);
        assertRankCoordinate(43, 2, 7);
        assertRankCoordinate(44, 3, 6);
        assertRankCoordinate(63, 7, 7);
    }

    private static void assertRankCoordinate(int rank, int expectedRow, int expectedCol) {
        int[] rc = ZIGZAG_ROW_COL[rank];
        if (rc[0] != expectedRow || rc[1] != expectedCol) {
            throw new IllegalStateException(
                    "Zig-zag table mismatch at rank " + rank
                            + ": expected (row=" + expectedRow + ", col=" + expectedCol + ")"
                            + " but got (row=" + rc[0] + ", col=" + rc[1] + ")");
        }
    }

    private static void checkZigzagRank(int rank) {
        if (rank < 0 || rank > 63) {
            throw new IllegalArgumentException("Zig-zag rank out of range: " + rank);
        }
    }

    private static void checkNaturalIndex(int k) {
        if (k < 0 || k > 63) {
            throw new IllegalArgumentException("Natural coefficient index out of range: " + k);
        }
    }
}
