package thesis.rap.alg;

import java.util.Random;

/** Independent dense-grid regression check for the UNIWARD base-cost operator. */
public final class SiUniwardCostEngineSelfTest {
    private static final double SIGMA = 1.0 / 64.0;
    private static final int N = 16;
    private static final int ANCHOR = 7;
    private static final int PAD = 16;

    private static final double[] LP = {
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

    private SiUniwardCostEngineSelfTest() {}

    public static void main(String[] args) {
        testSymmetricIndex();
        testOrthonormalIdctEnergy();
        testImpactWindowMatchesMeasuredResidualChange();
        testDenseCostAgreement();
        System.out.println("SiUniwardCostEngineSelfTest: ALL TESTS PASSED");
    }

    /** Verifies the Parseval premise used by the marginal RGB-MSE proxy. */
    private static void testOrthonormalIdctEnergy() {
        int[] quantizationSteps = {1, 7, 37, 255};
        double worstRelativeError = 0.0;
        for (int q : quantizationSteps) {
            for (int v = 0; v < 8; v++) {
                for (int u = 0; u < 8; u++) {
                    double[][] basis = JpegMath.idctBasis(u, v, q);
                    double energy = 0.0;
                    for (double[] row : basis) {
                        for (double value : row) energy += value * value;
                    }
                    double expected = (double) q * q;
                    double relative = Math.abs(energy - expected)
                            / Math.max(1.0, expected);
                    worstRelativeError = Math.max(
                            worstRelativeError, relative);
                    if (relative > 2.0e-14) {
                        throw new AssertionError(
                                "IDCT basis violates Parseval energy at u="
                                        + u + ", v=" + v + ", q=" + q
                                        + ": actual=" + energy
                                        + ", expected=" + expected);
                    }
                }
            }
        }
        System.out.printf(
                "IDCT Parseval regression max relative error: %.3e%n",
                worstRelativeError);
    }

    /**
     * Physical alignment check.
     *
     * <p>The dense regression below and the engine share the same window-offset
     * expression, so on its own it cannot detect a misaligned impact window.
     * This test does not assume any offset. It perturbs the plane by the exact
     * spatial impact of a +1 quantized-coefficient change, measures how the
     * directional residual actually changes, and requires the engine's
     * precomputed 23x23 impact response to coincide with that measured change
     * at the offset the engine uses.
     *
     * <p>With the corrected offset the two agree to machine precision; with the
     * published DDE off-by-one window they differ by roughly the full magnitude
     * of the impact response.</p>
     *
     * <p>Only interior blocks are used. For a block near the plane edge the
     * symmetric padding also mirrors the perturbation into the border, so the
     * measured residual change legitimately exceeds the single direct impact
     * response. UNIWARD sums only the direct response, so a boundary block
     * cannot discriminate between window offsets.</p>
     */
    private static void testImpactWindowMatchesMeasuredResidualChange() {
        int width = 64;
        int height = 64;
        double[][] plane = new double[height][width];
        Random random = new Random(987654321L);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                plane[y][x] = 128.0 + 40.0 * Math.sin(0.31 * x)
                        + 25.0 * Math.cos(0.23 * y) + 8.0 * random.nextGaussian();
            }
        }

        int pad = SiUniwardCostEngine.padding();
        int span = SiUniwardCostEngine.impactWindowSize();
        int offset = SiUniwardCostEngine.IMPACT_WINDOW_OFFSET;

        // Interior blocks only: at least PAD samples from every plane edge.
        int[][] blocks = {{16, 16}, {24, 32}, {32, 24}, {40, 40}};
        int[][] modes = {{5, 0, 24}, {4, 4, 68}, {2, 7, 99}, {3, 3, 57}};

        double worst = 0.0;
        double impactScale = 0.0;
        for (int[] block : blocks) {
            for (int[] mode : modes) {
                int px = block[0];
                int py = block[1];
                double[][] basis = JpegMath.idctBasis(mode[0], mode[1], mode[2]);

                double[][] stego = new double[height][];
                for (int y = 0; y < height; y++) stego[y] = plane[y].clone();
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) stego[py + y][px + x] += basis[y][x];
                }

                for (int f = 0; f < 3; f++) {
                    double[][] cover = SiUniwardCostEngine.directionalResidual(plane, f);
                    double[][] changed = SiUniwardCostEngine.directionalResidual(stego, f);
                    double[][] impact = SiUniwardCostEngine.unitImpactResponse(basis, f);
                    for (int iy = 0; iy < span; iy++) {
                        for (int ix = 0; ix < span; ix++) {
                            double measured = changed[py + offset + iy][px + offset + ix]
                                    - cover[py + offset + iy][px + offset + ix];
                            worst = Math.max(worst, Math.abs(measured - impact[iy][ix]));
                            impactScale = Math.max(impactScale, Math.abs(impact[iy][ix]));
                        }
                    }
                    // Nothing outside the declared window may move.
                    for (int y = 0; y < cover.length; y++) {
                        boolean insideRows = y >= py + offset && y < py + offset + span;
                        for (int x = 0; x < cover[0].length; x++) {
                            boolean insideCols = x >= px + offset && x < px + offset + span;
                            if (insideRows && insideCols) continue;
                            double leak = Math.abs(changed[y][x] - cover[y][x]);
                            if (leak > 1e-9) {
                                throw new AssertionError(
                                        "Residual changed outside the impact window at ("
                                                + x + "," + y + "): " + leak
                                                + "; window offset=" + offset
                                                + ", pad=" + pad);
                            }
                        }
                    }
                }
            }
        }

        double tolerance = 1e-9 * Math.max(1.0, impactScale);
        if (worst > tolerance) {
            throw new AssertionError(
                    "Impact window is misaligned with the measured residual change: max error="
                            + worst + " against impact magnitude " + impactScale
                            + " (offset=" + offset + ", alignment="
                            + SiUniwardCostEngine.IMPACT_WINDOW_ALIGNMENT + ")");
        }
        System.out.printf(
                "UNIWARD impact-window alignment=%s offset=%d max|measured-impact|=%.3e%n",
                SiUniwardCostEngine.IMPACT_WINDOW_ALIGNMENT, offset, worst);
    }

    private static void testSymmetricIndex() {
        assertEquals(0, SiUniwardCostEngine.symmetricIndex(-1, 4));
        assertEquals(1, SiUniwardCostEngine.symmetricIndex(-2, 4));
        assertEquals(2, SiUniwardCostEngine.symmetricIndex(-3, 4));
        assertEquals(3, SiUniwardCostEngine.symmetricIndex(-4, 4));
        assertEquals(3, SiUniwardCostEngine.symmetricIndex(4, 4));
        assertEquals(2, SiUniwardCostEngine.symmetricIndex(5, 4));
        assertEquals(1, SiUniwardCostEngine.symmetricIndex(6, 4));
        assertEquals(0, SiUniwardCostEngine.symmetricIndex(7, 4));
    }

    private static void testDenseCostAgreement() {
        int width = 32;
        int height = 24;
        double[][] plane = new double[height][width];
        Random random = new Random(20260715L);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                plane[y][x] = 16.0 + 220.0 * random.nextDouble()
                        + 7.0 * Math.sin(0.17 * x)
                        - 5.0 * Math.cos(0.11 * y);
            }
        }

        int[][] cases = {
                {0, 0, 5, 0, 17},       // top-left boundary block
                {8, 8, 4, 4, 23},       // interior retained half-path mode geometry
                {24, 16, 7, 2, 11},     // bottom-right boundary block
                {16, 0, 3, 5, 37}       // top boundary, nonseparable frequency pair
        };

        double maxRelativeError = 0.0;
        for (int[] c : cases) {
            int px = c[0], py = c[1], u = c[2], v = c[3], q = c[4];
            double[][] basis = JpegMath.idctBasis(u, v, q);
            double actual = SiUniwardCostEngine.residualCost(plane, px, py, basis);
            double expected = denseReferenceCost(plane, px, py, basis);
            double relative = Math.abs(actual - expected) / Math.max(1.0, Math.abs(expected));
            maxRelativeError = Math.max(maxRelativeError, relative);
            if (relative > 2e-12) {
                throw new AssertionError("cost mismatch at px=" + px + ", py=" + py
                        + ", u=" + u + ", v=" + v + ": actual=" + actual
                        + ", expected=" + expected + ", relative=" + relative);
            }
            double minus = SiUniwardCostEngine.rhoMinus(plane, px, py, basis);
            if (Math.abs(minus - actual) > 1e-12 * Math.max(1.0, Math.abs(actual))) {
                throw new AssertionError("base +/- costs must be symmetric");
            }

            int[] qTable = new int[64];
            java.util.Arrays.fill(qTable, 1);
            int k = JpegMath.naturalIndex(u, v);
            qTable[k] = q;
            SiUniwardCostEngine.PreparedPlane prepared =
                    SiUniwardCostEngine.prepare(plane, qTable);
            double preparedCost = SiUniwardCostEngine.residualCost(prepared, px, py, k);
            if (Math.abs(preparedCost - actual) > 2e-12 * Math.max(1.0, Math.abs(actual))) {
                throw new AssertionError("prepared and compatibility cost paths disagree");
            }
        }
        System.out.printf("UNIWARD dense cost regression max relative error: %.3e%n", maxRelativeError);
    }

    private static double denseReferenceCost(double[][] plane, int px, int py, double[][] basis) {
        double[] hp = highPass(LP);
        double[][][] filters = {
                outer(LP, hp),
                outer(hp, LP),
                outer(hp, hp)
        };

        int h = plane.length;
        int w = plane[0].length;
        double[][] padded = new double[h + 2 * PAD][w + 2 * PAD];
        for (int y = 0; y < padded.length; y++) {
            for (int x = 0; x < padded[0].length; x++) {
                padded[y][x] = plane[sym(y - PAD, h)][sym(x - PAD, w)];
            }
        }
        // The window offset is taken from the engine only for this dense
        // cross-check of the separable filtering path.  The alignment itself is
        // validated independently against the measured residual change in
        // testImpactWindowMatchesMeasuredResidualChange().
        int offset = SiUniwardCostEngine.IMPACT_WINDOW_OFFSET;
        double total = 0.0;
        for (double[][] kernel : filters) {
            for (int iy = 0; iy < 23; iy++) {
                int outY = py + offset + iy;
                for (int ix = 0; ix < 23; ix++) {
                    int outX = px + offset + ix;
                    double residual = 0.0;
                    double impact = 0.0;
                    for (int fy = 0; fy < N; fy++) {
                        int sourceY = outY + fy - ANCHOR;
                        int basisY = iy + fy - (N - 1);
                        for (int fx = 0; fx < N; fx++) {
                            int sourceX = outX + fx - ANCHOR;
                            int basisX = ix + fx - (N - 1);
                            double coefficient = kernel[fy][fx];
                            residual += padded[sourceY][sourceX] * coefficient;
                            if (basisY >= 0 && basisY < 8 && basisX >= 0 && basisX < 8) {
                                impact += basis[basisY][basisX] * coefficient;
                            }
                        }
                    }
                    total += Math.abs(impact) / (SIGMA + Math.abs(residual));
                }
            }
        }
        return total;
    }

    private static int sym(int index, int length) {
        if (length <= 1) return 0;
        int i = index;
        while (i < 0 || i >= length) {
            i = i < 0 ? -i - 1 : 2 * length - i - 1;
        }
        return i;
    }

    private static double[] highPass(double[] lowPass) {
        double[] hp = new double[lowPass.length];
        for (int i = 0; i < lowPass.length; i++) {
            hp[i] = ((i & 1) == 0 ? 1.0 : -1.0) * lowPass[lowPass.length - 1 - i];
        }
        return hp;
    }

    private static double[][] outer(double[] a, double[] b) {
        double[][] out = new double[a.length][b.length];
        for (int y = 0; y < a.length; y++) {
            for (int x = 0; x < b.length; x++) out[y][x] = a[y] * b[x];
        }
        return out;
    }

    private static void assertEquals(int expected, int actual) {
        if (expected != actual) {
            throw new AssertionError("expected=" + expected + ", actual=" + actual);
        }
    }
}
