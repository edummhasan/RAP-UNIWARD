package thesis.rap.alg;

import java.util.Arrays;
import java.util.Random;

/** Regression tests for the CPU/HIP/CUDA batch execution layer. */
public final class SiUniwardBatchBackendSelfTest {
    private SiUniwardBatchBackendSelfTest() {}

    public static void main(String[] args) {
        testBatchedDctMatchesScalar();
        testCpuBatchMatchesScalarCosts();
        testHipBatchWhenAvailable();
        testCudaBatchWhenAvailable();
        System.out.println("SiUniwardBatchBackendSelfTest: ALL TESTS PASSED");
    }

    private static void testBatchedDctMatchesScalar() {
        double[][] plane = deterministicPlane(32, 24, 2026091201L);
        int[] indices = JpegMath.targetExpandedMidBandNaturalIndices();
        int px = 16;
        int py = 8;
        double[] batch = JpegMath.forwardDctCoefficients(
                plane, px, py, indices);
        for (int i = 0; i < indices.length; i++) {
            int k = indices[i];
            double scalar = JpegMath.forwardDctCoefficient(
                    plane, px, py,
                    JpegMath.uFromNaturalIndex(k),
                    JpegMath.vFromNaturalIndex(k));
            long a = Double.doubleToLongBits(batch[i]);
            long b = Double.doubleToLongBits(scalar);
            if (a != b) {
                double relative = Math.abs(batch[i] - scalar)
                        / Math.max(1.0, Math.abs(scalar));
                if (relative > 2.0e-15) {
                    throw new AssertionError(
                            "batched DCT differs from scalar at k=" + k
                                    + ": batch=" + batch[i]
                                    + ", scalar=" + scalar
                                    + ", rel=" + relative);
                }
            }
        }
    }

    private static void testCpuBatchMatchesScalarCosts() {
        int width = 32;
        int height = 24;
        double[][] plane = deterministicPlane(width, height, 2026091202L);
        int[] qTable = new int[64];
        for (int k = 0; k < qTable.length; k++) qTable[k] = 1 + (k % 47);
        int[] indices = JpegMath.targetExpandedMidBandNaturalIndices();

        SiUniwardCostEngine.PreparedPlane prepared =
                SiUniwardCostEngine.prepare(plane, qTable);
        double[] batch = SiUniwardCostEngine.residualCostsCpu(
                prepared, width / 8, height / 8, indices);

        int cursor = 0;
        for (int by = 0; by < height / 8; by++) {
            for (int bx = 0; bx < width / 8; bx++) {
                int px = bx * 8;
                int py = by * 8;
                for (int k : indices) {
                    double scalar = SiUniwardCostEngine.residualCost(
                            prepared, px, py, k);
                    double actual = batch[cursor++];
                    if (Double.doubleToLongBits(scalar)
                            != Double.doubleToLongBits(actual)) {
                        throw new AssertionError(
                                "CPU batch/scalar mismatch at bx=" + bx
                                        + ", by=" + by + ", k=" + k
                                        + ": batch=" + actual
                                        + ", scalar=" + scalar);
                    }
                }
            }
        }
        if (cursor != batch.length) {
            throw new AssertionError("batch traversal count mismatch");
        }
    }

    private static void testHipBatchWhenAvailable() {
        if (!SiUniwardCostEngine.hipAvailable()) {
            System.out.println("SiUniwardBatchBackendSelfTest: HIP SKIP (library/device unavailable)");
            return;
        }
        int width = 64;
        int height = 64;
        double[][] plane = deterministicPlane(width, height, 2026091203L);
        int[] qTable = new int[64];
        Arrays.fill(qTable, 17);
        int[] indices = JpegMath.targetExpandedMidBandNaturalIndices();
        SiUniwardCostEngine.PreparedPlane prepared =
                SiUniwardCostEngine.prepare(plane, qTable);
        double[] cpu = SiUniwardCostEngine.residualCostsCpu(
                prepared, width / 8, height / 8, indices);
        double[] hip = SiUniwardCostEngine.residualCostsHip(
                prepared, width / 8, height / 8, indices);
        assertParity(cpu, hip, "HIP", 5.0e-11, 1.0e-9);
        System.out.println("SiUniwardBatchBackendSelfTest: HIP parity PASS on "
                + SiUniwardCostEngine.hipDeviceName());
    }

    private static void testCudaBatchWhenAvailable() {
        if (!SiUniwardCostEngine.cudaAvailable()) {
            System.out.println("SiUniwardBatchBackendSelfTest: CUDA SKIP (library/device unavailable)");
            return;
        }
        int width = 64;
        int height = 64;
        double[][] plane = deterministicPlane(width, height, 2026091501L);
        int[] qTable = new int[64];
        Arrays.fill(qTable, 17);
        int[] indices = JpegMath.targetExpandedMidBandNaturalIndices();
        SiUniwardCostEngine.PreparedPlane prepared =
                SiUniwardCostEngine.prepare(plane, qTable);
        double[] cpu = SiUniwardCostEngine.residualCostsCpu(
                prepared, width / 8, height / 8, indices);
        double[] cuda = SiUniwardCostEngine.residualCostsCuda(
                prepared, width / 8, height / 8, indices);
        assertParity(cpu, cuda, "CUDA", 5.0e-11, 1.0e-9);
        System.out.println("SiUniwardBatchBackendSelfTest: CUDA parity PASS on "
                + SiUniwardCostEngine.cudaDeviceName());
    }

    private static void assertParity(
            double[] cpu, double[] accelerated, String label,
            double relTol, double absTol) {
        if (cpu.length != accelerated.length) {
            throw new AssertionError("CPU/" + label + " length mismatch");
        }

        // difference at all can reorder carriers for a CPU extractor.
        double maxRelative = 0.0;
        for (int i = 0; i < cpu.length; i++) {
            double absolute = Math.abs(cpu[i] - accelerated[i]);
            double relative = absolute / Math.max(1.0, Math.abs(cpu[i]));
            if (Double.isFinite(relative)) maxRelative = Math.max(maxRelative, relative);
            if (Double.doubleToLongBits(cpu[i])
                    != Double.doubleToLongBits(accelerated[i])) {
                throw new AssertionError(
                        "CPU/" + label + " not bit-identical at i=" + i
                                + ": cpu=" + cpu[i]
                                + ", accelerated=" + accelerated[i]
                                + ", abs=" + absolute + ", rel=" + relative
                                + (absolute <= absTol || relative <= relTol
                                ? " (within the retired RAP-UNIWARD tolerance)" : ""));
            }
        }
        System.out.printf("CPU/%s batch max relative error: %.3e%n",
                label, maxRelative);
    }

    private static double[][] deterministicPlane(
            int width, int height, long seed) {
        Random random = new Random(seed);
        double[][] plane = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                plane[y][x] = 128.0
                        + 31.0 * Math.sin(0.071 * x)
                        + 23.0 * Math.cos(0.053 * y)
                        + 5.0 * random.nextGaussian();
            }
        }
        return plane;
    }
}
