package thesis.rap.alg;

import java.util.Arrays;
import java.util.Locale;

/** Verifies that the configured backend actually increments the expected path counters. */
public final class BackendModeContractSelfTest {
    private BackendModeContractSelfTest() {}

    public static void main(String[] args) {
        String mode = SiUniwardCostEngine.configuredBackendName();
        double[][] plane = new double[64][64];
        for (int y = 0; y < plane.length; y++) {
            for (int x = 0; x < plane[y].length; x++) {
                plane[y][x] = 128.0
                        + 27.0 * Math.sin(0.041 * x)
                        + 19.0 * Math.cos(0.037 * y);
            }
        }
        int[] quantization = new int[64];
        Arrays.fill(quantization, 17);
        int[] indices = JpegMath.targetExpandedMidBandNaturalIndices();
        SiUniwardCostEngine.PreparedPlane prepared =
                SiUniwardCostEngine.prepare(plane, quantization);

        SiUniwardCostEngine.resetBackendPerformance();
        double[] costs = SiUniwardCostEngine.residualCosts(
                prepared, 8, 8, indices);
        if (costs.length != 8 * 8 * indices.length) {
            throw new AssertionError("unexpected output length: " + costs.length);
        }
        for (int i = 0; i < costs.length; i++) {
            if (!Double.isFinite(costs[i]) || costs[i] < 0.0) {
                throw new AssertionError("invalid cost at " + i + ": " + costs[i]);
            }
        }

        SiUniwardCostEngine.BackendPerformance p =
                SiUniwardCostEngine.backendPerformance();
        switch (mode) {
            case "CPU" -> require(p.cpuBatches() == 1 && p.hipBatches() == 0
                    && p.cudaBatches() == 0 && p.verifyBatches() == 0, mode, p);
            case "HIP" -> require(p.cpuBatches() == 0 && p.hipBatches() == 1
                    && p.cudaBatches() == 0 && p.verifyBatches() == 0, mode, p);
            case "CUDA" -> require(p.cpuBatches() == 0 && p.hipBatches() == 0
                    && p.cudaBatches() == 1 && p.verifyBatches() == 0, mode, p);
            case "VERIFY_HIP" -> require(p.cpuBatches() == 1 && p.hipBatches() == 1
                    && p.cudaBatches() == 0 && p.verifyBatches() == 1, mode, p);
            case "VERIFY_CUDA" -> require(p.cpuBatches() == 1 && p.hipBatches() == 0
                    && p.cudaBatches() == 1 && p.verifyBatches() == 1, mode, p);
            case "VERIFY" -> require(p.cpuBatches() == 1
                    && p.hipBatches() + p.cudaBatches() == 1
                    && p.verifyBatches() == 1, mode, p);
            case "AUTO" -> require(p.cpuBatches() + p.hipBatches()
                    + p.cudaBatches() == 1, mode, p);
            default -> throw new AssertionError("unhandled backend: " + mode);
        }
        System.out.printf(Locale.ROOT,
                "BackendModeContractSelfTest: PASS mode=%s cpu=%d hip=%d cuda=%d verify=%d fallback=%d%n",
                mode, p.cpuBatches(), p.hipBatches(), p.cudaBatches(),
                p.verifyBatches(), p.autoFallbacks());
    }

    private static void require(
            boolean condition, String mode,
            SiUniwardCostEngine.BackendPerformance performance) {
        if (!condition) {
            throw new AssertionError(
                    "backend dispatch counter mismatch for " + mode + ": " + performance);
        }
    }
}
