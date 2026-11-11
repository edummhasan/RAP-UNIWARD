package thesis.rap.alg;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

/**
 * Standalone CPU-vs-HIP benchmark for the exact batched SI-UNIWARD base-cost
 * operator.  Preparation is intentionally outside the timed section because it
 * is shared by both backends; the HIP timing includes JNI transfer, kernel
 * execution, synchronization, and result transfer.
 */
public final class SiUniwardCpuHipBenchmark {
    private SiUniwardCpuHipBenchmark() {}

    public static void main(String[] args) {
        int width = args.length >= 1 ? Integer.parseInt(args[0]) : 512;
        int height = args.length >= 2 ? Integer.parseInt(args[1]) : width;
        if (width <= 0 || height <= 0 || width % 8 != 0 || height % 8 != 0) {
            throw new IllegalArgumentException(
                    "width and height must be positive multiples of 8");
        }

        System.out.println("============================================================");
        System.out.println(" RAP SI-UNIWARD CPU vs AMD HIP BENCHMARK");
        System.out.println("============================================================");
        System.out.printf(Locale.ROOT, "Plane             : %dx%d%n", width, height);
        System.out.printf(Locale.ROOT, "Blocks            : %d%n", (width / 8) * (height / 8));
        System.out.printf(Locale.ROOT, "Band              : %s (%d coefficients)%n",
                JpegMath.targetBandLabel(),
                JpegMath.targetExpandedMidBandNaturalIndices().length);
        System.out.printf(Locale.ROOT, "HIP available     : %s%n", SiUniwardCostEngine.hipAvailable());
        System.out.printf(Locale.ROOT, "HIP device        : %s%n", SiUniwardCostEngine.hipDeviceName());

        double[][] plane = deterministicPlane(width, height, 20260912L);
        int[] qTable = new int[64];
        Arrays.fill(qTable, 17);
        int[] indices = JpegMath.targetExpandedMidBandNaturalIndices();

        long prepareStart = System.nanoTime();
        SiUniwardCostEngine.PreparedPlane prepared =
                SiUniwardCostEngine.prepare(plane, qTable);
        double prepareMs = (System.nanoTime() - prepareStart) / 1_000_000.0;

        // Small warm-up keeps the benchmark from measuring only class loading.
        double[][] warmPlane = deterministicPlane(64, 64, 20260911L);
        SiUniwardCostEngine.PreparedPlane warmPrepared =
                SiUniwardCostEngine.prepare(warmPlane, qTable);
        SiUniwardCostEngine.residualCostsCpu(
                warmPrepared, 8, 8, indices);
        if (SiUniwardCostEngine.hipAvailable()) {
            SiUniwardCostEngine.residualCostsHip(
                    warmPrepared, 8, 8, indices);
        }

        long cpuStart = System.nanoTime();
        double[] cpu = SiUniwardCostEngine.residualCostsCpu(
                prepared, width / 8, height / 8, indices);
        double cpuMs = (System.nanoTime() - cpuStart) / 1_000_000.0;

        System.out.printf(Locale.ROOT, "Preparation       : %.3f ms%n", prepareMs);
        System.out.printf(Locale.ROOT, "CPU batch         : %.3f ms%n", cpuMs);
        System.out.printf(Locale.ROOT, "Candidates        : %,d%n", cpu.length);
        System.out.printf(Locale.ROOT, "CPU ns/candidate  : %.2f%n",
                cpuMs * 1_000_000.0 / cpu.length);

        if (!SiUniwardCostEngine.hipAvailable()) {
            System.out.println("HIP benchmark     : SKIPPED (library/device unavailable)");
            return;
        }

        long hipStart = System.nanoTime();
        double[] hip = SiUniwardCostEngine.residualCostsHip(
                prepared, width / 8, height / 8, indices);
        double hipMs = (System.nanoTime() - hipStart) / 1_000_000.0;

        double maxAbs = 0.0;
        double maxRel = 0.0;
        int worst = -1;
        for (int i = 0; i < cpu.length; i++) {
            double abs = Math.abs(cpu[i] - hip[i]);
            double rel = abs / Math.max(1.0, Math.abs(cpu[i]));
            if (rel > maxRel) {
                maxRel = rel;
                maxAbs = abs;
                worst = i;
            }
            if (abs > 1.0e-9 && rel > 5.0e-11) {
                throw new IllegalStateException(
                        "CPU/HIP parity failure at " + i
                                + ": cpu=" + cpu[i] + ", hip=" + hip[i]
                                + ", abs=" + abs + ", rel=" + rel);
            }
        }

        System.out.printf(Locale.ROOT, "HIP batch         : %.3f ms%n", hipMs);
        System.out.printf(Locale.ROOT, "HIP ns/candidate  : %.2f%n",
                hipMs * 1_000_000.0 / hip.length);
        System.out.printf(Locale.ROOT, "CPU/HIP speedup   : %.3fx%n", cpuMs / hipMs);
        System.out.printf(Locale.ROOT, "Parity max abs    : %.3e%n", maxAbs);
        System.out.printf(Locale.ROOT, "Parity max rel    : %.3e (index %d)%n", maxRel, worst);
        System.out.println("BENCHMARK STATUS   : PASS");
    }

    private static double[][] deterministicPlane(
            int width, int height, long seed) {
        Random random = new Random(seed);
        double[][] plane = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                plane[y][x] = 128.0
                        + 37.0 * Math.sin(0.031 * x)
                        + 29.0 * Math.cos(0.027 * y)
                        + 7.0 * random.nextGaussian();
            }
        }
        return plane;
    }
}
