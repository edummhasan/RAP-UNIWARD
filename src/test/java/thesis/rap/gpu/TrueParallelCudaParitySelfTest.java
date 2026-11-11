package thesis.rap.gpu;

import thesis.rap.alg.JpegMath;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.stc.CudaStcCodec;
import thesis.rap.stc.OriginalStcCodec;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;


public final class TrueParallelCudaParitySelfTest {
    private TrueParallelCudaParitySelfTest() {}

    public static void main(String[] args) throws Exception {
        if (!CudaTrueParallelNative.isAvailable()
                || !CudaDctNative.isAvailable()
                || !CudaStcCodec.isAvailable()) {
            System.out.println("TrueParallelCudaParitySelfTest: SKIPPED (CUDA unavailable)");
            System.out.println("HMAC: " + CudaTrueParallelNative.diagnostic());
            System.out.println("DCT: " + CudaDctNative.diagnostic());
            System.out.println("STC: " + CudaStcNative.diagnostic());

            if (Boolean.parseBoolean(System.getProperty("rap.require.cuda",
                    System.getenv().getOrDefault("RAP_REQUIRE_CUDA", "false")))) {
                System.out.println("TrueParallelCudaParitySelfTest: FAIL (RAP_REQUIRE_CUDA=true)");
                System.exit(3);
            }
            return;
        }
        testHmac();
        testP95();
        testDctAndVariance();
        testStc(10);
        testStc(12);
        System.out.println("TrueParallelCudaParitySelfTest: ALL CUDA PARITY TESTS PASSED");
    }

    private static void testHmac() throws Exception {
        byte[] key = new byte[32];
        for (int i = 0; i < key.length; i++) key[i] = (byte) (3 * i + 7);
        byte[] domain = "ahp-order-v1".getBytes(StandardCharsets.UTF_8);
        int qf = 83;
        int geometry = SamplingGeometry.JPEG_420.signatureCode();
        int[] fields = {
                1, 17, 2, 3, 4, 5, 44, -7,
                2, 91, 8, 9, 1, 6, 29, 13,
                1, 1234, 11, 12, 7, 0, 56, -1023
        };
        byte[] actual = CudaTrueParallelNative.hmacCandidateDigests(
                key, domain, qf, geometry, fields);
        byte[] expected = new byte[(fields.length / 8) * 32];
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(key, "HmacSHA256"));
        for (int candidate = 0; candidate < fields.length / 8; candidate++) {
            ByteBuffer serialized = ByteBuffer.allocate(40);
            serialized.putInt(qf).putInt(geometry);
            int base = candidate * 8;
            for (int i = 0; i < 8; i++) serialized.putInt(fields[base + i]);
            mac.update(domain);
            mac.update(serialized.array());
            byte[] digest = mac.doFinal();
            System.arraycopy(digest, 0, expected, candidate * 32, 32);
        }
        require(Arrays.equals(expected, actual), "CUDA HMAC differs from JCE HMAC");
    }

    private static void testP95() {
        double[] values = new double[10001];
        for (int i = 0; i < values.length; i++) {
            values[i] = ((i * 7919L) % 10007L) / 17.0;
        }
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int index = Math.max(0, Math.min(sorted.length - 1,
                (int) Math.ceil(0.95 * sorted.length) - 1));
        double expected = sorted[index];
        double actual = CudaTrueParallelNative.percentile95(values);
        require(Double.doubleToLongBits(expected) == Double.doubleToLongBits(actual),
                "CUDA P95 differs from exact Java nearest-rank P95");
    }

    private static void testDctAndVariance() {
        double[][] plane = new double[16][16];
        for (int y = 0; y < plane.length; y++) {
            for (int x = 0; x < plane[y].length; x++) {
                plane[y][x] = 16.0 + ((x * 29 + y * 47 + x * y * 3) % 221);
            }
        }
        CudaDctNative.Result actual = CudaDctNative.compute(plane, 2, 2);
        double[] dct = actual.dctView();
        double[] variance = actual.variancesView();
        int[] natural = new int[64];
        for (int i = 0; i < 64; i++) natural[i] = i;
        for (int by = 0; by < 2; by++) {
            for (int bx = 0; bx < 2; bx++) {
                int block = by * 2 + bx;
                double[] expected = JpegMath.forwardDctCoefficients(
                        plane, bx * 8, by * 8, natural);
                for (int k = 0; k < 64; k++) {
                    require(Double.doubleToLongBits(expected[k])
                                    == Double.doubleToLongBits(dct[block * 64 + k]),
                            "CUDA DCT binary64 parity failure at block=" + block + " k=" + k);
                }
                double mean = 0.0;
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) mean += plane[by * 8 + y][bx * 8 + x];
                }
                mean /= 64.0;
                double expectedVariance = 0.0;
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        double centered = plane[by * 8 + y][bx * 8 + x] - mean;
                        expectedVariance += centered * centered;
                    }
                }
                expectedVariance /= 64.0;
                require(Double.doubleToLongBits(expectedVariance)
                                == Double.doubleToLongBits(variance[block]),
                        "CUDA variance binary64 parity failure at block=" + block);
            }
        }
    }

    private static void testStc(int height) {
        int messageLength = height == 10 ? 128 : 144;
        int coverLength = messageLength * 8 + 13;
        byte[] cover = new byte[coverLength];
        double[] costs = new double[coverLength];
        byte[] message = new byte[messageLength];
        for (int i = 0; i < coverLength; i++) {
            cover[i] = (byte) ((i * 5 + 3) & 1);
            costs[i] = 0.125 + ((i * 104729L) % 65521L) / 257.0;
        }
        for (int i = 0; i < messageLength; i++) {
            message[i] = (byte) ((i * 7 + 1) & 1);
        }

        OriginalStcCodec cpu = new OriginalStcCodec();
        CudaStcCodec gpu = new CudaStcCodec();
        byte[] cpuStego = cpu.embed(cover, costs, message, height);
        byte[] gpuStego = gpu.embed(cover, costs, message, height);
        require(Arrays.equals(cpuStego, gpuStego),
                "CUDA STC stego differs from CPU STC at h=" + height);
        byte[] cpuMessage = cpu.extract(gpuStego, messageLength, height);
        byte[] gpuMessage = gpu.extract(gpuStego, messageLength, height);
        require(Arrays.equals(message, cpuMessage),
                "CPU extraction did not recover test message at h=" + height);
        require(Arrays.equals(message, gpuMessage),
                "CUDA extraction did not recover test message at h=" + height);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
