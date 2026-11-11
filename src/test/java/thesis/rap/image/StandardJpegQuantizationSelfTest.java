package thesis.rap.image;

import java.util.Arrays;

/** Regression checks for exact IJG/libjpeg quality scaling in the supported QF range. */
public final class StandardJpegQuantizationSelfTest {
    private StandardJpegQuantizationSelfTest() {}

    public static void main(String[] args) {
        for (int qf = 75; qf <= 100; qf++) {
            int scale = StandardJpegQuantization.qualityScale(qf);
            int expectedScale = 200 - 2 * qf;
            if (scale != expectedScale) {
                throw new AssertionError("quality scale mismatch at QF " + qf);
            }
            assertRange(StandardJpegQuantization.luminance(qf), qf);
            assertRange(StandardJpegQuantization.chrominance(qf), qf);
        }

        int[] q75LumaHead = Arrays.copyOf(StandardJpegQuantization.luminance(75), 8);
        if (!Arrays.equals(q75LumaHead, new int[]{8, 6, 5, 8, 12, 20, 26, 31})) {
            throw new AssertionError("QF 75 luminance table does not match IJG scaling");
        }
        int[] q75ChromaHead = Arrays.copyOf(StandardJpegQuantization.chrominance(75), 8);
        if (!Arrays.equals(q75ChromaHead, new int[]{9, 9, 12, 24, 50, 50, 50, 50})) {
            throw new AssertionError("QF 75 chrominance table does not match IJG scaling");
        }
        for (int value : StandardJpegQuantization.luminance(100)) {
            if (value != 1) throw new AssertionError("QF 100 luminance table must be all ones");
        }
        for (int value : StandardJpegQuantization.chrominance(100)) {
            if (value != 1) throw new AssertionError("QF 100 chrominance table must be all ones");
        }

        System.out.println("StandardJpegQuantizationSelfTest: ALL TESTS PASSED");
    }

    private static void assertRange(int[] table, int qf) {
        if (table.length != 64) throw new AssertionError("quantization table length");
        for (int value : table) {
            if (value < 1 || value > 255) {
                throw new AssertionError("QF " + qf + " quantization value outside baseline range: " + value);
            }
        }
    }
}
