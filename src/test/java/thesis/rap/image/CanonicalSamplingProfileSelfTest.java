package thesis.rap.image;

import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import java.util.Arrays;

/** Regression coverage for the 4:2:0 compatibility guard. */
public final class CanonicalSamplingProfileSelfTest {
    private CanonicalSamplingProfileSelfTest() {}

    public static void main(String[] args) throws Exception {
        CanonicalReferenceJpeg.requireEvaluatedFourTwoZero(
                image(new int[][]{{2, 2}, {1, 1}, {1, 1}}));
        assertRejected(image(new int[][]{{1, 1}, {1, 1}, {1, 1}}),
                "4:4:4 must be rejected");
        assertRejected(image(new int[][]{{2, 1}, {1, 1}, {1, 1}}),
                "4:2:2 must be rejected");
        assertRejected(image(new int[][]{{2, 2}, {2, 1}, {1, 1}}),
                "asymmetric chroma sampling must be rejected");
        String writer = ImageUtil.jpegWriterDescription();
        assert writer.contains("class=") && writer.contains("version=")
                : "template-writer provenance must include class and version";
        System.out.println(
                "CanonicalSamplingProfileSelfTest: ALL TESTS PASSED");
    }

    private static void assertRejected(JpegImage image, String message) {
        boolean rejected = false;
        try {
            CanonicalReferenceJpeg.requireEvaluatedFourTwoZero(image);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected : message;
    }

    private static JpegImage image(int[][] sampling) {
        JpegComponent[] components = new JpegComponent[3];
        int[] quant = new int[64];
        Arrays.fill(quant, 1);
        for (int i = 0; i < components.length; i++) {
            components[i] = new JpegComponent(
                    i, i == 0 ? 2 : 1, i == 0 ? 2 : 1,
                    sampling[i][0], sampling[i][1], i == 0 ? 0 : 1,
                    new int[(i == 0 ? 4 : 1) * 64],
                    Arrays.copyOf(quant, quant.length));
        }
        return new JpegImage(16, 16, components);
    }
}
