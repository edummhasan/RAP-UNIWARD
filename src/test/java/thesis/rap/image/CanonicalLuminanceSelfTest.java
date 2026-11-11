package thesis.rap.image;

import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import java.util.Arrays;

/** Regression tests for decoder-independent canonical Y reconstruction. */
public final class CanonicalLuminanceSelfTest {
    private CanonicalLuminanceSelfTest() {}

    public static void main(String[] args) {
        testIdealInverseDctDcReconstruction();
        testReconstructionClipsToSampleRange();
        testFullResolutionYIsRequired();
        System.out.println(
                "CanonicalLuminanceSelfTest: ALL TESTS PASSED");
    }

    private static void testIdealInverseDctDcReconstruction() {
        int[] yCoefficients = new int[64];
        yCoefficients[0] = 8;
        double[][] luminance = ImageUtil.canonicalReferenceLuminance(
                threeComponentImage(
                        8, 8, 2, 2, yCoefficients));
        assert luminance.length == 8;
        assert luminance[0].length == 8;
        for (double[] row : luminance) {
            for (double value : row) {
                assert Math.abs(value - 129.0) < 1.0e-12
                        : "DC-only ideal IDCT must produce a uniform block";
            }
        }
        double[] histogram = ImageUtil.histogram(luminance);
        assert Math.abs(histogram[129] - 1.0) < 1.0e-12;
    }

    private static void testReconstructionClipsToSampleRange() {
        int[] yCoefficients = new int[64];
        yCoefficients[0] = 1024;
        double[][] luminance = ImageUtil.canonicalReferenceLuminance(
                threeComponentImage(
                        5, 3, 1, 1, yCoefficients));
        assert luminance.length == 3;
        assert luminance[0].length == 5;
        for (double[] row : luminance) {
            for (double value : row) {
                assert value == 255.0;
            }
        }
    }

    private static void testFullResolutionYIsRequired() {
        int[] quant = unitQuantTable();
        JpegImage invalid = new JpegImage(
                8, 8,
                new JpegComponent[]{
                        new JpegComponent(
                                0, 1, 1, 1, 1, 0,
                                new int[64], quant),
                        new JpegComponent(
                                1, 1, 1, 2, 1, 1,
                                new int[64], quant),
                        new JpegComponent(
                                2, 1, 1, 1, 1, 1,
                                new int[64], quant)
                });
        boolean rejected = false;
        try {
            ImageUtil.canonicalReferenceLuminance(invalid);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected
                : "component 0 must be the full-resolution Y component";
    }

    private static JpegImage threeComponentImage(
            int width,
            int height,
            int yHorizontalSampling,
            int yVerticalSampling,
            int[] yCoefficients
    ) {
        int[] quant = unitQuantTable();
        return new JpegImage(
                width, height,
                new JpegComponent[]{
                        new JpegComponent(
                                0, 1, 1,
                                yHorizontalSampling,
                                yVerticalSampling,
                                0, yCoefficients, quant),
                        new JpegComponent(
                                1, 1, 1, 1, 1, 1,
                                new int[64], quant),
                        new JpegComponent(
                                2, 1, 1, 1, 1, 1,
                                new int[64], quant)
                });
    }

    private static int[] unitQuantTable() {
        int[] quant = new int[64];
        Arrays.fill(quant, 1);
        return quant;
    }
}
