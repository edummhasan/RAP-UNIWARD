package thesis.rap.image;

import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import java.util.Arrays;

public final class CanonicalQuantizationSelfTest {
    private CanonicalQuantizationSelfTest() {}

    public static void main(String[] args) {
        assert CanonicalReferenceJpeg.quantize(1.49) == 1;
        assert CanonicalReferenceJpeg.quantize(1.5) == 2;
        assert CanonicalReferenceJpeg.quantize(-1.49) == -1;
        assert CanonicalReferenceJpeg.quantize(-1.5) == -2
                : "negative exact half must round away from zero";
        assert CanonicalReferenceJpeg.quantize(0.5) == 1;
        assert CanonicalReferenceJpeg.quantize(-0.5) == -1;
        testWriteBackVerification();
        System.out.println("CanonicalQuantizationSelfTest: ALL TESTS PASSED");
    }

    private static void testWriteBackVerification() {
        int[] coefficients = new int[64];
        int[] quantization = new int[64];
        Arrays.fill(quantization, 7);
        JpegImage intended = new JpegImage(
                8, 8,
                new JpegComponent[]{
                        new JpegComponent(
                                0, 1, 1, 1, 1, 0,
                                coefficients.clone(),
                                quantization.clone())
                });
        JpegImage identical = new JpegImage(
                8, 8,
                new JpegComponent[]{
                        new JpegComponent(
                                0, 1, 1, 1, 1, 0,
                                coefficients.clone(),
                                quantization.clone())
                });
        CanonicalReferenceJpeg.verifyWrittenReference(
                intended, identical);

        identical.component(0).setCoeff(0, 9, 1);
        boolean rejected = false;
        try {
            CanonicalReferenceJpeg.verifyWrittenReference(
                    intended, identical);
        } catch (IllegalStateException expected) {
            rejected = true;
        }
        assert rejected
                : "canonical coefficient write-back mismatch must fail";
    }
}
