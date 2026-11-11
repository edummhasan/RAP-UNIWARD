package thesis.rap.image;

import thesis.rap.alg.JpegMath;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegCore;
import thesis.rap.jni.JpegImage;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/** Builds the reference JPEG from the same exact pre-quantization DCT used for side information. */
public final class CanonicalReferenceJpeg {
    private CanonicalReferenceJpeg() {}

    public static JpegImage build(Path png, Path output, int qf) throws Exception {
        return build(
                png, output, qf, SamplingGeometry.JPEG_420);
    }

    public static JpegImage build(
            Path png,
            Path output,
            int qf,
            SamplingGeometry geometry
    ) throws Exception {
        return build(SourceImageContext.load(png), output, qf, geometry);
    }

    /**
     * Builds one canonical QF reference from a run-scoped source context.
     * PNG decoding and RGB-to-YCbCr conversion therefore occur once per input
     * image rather than once per QF.
     */
    public static JpegImage build(
            SourceImageContext source,
            Path output,
            int qf,
            SamplingGeometry geometry
    ) throws Exception {
        if (source == null) {
            throw new IllegalArgumentException("source cannot be null");
        }
        if (geometry == null) {
            throw new IllegalArgumentException("geometry cannot be null");
        }
        Path template = Files.createTempFile(
                "rap-jpeg-template-" + geometry.outputTag()
                        + "-qf" + qf + "-",
                ".jpg");
        try {
            ImageUtil.writeJpeg(
                    source.sourceImage(), template, qf, geometry);
            JpegImage jpeg = JpegCore.readCoefficients(template.toString());
            geometry.require(jpeg);
            int[] standardLuma = StandardJpegQuantization.luminance(qf);
            int[] standardChroma = StandardJpegQuantization.chrominance(qf);
            for (int componentIndex = 0; componentIndex < jpeg.components.length; componentIndex++) {
                JpegComponent component = jpeg.component(componentIndex);
                int[] canonicalQuantTable = componentIndex == 0 ? standardLuma : standardChroma;
                System.arraycopy(canonicalQuantTable, 0, component.quantTable, 0, 64);
                double[] rawDct = source.rawDctCoefficients(
                        jpeg, componentIndex);
                for (int by = 0; by < component.heightBlocks; by++) {
                    for (int bx = 0; bx < component.widthBlocks; bx++) {
                        int block = by * component.widthBlocks + bx;
                        int rawBase = block * 64;
                        for (int k = 0; k < 64; k++) {
                            int quantStep = Math.max(1, component.quantTable[k]);
                            int quantized = quantize(
                                    rawDct[rawBase + k] / quantStep);
                            if (quantized < Short.MIN_VALUE
                                    || quantized > Short.MAX_VALUE) {
                                throw new IllegalStateException(
                                        "Canonical coefficient is outside the signed JCOEF range");
                            }
                            // Baseline entropy coding caps AC magnitudes at 1023
                            // (libjpeg MAX_COEF_BITS = 10).  Detect it here with a
                            // precise message so the QF trial is rejected cleanly
                            // instead of aborting inside libjpeg.
                            if (k != 0
                                    && Math.abs(quantized)
                                            > JpegMath.BASELINE_MAX_AC_MAGNITUDE) {
                                throw new UnencodableReferenceException(
                                        qf,
                                        geometry,
                                        "Canonical AC coefficient magnitude "
                                                + Math.abs(quantized)
                                                + " exceeds the baseline JPEG limit of "
                                                + JpegMath.BASELINE_MAX_AC_MAGNITUDE
                                                + " at geometry "
                                                + geometry.label()
                                                + ", QF " + qf + ", component "
                                                + componentIndex + ", natural index " + k);
                            }
                            component.setCoeff(block, k, quantized);
                        }
                    }
                }
            }
            JpegCore.writeCoefficients(template.toString(), output.toString(), jpeg);
            JpegImage written = JpegCore.readCoefficients(output.toString());
            geometry.require(written);
            verifyWrittenReference(jpeg, written);
            return written;
        } finally {
            Files.deleteIfExists(template);
        }
    }

    static int quantize(double coefficientInBinUnits) {
        if (!Double.isFinite(coefficientInBinUnits)) throw new IllegalArgumentException("Non-finite DCT coefficient");
        return JpegMath.roundHalfAwayFromZero(coefficientInBinUnits);
    }

    /**
     * Compatibility helper that enforces only the 4:2:0 population.  The
     * current public multi-geometry path calls {@link #build(Path, Path, int,
     * SamplingGeometry)} and validates whichever declared color-subsampling
     * population was selected.
     */
    public static void requireEvaluatedFourTwoZero(JpegImage jpeg) {
        SamplingGeometry.JPEG_420.require(jpeg);
    }

    /** Enforces membership in exactly one declared color-subsampling population. */
    public static SamplingGeometry requireSupportedGeometry(
            JpegImage jpeg
    ) {
        return SamplingGeometry.detect(jpeg);
    }

    static void verifyWrittenReference(
            JpegImage intended, JpegImage written
    ) {
        if (intended.width != written.width
                || intended.height != written.height
                || intended.components.length != written.components.length) {
            throw new IllegalStateException(
                    "Canonical JPEG layout changed during coefficient serialization");
        }
        for (int i = 0; i < intended.components.length; i++) {
            JpegComponent a = intended.component(i);
            JpegComponent b = written.component(i);
            if (a.componentIndex != b.componentIndex
                    || a.widthBlocks != b.widthBlocks
                    || a.heightBlocks != b.heightBlocks
                    || a.hSamplingFactor != b.hSamplingFactor
                    || a.vSamplingFactor != b.vSamplingFactor
                    || a.quantTableIndex != b.quantTableIndex
                    || !Arrays.equals(a.quantTable, b.quantTable)
                    || !Arrays.equals(a.coefficients, b.coefficients)) {
                throw new IllegalStateException(
                        "Canonical JPEG component " + i
                                + " failed layout/table/coefficient write-back verification");
            }
        }
    }
}
