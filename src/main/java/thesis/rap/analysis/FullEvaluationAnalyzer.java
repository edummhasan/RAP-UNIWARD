package thesis.rap.analysis;

import thesis.rap.alg.Candidate;
import thesis.rap.jni.JpegCore;
import thesis.rap.jni.JpegImage;

import java.util.List;

/** Builds the complete visual + JPEG-domain evaluation object for one real embedding run. */
public final class FullEvaluationAnalyzer {
    private FullEvaluationAnalyzer() {}

    public static EvaluationSummary evaluate(
            String originalPngPath,
            String referenceJpegPath,
            String stegoJpegPath,
            int payloadBits,
            int selectedQF,
            List<Candidate> candidates,
            List<Candidate> activeCarriers,
            int engineeringCapacity,
            StcRunMetrics stcMetrics,
            boolean embeddingSuccess,
            boolean extractionSuccess,
            boolean aesGcmVerified,
            boolean recoveredMessageMatches
    ) throws Exception {
        EvaluationSummary summary = new EvaluationSummary();
        summary.pngVsStego = ImageQualityAnalyzer.compare(
                "Original PNG vs Stego JPEG (format-changing informational comparison)", originalPngPath, stegoJpegPath);
        summary.referenceVsStego = ImageQualityAnalyzer.compare(
                "Reference JPEG vs Stego JPEG (controlled embedding comparison)", referenceJpegPath, stegoJpegPath);

        JpegImage refJpeg = JpegCore.readCoefficients(referenceJpegPath);
        JpegImage stegoJpeg = JpegCore.readCoefficients(stegoJpegPath);
        summary.jpegMetadata = JpegMetadataAnalyzer.analyze(refJpeg);
        summary.coefficientReport = JpegCoefficientAnalyzer.compare(
                refJpeg,
                stegoJpeg,
                payloadBits,
                selectedQF,
                candidates,
                activeCarriers,
                engineeringCapacity,
                stcMetrics,
                embeddingSuccess,
                extractionSuccess,
                aesGcmVerified,
                recoveredMessageMatches
        );
        return summary;
    }
}
