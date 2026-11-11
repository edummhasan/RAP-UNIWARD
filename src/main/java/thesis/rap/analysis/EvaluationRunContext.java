package thesis.rap.analysis;

import thesis.rap.payload.PayloadPackage;

/** Metadata that links run-level, attempt-level, per-channel, and per-position statistics. */
public record EvaluationRunContext(
        String runId,
        String imageName,
        String messageName,
        String geometryMode,
        String samplingGeometry,
        int qfMin,
        int qfMax,
        String selectionPolicy,
        double maxCandidateLoading,
        double maxStcUtilization,
        PayloadPackage.PayloadLayout payloadLayout,
        JpegMetadata jpegMetadata,
        QfSelectionMetrics qfSelectionMetrics,
        RunReproducibility reproducibility,
        long embeddingElapsedMillis,
        long evaluationElapsedMillis
) {}
