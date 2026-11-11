package thesis.rap.analysis;

/** Reproducibility metadata derived from the actual libjpeg coefficient structure. */
public record JpegMetadata(
        int width,
        int height,
        int componentCount,
        String subsampling,
        int maxHSamplingFactor,
        int maxVSamplingFactor,
        int mcuColumns,
        int mcuRows,
        long mcuCount,
        int yBlockCount,
        int cbBlockCount,
        int crBlockCount,
        String samplingFactors,
        String quantTableIndexes,
        String quantizationTableSha256
) {}
