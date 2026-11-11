package thesis.rap.analysis;

/** Immutable identifiers and runtime details needed to reproduce or audit one run. */
public record RunReproducibility(
        String timestampUtc,
        String applicationVersion,
        String inputImageSha256,
        String messageSha256,
        String referenceJpegSha256,
        String stegoJpegSha256,
        String javaVersion,
        String osName,
        String osArch,
        int libjpegVersion,
        String referenceEncoder
) {}
