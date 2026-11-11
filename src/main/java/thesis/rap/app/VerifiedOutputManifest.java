package thesis.rap.app;

import thesis.rap.analysis.FileFingerprint;
import thesis.rap.image.ImageUtil;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;
import thesis.rap.stc.OriginalStcCodec;
import thesis.rap.util.AtomicFiles;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Versioned hash record required before an existing stego output may be resumed. */
final class VerifiedOutputManifest {
    private static final String FORMAT = "RAP-VERIFIED-OUTPUT-1";

    private VerifiedOutputManifest() {}

    static Path sidecarFor(Path stego) {
        java.util.Objects.requireNonNull(stego, "stegoOutputPath");
        return stego.resolveSibling(stego.getFileName() + ".verified");
    }

    static void write(Path stego, Path png, Path message, int minQf, int maxQf, String applicationVersion)
            throws Exception {
        write(stego, png, message, minQf, maxQf,
                applicationVersion, StcCarrierPlan.DEFAULT_STC_HEIGHT,
                SamplingGeometryMode.GEOMETRY_420,
                SamplingGeometry.JPEG_420);
    }

    static void write(
            Path stego, Path png, Path message,
            int minQf, int maxQf, String applicationVersion,
            int stcHeight
    ) throws Exception {
        write(stego, png, message, minQf, maxQf,
                applicationVersion, stcHeight,
                SamplingGeometryMode.GEOMETRY_420,
                SamplingGeometry.JPEG_420);
    }

    static void write(
            Path stego, Path png, Path message,
            int minQf, int maxQf, String applicationVersion,
            int stcHeight,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry
    ) throws Exception {
        Path sidecar = sidecarFor(stego);
        Map<String, String> values = expected(
                stego, png, message, minQf, maxQf,
                applicationVersion, stcHeight,
                geometryMode, geometry);
        StringBuilder text = new StringBuilder();
        for (Map.Entry<String, String> entry : values.entrySet()) {
            text.append(entry.getKey()).append('=').append(entry.getValue()).append('\n');
        }
        AtomicFiles.writeString(
                sidecar, text.toString(), StandardCharsets.UTF_8);
    }

    static boolean matches(Path stego, Path png, Path message, int minQf, int maxQf, String applicationVersion)
            throws Exception {
        return matches(stego, png, message, minQf, maxQf,
                applicationVersion, StcCarrierPlan.DEFAULT_STC_HEIGHT,
                SamplingGeometryMode.GEOMETRY_420,
                SamplingGeometry.JPEG_420);
    }

    static boolean matches(
            Path stego, Path png, Path message,
            int minQf, int maxQf, String applicationVersion,
            int stcHeight
    ) throws Exception {
        return matches(stego, png, message, minQf, maxQf,
                applicationVersion, stcHeight,
                SamplingGeometryMode.GEOMETRY_420,
                SamplingGeometry.JPEG_420);
    }

    static boolean matches(
            Path stego, Path png, Path message,
            int minQf, int maxQf, String applicationVersion,
            int stcHeight,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry
    ) throws Exception {
        Path sidecar = sidecarFor(stego);
        if (!Files.isRegularFile(stego) || Files.size(stego) == 0 || !Files.isRegularFile(sidecar)) return false;
        Map<String, String> observed = parse(Files.readAllLines(sidecar, StandardCharsets.UTF_8));
        return observed.equals(expected(
                stego, png, message, minQf, maxQf,
                applicationVersion, stcHeight,
                geometryMode, geometry));
    }

    static void delete(Path stego) throws Exception {
        Files.deleteIfExists(sidecarFor(stego));
    }

    private static Map<String, String> expected(
            Path stego, Path png, Path message, int minQf, int maxQf,
            String applicationVersion, int stcHeight,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry) throws Exception {
        StcCarrierPlan.validateSupportedHeight(stcHeight);
        if (geometryMode == null || geometry == null
                || !geometryMode.geometries().contains(geometry)) {
            throw new IllegalArgumentException(
                    "Manifest geometry must belong to its selected mode");
        }
        Map<String, String> values = new LinkedHashMap<>();
        values.put("format", FORMAT);
        values.put("application_version", applicationVersion);
        values.put("qf_min", String.valueOf(minQf));
        values.put("qf_max", String.valueOf(maxQf));
        values.put("stc_height", String.valueOf(stcHeight));
        values.put("stc_numeric_profile", OriginalStcCodec.NUMERIC_PROFILE);
        ThreeStateEmbeddingEngine.Profile balanceProfile =
                ThreeStateEmbeddingEngine.Profile.configured();
        values.put("balance_profile", balanceProfile.code());
        values.put("balance_zero_budget", balanceProfile ==
                ThreeStateEmbeddingEngine.Profile.UNRESTRICTED
                ? "UNBOUNDED" : String.valueOf(balanceProfile.zeroBudget()));
        values.put("balance_max_attempts", String.valueOf(balanceProfile.maxAttempts()));
        values.put("geometry_mode", geometryMode.token());
        values.put("sampling_geometry", geometry.label());
        values.put("geometry_signature_hex", geometry.signatureHex());
        values.put("carrier_order_domain", "ahp-order-v1");
        values.put("sampling_factors",
                "Y=" + geometry.horizontalSampling(0) + "x"
                        + geometry.verticalSampling(0)
                        + ";Cb=" + geometry.horizontalSampling(1) + "x"
                        + geometry.verticalSampling(1)
                        + ";Cr=" + geometry.horizontalSampling(2) + "x"
                        + geometry.verticalSampling(2));
        values.put("reference_template_writer",
                ImageUtil.jpegWriterDescription());
        values.put("png_sha256", FileFingerprint.sha256(png));
        values.put("message_sha256", FileFingerprint.sha256(message));
        values.put("stego_sha256", FileFingerprint.sha256(stego));
        return values;
    }

    private static Map<String, String> parse(List<String> lines) {
        Map<String, String> values = new LinkedHashMap<>();
        for (String line : lines) {
            if (line.isBlank()) continue;
            int equals = line.indexOf('=');
            if (equals <= 0 || equals == line.length() - 1) return Map.of();
            String previous = values.put(line.substring(0, equals), line.substring(equals + 1));
            if (previous != null) return Map.of();
        }
        return values;
    }
}
