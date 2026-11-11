package thesis.rap.analysis;

import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;
import thesis.rap.app.StcCarrierPlan;
import thesis.rap.stc.OriginalStcCodec;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

/** Logs every attempted image-message pair, including capacity skips and failures. */
public final class AttemptCsvWriter {
    private AttemptCsvWriter() {}

    public static void append(
            Path csv,
            String runId,
            String image,
            String message,
            int qfMin,
            int qfMax,
            int requiredBits,
            Integer selectedQf,
            String status,
            String reason
    ) throws IOException {
        append(csv, runId, image, message,
                SamplingGeometryMode.GEOMETRY_420,
                SamplingGeometry.JPEG_420,
                StcCarrierPlan.DEFAULT_STC_HEIGHT,
                qfMin, qfMax, requiredBits, selectedQf,
                status, reason);
    }

    public static void append(
            Path csv,
            String runId,
            String image,
            String message,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry,
            int stcHeight,
            int qfMin,
            int qfMax,
            int requiredBits,
            Integer selectedQf,
            String status,
            String reason
    ) throws IOException {
        StcCarrierPlan.validateSupportedHeight(stcHeight);
        MessageGroup group = MessageGroup.from(message);
        CsvUtil.append(csv,
                List.of("schema_version","timestamp_utc","run_id","message_group","message_number",
                        "message_name","image","message","geometry_mode",
                        "sampling_geometry","stc_height","stc_numeric_profile",
                        "qf_min","qf_max",
                        "required_bits","selected_qf","status","reason"),
                List.of(EvaluationCsvWriter.SCHEMA_VERSION, Instant.now().toString(), runId,
                        group.label(), String.valueOf(group.number()), group.name(), image, message,
                        geometryMode.token(), geometry.label(),
                        String.valueOf(stcHeight), OriginalStcCodec.NUMERIC_PROFILE,
                        String.valueOf(qfMin), String.valueOf(qfMax), String.valueOf(requiredBits),
                        selectedQf == null ? "" : String.valueOf(selectedQf), status, reason == null ? "" : reason));
    }
}
