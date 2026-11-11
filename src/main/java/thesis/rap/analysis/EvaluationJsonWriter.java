package thesis.rap.analysis;

import thesis.rap.payload.PayloadPackage;
import thesis.rap.stc.OriginalStcCodec;
import thesis.rap.util.AtomicFiles;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** Dependency-free structured JSON export for one completed real embedding run. */
public final class EvaluationJsonWriter {
    private EvaluationJsonWriter() {}

    public static void write(Path path, EvaluationRunContext ctx, EvaluationSummary s) throws IOException {
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        StringBuilder b = new StringBuilder(64 * 1024);
        JpegCoefficientReport c = s.coefficientReport;
        PayloadPackage.PayloadLayout p = ctx.payloadLayout();
        JpegMetadata j = ctx.jpegMetadata();
        QfSelectionMetrics q = ctx.qfSelectionMetrics();
        RunReproducibility r = ctx.reproducibility();

        b.append("{\n");
        field(b, 1, "schema_version", EvaluationCsvWriter.SCHEMA_VERSION, true);
        field(b, 1, "run_id", ctx.runId(), true);
        field(b, 1, "image", ctx.imageName(), true);
        field(b, 1, "message", ctx.messageName(), true);
        field(b, 1, "geometry_mode", ctx.geometryMode(), true);
        field(b, 1, "sampling_geometry", ctx.samplingGeometry(), true);

        b.append(indent(1)).append("\"reproducibility\": {\n");
        field(b, 2, "timestamp_utc", r.timestampUtc(), true);
        field(b, 2, "application_version", r.applicationVersion(), true);
        field(b, 2, "input_image_sha256", r.inputImageSha256(), true);
        field(b, 2, "message_sha256", r.messageSha256(), true);
        field(b, 2, "reference_jpeg_sha256", r.referenceJpegSha256(), true);
        field(b, 2, "stego_jpeg_sha256", r.stegoJpegSha256(), true);
        field(b, 2, "java_version", r.javaVersion(), true);
        field(b, 2, "os_name", r.osName(), true);
        field(b, 2, "os_arch", r.osArch(), true);
        numberField(b, 2, "libjpeg_version", r.libjpegVersion(), true);
        field(b, 2, "reference_encoder", r.referenceEncoder(), false);
        b.append(indent(1)).append("},\n");

        b.append(indent(1)).append("\"qf_selection\": {\n");
        numberField(b, 2, "qf_min", ctx.qfMin(), true);
        numberField(b, 2, "qf_max", ctx.qfMax(), true);
        field(b, 2, "selection_policy", ctx.selectionPolicy(), true);
        numberField(b, 2, "selected_qf", c.selectedQF, true);
        numberField(b, 2, "qfs_tested", q.qfsTested(), true);
        numberField(b, 2, "failed_before_selection", q.failedQfsBeforeSelection(), true);
        nullableNumberField(b, 2, "previous_tested_qf", q.previousTestedQf(), true);
        numberField(b, 2, "candidate_headroom_bits", q.candidateHeadroomBits(), true);
        numberField(b, 2, "capacity_headroom_bits", q.capacityHeadroomBits(), false);
        b.append(indent(1)).append("},\n");

        b.append(indent(1)).append("\"jpeg_metadata\": {\n");
        numberField(b, 2, "width", j.width(), true); numberField(b, 2, "height", j.height(), true);
        numberField(b, 2, "component_count", j.componentCount(), true); field(b, 2, "subsampling", j.subsampling(), true);
        field(b, 2, "sampling_factors", j.samplingFactors(), true); numberField(b, 2, "mcu_count", j.mcuCount(), true);
        numberField(b, 2, "y_blocks", j.yBlockCount(), true); numberField(b, 2, "cb_blocks", j.cbBlockCount(), true);
        numberField(b, 2, "cr_blocks", j.crBlockCount(), true); field(b, 2, "quant_table_indexes", j.quantTableIndexes(), true);
        field(b, 2, "quant_tables_sha256", j.quantizationTableSha256(), false);
        b.append(indent(1)).append("},\n");

        b.append(indent(1)).append("\"payload\": {\n");
        numberField(b, 2, "raw_message_bytes", p.rawMessageBytes(), true);
        numberField(b, 2, "huffman_candidate_bytes", p.huffmanCandidateBytes(), true);
        booleanField(b, 2, "huffman_used", p.huffmanUsed(), true);
        numberField(b, 2, "content_bytes", p.contentBytes(), true);
        numberField(b, 2, "preprocessing_metadata_bytes", p.preprocessingMetadataBytes(), true);
        numberField(b, 2, "prepared_plaintext_bytes", p.preparedPlaintextBytes(), true);
        numberField(b, 2, "protocol_marker_bytes", p.protocolMarkerBytes(), true);
        numberField(b, 2, "qf_field_bytes", p.qfFieldBytes(), true);
        numberField(b, 2, "ciphertext_length_field_bytes", p.ciphertextLengthFieldBytes(), true);
        numberField(b, 2, "nonce_bytes", p.nonceBytes(), true); numberField(b, 2, "authentication_tag_bytes", p.authenticationTagBytes(), true);
        numberField(b, 2, "aad_bytes", PayloadPackage.GEOMETRY_AAD_LEN, true);
        numberField(b, 2, "fixed_header_bytes", p.fixedHeaderBytes(), true); numberField(b, 2, "ciphertext_bytes", p.ciphertextBytes(), true);
        numberField(b, 2, "total_payload_bytes", p.totalPayloadBytes(), true); numberField(b, 2, "total_payload_bits", p.totalPayloadBits(), true);
        numberField(b, 2, "fixed_framing_overhead_bytes", p.fixedFramingOverheadBytes(), true);
        numberField(b, 2, "net_payload_expansion_bytes", p.netPayloadExpansionBytes(), true);
        doubleField(b, 2, "expansion_ratio", p.expansionRatio(), false);
        b.append(indent(1)).append("},\n");

        b.append(indent(1)).append("\"image_quality\": {\n");
        comparison(b, 2, "reference_vs_stego", s.referenceVsStego, true);
        comparison(b, 2, "png_vs_stego_informational", s.pngVsStego, false);
        b.append(indent(1)).append("},\n");

        b.append(indent(1)).append("\"stc_and_coefficients\": {\n");
        numberField(
                b, 2, "stc_constraint_height",
                c.stcConstraintHeight, true);
        field(
                b, 2, "stc_numeric_profile",
                OriginalStcCodec.NUMERIC_PROFILE, true);
        numberField(b, 2, "payload_bits", c.payloadBits, true); numberField(b, 2, "eligible_candidates", c.candidateCount, true);
        numberField(
                b, 2, "former_four_mode_near_half_candidate_count",
                c.formerFourModeNearHalfCandidateCount, true);
        numberField(b, 2, "fully_wet_candidate_count", c.fullyWetCandidateCount, true);
        numberField(b, 2, "actual_stc_cover_elements", c.activeCarrierCount, true); numberField(b, 2, "fixed_header_reserve", c.stcReserveBits, true);
        numberField(b, 2, "engineering_message_capacity", c.conservativeCapacity, true); doubleField(b, 2, "true_stc_relative_payload", c.activeCarrierPayloadRate, true);
        numberField(b, 2, "stc_distinct_column_width_limit", c.stcDistinctColumnWidthLimit, true);
        numberField(b, 2, "stc_header_max_submatrix_width", c.stcHeaderMaximumSubmatrixWidth, true);
        numberField(b, 2, "stc_body_max_submatrix_width", c.stcBodyMaximumSubmatrixWidth, true);
        booleanField(b, 2, "stc_duplicate_column_risk", c.stcDuplicateColumnRisk, true);
        numberField(b, 2, "stc_total_changed_elements", c.stcTotalChangedElements, true); numberField(b, 2, "modified_coefficients", c.modifiedCoefficients, true);
        numberField(b, 2, "stc_coefficient_change_mismatch", c.stcCoefficientChangeMismatch, true);
        doubleField(b, 2, "embedding_efficiency_bits_per_modification", c.embeddingEfficiencyBitsPerModification, true);
        doubleField(b, 2, "candidate_loading", c.candidateLoading, true); doubleField(b, 2, "overall_bpnzac", c.overallBpnzAC, true);
        doubleField(b, 2, "global_modification_rate", c.modificationRateAllCoefficients, true);
        doubleField(b, 2, "candidate_modification_rate", c.modificationRateCandidates, true);
        numberField(b, 2, "zero_to_nonzero", c.modifiedZeroToNonzero, true); numberField(b, 2, "nonzero_to_zero", c.modifiedNonzeroToZero, true);
        numberField(b, 2, "high_tail_modified_coefficients", c.highTailModifiedCoefficients, true);
        doubleField(b, 2, "global_dct_js", c.dctExactJensenShannon, true); doubleField(b, 2, "target_band_js", c.targetBandJensenShannon, true);
        doubleField(b, 2, "high_tail_js", c.highTailJensenShannon, false);
        b.append(indent(1)).append("},\n");

        b.append(indent(1)).append("\"channels\": [\n");
        for (int i = 0; i < c.channelMetrics.size(); i++) {
            ChannelMetrics x = c.channelMetrics.get(i);
            b.append(indent(2)).append('{');
            inlineField(b, "channel", x.channel, true); inlineNumber(b, "eligible_candidates", x.eligibleCandidates, true);
            inlineNumber(b, "active_carriers", x.activeCarriers, true); inlineNumber(b, "modified_coefficients", x.modifiedCoefficients, true);
            inlineDouble(b, "modification_rate_eligible", x.modificationRateEligible, true); inlineDouble(b, "jensen_shannon", x.jensenShannon, false);
            b.append('}').append(i + 1 < c.channelMetrics.size() ? ",\n" : "\n");
        }
        b.append(indent(1)).append("],\n");

        b.append(indent(1)).append("\"zz_positions_combined\": [\n");
        for (int i = 0; i < c.positionMetrics.size(); i++) {
            DctPositionMetrics x = c.positionMetrics.get(i);
            b.append(indent(2)).append('{');
            inlineNumber(b, "zz0", x.zzRank, true); inlineNumber(b, "zz1", x.zzRank + 1, true);
            inlineNumber(b, "eligible_candidates", x.eligibleCandidates, true); inlineNumber(b, "modified_coefficients", x.modifiedCoefficients, true);
            inlineDouble(b, "modification_rate_eligible", x.modificationRateEligible, true); inlineDouble(b, "jensen_shannon", x.jensenShannon, false);
            b.append('}').append(i + 1 < c.positionMetrics.size() ? ",\n" : "\n");
        }
        b.append(indent(1)).append("],\n");

        b.append(indent(1)).append("\"zz_positions_by_channel\": [\n");
        for (int i = 0; i < c.channelPositionMetrics.size(); i++) {
            DctChannelPositionMetrics x = c.channelPositionMetrics.get(i);
            b.append(indent(2)).append('{');
            inlineField(b, "channel", x.channel, true); inlineNumber(b, "zz0", x.zzRank, true); inlineNumber(b, "zz1", x.thesisZzRankOneBased, true);
            inlineNumber(b, "eligible_candidates", x.eligibleCandidates, true); inlineNumber(b, "modified_coefficients", x.modifiedCoefficients, true);
            inlineDouble(b, "modification_rate_eligible", x.modificationRateEligible, true); inlineDouble(b, "jensen_shannon", x.jensenShannon, false);
            b.append('}').append(i + 1 < c.channelPositionMetrics.size() ? ",\n" : "\n");
        }
        b.append(indent(1)).append("],\n");

        b.append(indent(1)).append("\"verification\": {\n");
        booleanField(b, 2, "embedding_success", c.embeddingSuccess, true); booleanField(b, 2, "extraction_success", c.extractionSuccess, true);
        booleanField(b, 2, "aes_gcm_verified", c.aesGcmVerified, true); booleanField(b, 2, "message_matches", c.recoveredMessageMatches, false);
        b.append(indent(1)).append("}\n");
        b.append("}\n");
        AtomicFiles.writeString(path, b.toString(), StandardCharsets.UTF_8);
    }

    private static void comparison(StringBuilder b, int level, String name, ImageComparisonMetrics m, boolean comma) {
        b.append(indent(level)).append('"').append(escape(name)).append("\": {");
        b.append("\"mse\":").append(num(m.mse)).append(',');
        b.append("\"psnr\":").append(num(m.psnr)).append(',');
        b.append("\"ssim\":").append(num(m.ssim)).append(',');
        b.append("\"entropy_difference\":").append(num(m.entropyDiff)).append(',');
        b.append("\"histogram_l1\":").append(num(m.histogramL1)).append(',');
        b.append("\"kl\":").append(num(m.klDivergence)).append(',');
        b.append("\"file_size_a\":").append(m.fileSizeA).append(',');
        b.append("\"file_size_b\":").append(m.fileSizeB).append(',');
        b.append("\"file_size_difference\":").append(m.fileSizeDiff).append(',');
        b.append("\"file_size_ratio\":").append(num(m.fileSizeRatio)).append('}');
        b.append(comma ? ",\n" : "\n");
    }

    private static void field(StringBuilder b, int level, String name, String value, boolean comma) {
        b.append(indent(level)).append('"').append(escape(name)).append("\": \"").append(escape(value)).append('"');
        b.append(comma ? ",\n" : "\n");
    }
    private static void numberField(StringBuilder b, int level, String name, long value, boolean comma) {
        b.append(indent(level)).append('"').append(escape(name)).append("\": ").append(value); b.append(comma ? ",\n" : "\n");
    }
    private static void nullableNumberField(StringBuilder b, int level, String name, Number value, boolean comma) {
        b.append(indent(level)).append('"').append(escape(name)).append("\": ").append(value == null ? "null" : value); b.append(comma ? ",\n" : "\n");
    }
    private static void doubleField(StringBuilder b, int level, String name, double value, boolean comma) {
        b.append(indent(level)).append('"').append(escape(name)).append("\": ").append(num(value)); b.append(comma ? ",\n" : "\n");
    }
    private static void booleanField(StringBuilder b, int level, String name, boolean value, boolean comma) {
        b.append(indent(level)).append('"').append(escape(name)).append("\": ").append(value); b.append(comma ? ",\n" : "\n");
    }
    private static void inlineField(StringBuilder b, String name, String value, boolean comma) {
        b.append('"').append(escape(name)).append("\":\"").append(escape(value)).append('"'); if (comma) b.append(',');
    }
    private static void inlineNumber(StringBuilder b, String name, long value, boolean comma) {
        b.append('"').append(escape(name)).append("\":").append(value); if (comma) b.append(',');
    }
    private static void inlineDouble(StringBuilder b, String name, double value, boolean comma) {
        b.append('"').append(escape(name)).append("\":").append(num(value)); if (comma) b.append(',');
    }
    private static String indent(int level) { return "  ".repeat(level); }
    private static String num(double v) { return Double.isFinite(v) ? String.format(Locale.US, "%.15g", v) : "null"; }
    private static String escape(String s) { return s == null ? "" : s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r"); }
}
