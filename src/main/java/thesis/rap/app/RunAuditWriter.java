package thesis.rap.app;

import thesis.rap.alg.Candidate;
import thesis.rap.alg.CandidateSet;
import thesis.rap.alg.JpegMath;
import thesis.rap.alg.StatisticalTransitionCostModel;
import thesis.rap.alg.StatisticalCostSummary;
import thesis.rap.analysis.FileFingerprint;
import thesis.rap.analysis.JpegMetadata;
import thesis.rap.analysis.JpegMetadataAnalyzer;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegCore;
import thesis.rap.jni.JpegImage;
import thesis.rap.payload.PayloadPackage;
import thesis.rap.stc.OriginalStcCodec;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.zip.GZIPOutputStream;

/**
 * When auditing is enabled, writes the machine-readable Algorithm-4/6 audit
 * bundle and publishes it by one directory rename.
 */
public final class RunAuditWriter {
    public static final int SCHEMA_VERSION = 1;

    private RunAuditWriter() {}

    public static Path writeSuccess(
            String runId,
            String applicationVersion,
            String startedUtc,
            Path png,
            Path message,
            Path reference,
            Path stego,
            ThreeStateEmbeddingEngine.Profile profile,
            PayloadPackage.PayloadLayout payloadLayout,
            int selectedQf,
            List<QfBandCoefficientReport.Row> qfRows,
            List<ThreeStateEmbeddingEngine.Attempt> balanceAttempts,
            FixedQfEmbeddingKernel.Result kernel,
            AuditProfile auditProfile
    ) throws Exception {
        return writeSuccess(
                runId, applicationVersion, startedUtc, png, message,
                reference, stego, profile, payloadLayout, selectedQf,
                qfRows, balanceAttempts, kernel, auditProfile,
                SamplingGeometryMode.GEOMETRY_420,
                SamplingGeometry.JPEG_420);
    }

    public static Path writeSuccess(
            String runId,
            String applicationVersion,
            String startedUtc,
            Path png,
            Path message,
            Path reference,
            Path stego,
            ThreeStateEmbeddingEngine.Profile profile,
            PayloadPackage.PayloadLayout payloadLayout,
            int selectedQf,
            List<QfBandCoefficientReport.Row> qfRows,
            List<ThreeStateEmbeddingEngine.Attempt> balanceAttempts,
            FixedQfEmbeddingKernel.Result kernel,
            AuditProfile auditProfile,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry
    ) throws Exception {
        validateRunId(runId);
        if (auditProfile == null) {
            throw new IllegalArgumentException("auditProfile cannot be null");
        }
        Path temp = createTempRunDirectory(runId);
        try {
            JpegMetadata jpegMetadata = JpegMetadataAnalyzer.analyze(
                    JpegCore.readCoefficients(reference.toString()));
            CandidateAuditMetadata candidateAudit =
                    writeCandidateAudit(
                            temp, runId, selectedQf, kernel,
                            auditProfile);
            writeRunJson(
                    temp.resolve("run.json"), runId, applicationVersion,
                    startedUtc, "VERIFIED_SUCCESS", "", png, message,
                    reference, stego, profile, payloadLayout,
                    selectedQf, "", kernel.header(), jpegMetadata,
                    kernel.carrierPlan().constraintHeight(),
                    auditProfile, candidateAudit,
                    geometryMode, geometry);
            writeQfTrials(
                    temp.resolve("qf_trials.csv"), runId, qfRows);
            writeBalanceAttempts(
                    temp.resolve("balance_attempts.csv"),
                    runId, profile, balanceAttempts);
            writeJpegStatistics(
                    temp.resolve("jpeg_statistics.csv"), runId,
                    selectedQf, reference, stego,
                    kernel.candidateSet().universe());
            writeVerificationJson(
                    temp.resolve("verification.json"), runId,
                    selectedQf, true, "", "",
                    kernel.carrierPlan().constraintHeight(),
                    auditProfile, candidateAudit,
                    geometryMode, geometry);
            return publish(temp, runId);
        } catch (Exception failure) {
            deleteTree(temp);
            throw failure;
        }
    }

    public static Path writeFailure(
            String runId,
            String applicationVersion,
            String startedUtc,
            Path png,
            Path message,
            ThreeStateEmbeddingEngine.Profile profile,
            PayloadPackage.PayloadLayout payloadLayout,
            List<QfBandCoefficientReport.Row> qfRows,
            List<ThreeStateEmbeddingEngine.Attempt> balanceAttempts,
            int stcHeight,
            String failureReasonCode,
            String reason,
            AuditProfile auditProfile
    ) throws Exception {
        return writeFailure(
                runId, applicationVersion, startedUtc, png, message,
                profile, payloadLayout, qfRows, balanceAttempts,
                stcHeight, failureReasonCode, reason, auditProfile,
                SamplingGeometryMode.GEOMETRY_420,
                SamplingGeometry.JPEG_420);
    }

    public static Path writeFailure(
            String runId,
            String applicationVersion,
            String startedUtc,
            Path png,
            Path message,
            ThreeStateEmbeddingEngine.Profile profile,
            PayloadPackage.PayloadLayout payloadLayout,
            List<QfBandCoefficientReport.Row> qfRows,
            List<ThreeStateEmbeddingEngine.Attempt> balanceAttempts,
            int stcHeight,
            String failureReasonCode,
            String reason,
            AuditProfile auditProfile,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry
    ) throws Exception {
        validateRunId(runId);
        if (auditProfile == null) {
            throw new IllegalArgumentException("auditProfile cannot be null");
        }
        if (failureReasonCode == null
                || !failureReasonCode.matches("[A-Z][A-Z0-9_]{1,63}")) {
            throw new IllegalArgumentException(
                    "failureReasonCode must be a stable uppercase identifier");
        }
        Path temp = createTempRunDirectory(runId);
        try {
            writeRunJson(
                    temp.resolve("run.json"), runId, applicationVersion,
                    startedUtc,
                    "NO_VERIFIED_QF".equals(failureReasonCode)
                            ? "NO_VERIFIED_QF" : "FAILED",
                    failureReasonCode, png, message,
                    null, null, profile, payloadLayout,
                    null, reason, null, null,
                    stcHeight, auditProfile,
                    CandidateAuditMetadata.empty(),
                    geometryMode, geometry);
            writeQfTrials(
                    temp.resolve("qf_trials.csv"), runId, qfRows);
            writeBalanceAttempts(
                    temp.resolve("balance_attempts.csv"),
                    runId, profile, balanceAttempts);
            writeCsv(
                    temp.resolve("candidates.csv"),
                    List.of(candidateSummaryHeader()));
            writeCsv(
                    temp.resolve("jpeg_statistics.csv"),
                    List.of(statisticsHeader()));
            writeVerificationJson(
                    temp.resolve("verification.json"), runId,
                    null, false, failureReasonCode, reason,
                    stcHeight,
                    auditProfile, CandidateAuditMetadata.empty(),
                    geometryMode, geometry);
            return publish(temp, runId);
        } catch (Exception failure) {
            deleteTree(temp);
            throw failure;
        }
    }

    /** Compatibility overload for height-10 frozen-profile callers. */
    public static Path writeFailure(
            String runId,
            String applicationVersion,
            String startedUtc,
            Path png,
            Path message,
            ThreeStateEmbeddingEngine.Profile profile,
            PayloadPackage.PayloadLayout payloadLayout,
            List<QfBandCoefficientReport.Row> qfRows,
            List<ThreeStateEmbeddingEngine.Attempt> balanceAttempts,
            String failureReasonCode,
            String reason,
            AuditProfile auditProfile
    ) throws Exception {
        return writeFailure(
                runId, applicationVersion, startedUtc, png, message,
                profile, payloadLayout, qfRows, balanceAttempts,
                StcCarrierPlan.DEFAULT_STC_HEIGHT,
                failureReasonCode, reason, auditProfile,
                SamplingGeometryMode.GEOMETRY_420,
                SamplingGeometry.JPEG_420);
    }

    private static Path createTempRunDirectory(String runId)
            throws IOException {
        Path root = thesis.rap.runtime.RuntimePaths.results("audit");
        Files.createDirectories(root);
        return Files.createTempDirectory(root, "." + runId + ".tmp-");
    }

    private static void validateRunId(String runId) {
        if (runId == null
                || !runId.matches("[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
                || runId.contains("..")) {
            throw new IllegalArgumentException(
                    "runId is not a safe audit-directory identifier");
        }
    }

    private static Path publish(Path temp, String runId)
            throws IOException {
        Path target = temp.getParent().resolve(runId);
        if (Files.exists(target)) {
            throw new IOException(
                    "Audit run directory already exists: "
                            + target.toAbsolutePath());
        }
        try {
            return Files.move(
                    temp, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException unsupported) {
            return Files.move(temp, target);
        }
    }

    private static void writeRunJson(
            Path path,
            String runId,
            String applicationVersion,
            String startedUtc,
            String status,
            String failureReasonCode,
            Path png,
            Path message,
            Path reference,
            Path stego,
            ThreeStateEmbeddingEngine.Profile profile,
            PayloadPackage.PayloadLayout payloadLayout,
            Integer selectedQf,
            String reason,
            byte[] payloadHeader,
            JpegMetadata jpegMetadata,
            int stcHeight,
            AuditProfile auditProfile,
            CandidateAuditMetadata candidateAudit,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry
    ) throws Exception {
        StringBuilder b = new StringBuilder(4096);
        StcCarrierPlan.validateSupportedHeight(stcHeight);
        if (geometryMode == null || geometry == null
                || !geometryMode.geometries().contains(geometry)) {
            throw new IllegalArgumentException(
                    "Audit geometry must belong to its selected mode");
        }
        b.append("{\n");
        jsonNumber(b, "schema_version", SCHEMA_VERSION, true);
        jsonString(b, "run_id", runId, true);
        jsonString(b, "application_version", applicationVersion, true);
        jsonString(b, "started_utc", startedUtc, true);
        jsonString(b, "completed_utc", Instant.now().toString(), true);
        jsonString(b, "status", status, true);
        jsonString(
                b, "failure_reason_code",
                failureReasonCode, true);
        jsonString(b, "reason", reason, true);
        jsonString(b, "balance_profile", profile.code(), true);
        jsonString(b, "audit_profile", auditProfile.name(), true);
        jsonString(
                b, "candidate_audit_semantics",
                "candidates.csv=grouped-summary|"
                        + "candidate_changes.csv=modified-only-when-enabled|"
                        + "candidates_full.csv.gz=complete-universe-when-enabled",
                true);
        jsonString(
                b, "candidate_stream_hash_basis",
                "SHA-256 of canonical uncompressed detailed CSV header and "
                        + "all universe rows in universe order",
                true);
        jsonString(
                b, "candidate_stream_sha256",
                candidateAudit.candidateStreamSha256(), true);
        jsonString(
                b, "keyed_order_sha256",
                candidateAudit.keyedOrderSha256(), true);
        jsonString(
                b, "header_allocation_sha256",
                candidateAudit.headerAllocationSha256(), true);
        jsonString(
                b, "body_allocation_sha256",
                candidateAudit.bodyAllocationSha256(), true);
        jsonString(
                b, "modification_stream_sha256",
                candidateAudit.modificationStreamSha256(), true);
        jsonNumber(
                b, "candidate_universe_rows",
                candidateAudit.universeRows(), true);
        jsonNumber(
                b, "candidate_change_rows",
                candidateAudit.changeRows(), true);
        jsonString(
                b, "candidate_detail_files",
                candidateAudit.detailFiles(), true);
        jsonNumber(
                b, "qf_min",
                PayloadPackage.MIN_SUPPORTED_QF, true);
        jsonNumber(
                b, "qf_max",
                PayloadPackage.MAX_SUPPORTED_QF, true);
        jsonString(b, "geometry_mode", geometryMode.token(), true);
        jsonString(b, "sampling_geometry", geometry.label(), true);
        jsonString(
                b, "sampling_factors_expected",
                "Y=" + geometry.horizontalSampling(0) + "x"
                        + geometry.verticalSampling(0)
                        + ";Cb=" + geometry.horizontalSampling(1) + "x"
                        + geometry.verticalSampling(1)
                        + ";Cr=" + geometry.horizontalSampling(2) + "x"
                        + geometry.verticalSampling(2), true);
        jsonNullableNumber(b, "selected_qf", selectedQf, true);
        jsonNumber(
                b, "payload_bits",
                payloadLayout.totalPayloadBits(), true);
        jsonNumber(
                b, "raw_message_bytes",
                payloadLayout.rawMessageBytes(), true);
        jsonNumber(
                b, "prepared_plaintext_bytes",
                payloadLayout.preparedPlaintextBytes(), true);
        jsonNumber(
                b, "payload_bytes",
                payloadLayout.totalPayloadBytes(), true);
        jsonNumber(
                b, "payload_fixed_framing_overhead_bytes",
                payloadLayout.fixedFramingOverheadBytes(), true);
        jsonNumber(
                b, "payload_net_expansion_bytes",
                payloadLayout.netPayloadExpansionBytes(), true);
        jsonNumber(
                b, "body_message_bits",
                payloadLayout.ciphertextBytes() * 8L, true);
        jsonNumber(
                b, "payload_protocol_marker_bytes",
                payloadLayout.protocolMarkerBytes(), true);
        jsonNumber(
                b, "payload_header_bytes",
                payloadLayout.fixedHeaderBytes(), true);
        jsonNumber(
                b, "header_bits",
                payloadLayout.fixedHeaderBytes() * 8L, true);
        jsonNumber(
                b, "header_reserve",
                StcCarrierPlan.DEFAULT_HEADER_RESERVE_BITS, true);
        jsonNumber(
                b, "header_carriers",
                payloadLayout.fixedHeaderBytes() * 8L
                        + StcCarrierPlan.DEFAULT_HEADER_RESERVE_BITS,
                true);
        jsonNumber(
                b, "body_width_cap",
                StcCarrierPlan.bodyWidthCap(stcHeight), true);
        jsonNumber(
                b, "stc_height",
                stcHeight, true);
        jsonString(
                b, "stc_numeric_profile",
                OriginalStcCodec.NUMERIC_PROFILE, true);
        jsonNumber(
                b, "stc_metric_array_bytes_per_call",
                StcCarrierPlan.nativeMetricBytesEstimate(stcHeight), true);
        jsonNumber(
                b, "balance_max_attempts",
                ThreeStateEmbeddingEngine.BALANCE_MAX_ATTEMPTS, true);
        jsonNumber(
                b, "bounded_zero_budget",
                ThreeStateEmbeddingEngine.BOUNDED_ZERO_BUDGET, true);
        jsonNumber(b, "strict_zero_budget", 0, true);
        jsonDouble(
                b, "balance_step_scale",
                ThreeStateEmbeddingEngine.STEP_SCALE, true);
        jsonDouble(
                b, "balance_parameter_bound_scale",
                ThreeStateEmbeddingEngine.MULTIPLIER_BOUND_SCALE,
                true);
        jsonString(b, "target_components", "Cb,Cr", true);
        jsonNumber(b, "target_zz0_start",
                JpegMath.TARGET_ZZ_START_RANK, true);
        jsonNumber(b, "target_zz0_end",
                JpegMath.TARGET_ZZ_END_RANK, true);
        jsonString(b, "wavelet_filters", "db8-three-directional", true);
        jsonNumber(b, "wavelet_filter_taps", 16, true);
        jsonNumber(b, "symmetric_padding_samples", 16, true);
        jsonNumber(b, "unit_impact_width", 23, true);
        jsonDouble(
                b, "uniward_sigma",
                thesis.rap.alg.SiUniwardCostEngine.SIGMA, true);
        jsonDouble(
                b, "positive_preweight_offset",
                JpegMath.SI_COST_FLOOR, true);
        jsonDouble(b, "wet_cost", JpegMath.WET_COST, true);
        jsonDouble(b, "rounding_bin_tolerance", 1.0e-7, true);
        jsonDouble(
                b, "near_half_diagnostic_threshold",
                JpegMath.NEAR_HALF_DIAGNOSTIC_THRESHOLD, true);
        jsonDouble(b, "histogram_stabilizer", 1.0e-6, true);
        jsonDouble(b, "cb_channel_factor", 1.0, true);
        jsonDouble(b, "cr_channel_factor", 1.0, true);
        jsonString(b, "statistical_cost_model",
                StatisticalTransitionCostModel.MODEL_VERSION, true);
        jsonString(b, "statistical_normalization",
                "joint-CbCr-fixed-QF-p95-clipped", true);
        jsonDouble(b, "statistical_regularizer_strength",
                StatisticalTransitionCostModel.REGULARIZER_STRENGTH, true);
        jsonDouble(b, "statistical_mse_weight",
                StatisticalTransitionCostModel.MSE_WEIGHT, true);
        jsonDouble(b, "statistical_ssim_weight",
                StatisticalTransitionCostModel.SSIM_WEIGHT, true);
        jsonDouble(b, "statistical_kl_weight",
                StatisticalTransitionCostModel.KL_WEIGHT, true);
        jsonDouble(b, "statistical_entropy_weight",
                StatisticalTransitionCostModel.ENTROPY_WEIGHT, true);
        jsonDouble(b, "zero_crossing_alpha", 0.05, true);
        jsonDouble(b, "zero_crossing_denominator_offset", 1.0, true);
        jsonNumber(
                b, "keyed_order_window",
                StcKeyedOrder.localityWindow(), true);
        jsonString(
                b, "quantization_rounding",
                "half-away-from-zero", true);
        jsonString(
                b, "side_information_modulation",
                "ternary-difference", true);
        jsonNumber(b, "aes_key_bits", 128, true);
        jsonNumber(
                b, "gcm_nonce_bytes",
                PayloadPackage.NONCE_LEN, true);
        jsonNumber(
                b, "gcm_tag_bytes",
                PayloadPackage.TAG_LEN, true);
        jsonNumber(
                b, "gcm_aad_bytes",
                PayloadPackage.GEOMETRY_AAD_LEN, true);
        jsonString(
                b, "gcm_nonce_generation",
                "java.security.SecureRandom", true);
        jsonString(
                b, "gcm_nonce_hex",
                nonceHex(payloadHeader), true);
        jsonString(
                b, "payload_wire_format",
                "ciphertext-length-u16be|nonce-12|tag-12|ciphertext; qf-in-aad-only",
                true);
        jsonString(
                b, "payload_protocol_version_transport",
                "out-of-band-application-version", true);
        jsonString(
                b, "si_uniward_comparison_scope",
                "base-cost|exact-direction-error|candidate-structure|"
                        + "jpeg-statistics",
                true);
        jsonBoolean(
                b, "si_uniward_stego_generated_by_this_run",
                false, true);
        jsonString(
                b, "quantization_profile_id",
                "baseline-jpeg-standard-qf-tables-v1", true);
        jsonBoolean(
                b, "jpeg_profile_metadata_available",
                jpegMetadata != null, true);
        jsonNullableNumber(
                b, "jpeg_width",
                jpegMetadata == null ? null : jpegMetadata.width(), true);
        jsonNullableNumber(
                b, "jpeg_height",
                jpegMetadata == null ? null : jpegMetadata.height(), true);
        jsonNullableNumber(
                b, "jpeg_component_count",
                jpegMetadata == null ? null
                        : jpegMetadata.componentCount(), true);
        jsonString(
                b, "jpeg_subsampling",
                jpegMetadata == null ? ""
                        : jpegMetadata.subsampling(), true);
        jsonString(
                b, "jpeg_sampling_factors",
                jpegMetadata == null ? ""
                        : jpegMetadata.samplingFactors(), true);
        jsonString(
                b, "jpeg_quant_table_indexes",
                jpegMetadata == null ? ""
                        : jpegMetadata.quantTableIndexes(), true);
        jsonString(
                b, "jpeg_quantization_tables_sha256",
                jpegMetadata == null ? ""
                        : jpegMetadata.quantizationTableSha256(), true);
        jsonString(
                b, "reference_luminance_reconstruction",
                "canonical-quantized-y|ideal-orthonormal-idct|"
                        + "level-shift-128|clip-0-255|crop-to-image",
                true);
        jsonString(
                b, "stc_implementation_id",
                "rap-portable-scalar-binary64-stc-height" + stcHeight,
                true);
        jsonString(b, "input_png", absolute(png), true);
        jsonString(b, "message_file", absolute(message), true);
        jsonString(
                b, "reference_jpeg",
                reference == null ? "" : absolute(reference), true);
        jsonString(
                b, "stego_jpeg",
                stego == null ? "" : absolute(stego), true);
        jsonString(b, "input_png_sha256", hashIfFile(png), true);
        jsonString(b, "message_sha256", hashIfFile(message), true);
        jsonString(
                b, "reference_jpeg_sha256",
                hashIfFile(reference), true);
        jsonString(
                b, "stego_jpeg_sha256",
                hashIfFile(stego), false);
        b.append("}\n");
        Files.writeString(path, b.toString(), StandardCharsets.UTF_8);
    }

    private static String nonceHex(byte[] header) {
        if (header == null) return "";
        int offset = PayloadPackage.QF_LEN
                + PayloadPackage.CIPHERTEXT_LENGTH_FIELD_LEN;
        int end = offset + PayloadPackage.NONCE_LEN;
        if (header.length < end) {
            throw new IllegalArgumentException(
                    "Payload header is too short to contain the GCM nonce");
        }
        StringBuilder out =
                new StringBuilder(PayloadPackage.NONCE_LEN * 2);
        for (int i = offset; i < end; i++) {
            out.append(String.format(
                    Locale.ROOT, "%02x", header[i] & 0xff));
        }
        return out.toString();
    }

    private static void writeVerificationJson(
            Path path,
            String runId,
            Integer selectedQf,
            boolean verified,
            String failureReasonCode,
            String reason,
            int stcHeight,
            AuditProfile auditProfile,
            CandidateAuditMetadata candidateAudit,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry
    ) throws IOException {
        StringBuilder b = new StringBuilder(2048);
        b.append("{\n");
        jsonNumber(b, "schema_version", SCHEMA_VERSION, true);
        jsonString(b, "run_id", runId, true);
        jsonNullableNumber(b, "selected_qf", selectedQf, true);
        jsonString(b, "geometry_mode", geometryMode.token(), true);
        jsonString(b, "sampling_geometry", geometry.label(), true);
        jsonNumber(b, "stc_height", stcHeight, true);
        jsonString(
                b, "stc_numeric_profile",
                OriginalStcCodec.NUMERIC_PROFILE, true);
        jsonNumber(
                b, "stc_metric_array_bytes_per_call",
                StcCarrierPlan.nativeMetricBytesEstimate(stcHeight), true);
        jsonString(b, "audit_profile", auditProfile.name(), true);
        jsonString(
                b, "candidate_stream_sha256",
                candidateAudit.candidateStreamSha256(), true);
        jsonString(
                b, "keyed_order_sha256",
                candidateAudit.keyedOrderSha256(), true);
        jsonString(
                b, "header_allocation_sha256",
                candidateAudit.headerAllocationSha256(), true);
        jsonString(
                b, "body_allocation_sha256",
                candidateAudit.bodyAllocationSha256(), true);
        jsonString(
                b, "modification_stream_sha256",
                candidateAudit.modificationStreamSha256(), true);
        jsonString(
                b, "verification_basis",
                verified
                        ? "fixed-qf-kernel|reopened-coefficients|"
                                + "authenticated-extraction|byte-equality"
                        : "no-accepted-fixed-qf-kernel",
                true);
        jsonBoolean(b, "universe_domain_verified", verified, true);
        jsonString(b, "universe_domain_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "unit_change_verified", verified, true);
        jsonString(b, "unit_change_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "wet_direction_verified", verified, true);
        jsonString(b, "wet_direction_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "inactive_unchanged_verified", verified, true);
        jsonString(b, "inactive_unchanged_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "header_syndrome_verified", verified, true);
        jsonString(b, "header_syndrome_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "header_decode_verified", verified, true);
        jsonString(b, "header_decode_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "body_syndrome_verified", verified, true);
        jsonString(b, "body_syndrome_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "body_decode_verified", verified, true);
        jsonString(b, "body_decode_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "zero_budget_verified", verified, true);
        jsonString(b, "zero_budget_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "coefficient_writeback_verified", verified, true);
        jsonString(b, "coefficient_writeback_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "authenticated_extraction_verified", verified, true);
        jsonString(b, "authenticated_extraction_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "aes_gcm_aad_verified", verified, true);
        jsonString(b, "aes_gcm_aad_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "payload_format_verified", verified, true);
        jsonString(b, "payload_format_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "message_byte_equality_verified", verified, true);
        jsonString(b, "message_byte_equality_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonBoolean(b, "overall_verified", verified, true);
        jsonString(
                b, "success_reason_code",
                verified ? "OK" : "NOT_VERIFIED", true);
        jsonString(
                b, "failure_reason_code",
                verified ? "" : failureReasonCode, true);
        jsonString(b, "reason", reason, false);
        b.append("}\n");
        Files.writeString(path, b.toString(), StandardCharsets.UTF_8);
    }

    private static void writeQfTrials(
            Path path,
            String runId,
            List<QfBandCoefficientReport.Row> rows
    ) throws IOException {
        List<List<?>> out = new ArrayList<>();
        out.add(List.of(
                "schema_version", "run_id", "qf",
                "geometry_mode", "sampling_geometry",
                "required_payload_bits", "universe_count",
                "dry_candidate_count", "fully_wet_count",
                "negative_state_count", "zero_state_count",
                "positive_state_count", "dry_plus_directions",
                "dry_minus_directions", "wet_plus_directions",
                "wet_minus_directions", "engineering_capacity",
                "statistical_cost_model", "mse_p95_scale",
                "ssim_p95_scale", "kl_p95_scale",
                "entropy_p95_scale", "cb_dry_count", "cr_dry_count",
                "cb_mean_statistical_multiplier",
                "cr_mean_statistical_multiplier",
                "allocation_model",
                "header_gibbs_lambda",
                "body_gibbs_lambda",
                "cb_entropy_contribution_share",
                "cr_entropy_contribution_share",
                "actual_modified_cb", "actual_modified_cr",
                "actual_modified_cb_share",
                "actual_modified_cr_share",
                "capacity_pass", "kernel_ran",
                "balance_attempt_count", "terminal_status",
                "selected", "reason"));
        for (QfBandCoefficientReport.Row r : rows) {
            StatisticalCostSummary s = r.statisticalCostSummary;
            StatisticalCostSummary.Channel cb =
                    s == null ? null : s.cb();
            StatisticalCostSummary.Channel cr =
                    s == null ? null : s.cr();
            ChannelAllocationEstimate a = r.channelAllocation;
            out.add(List.of(
                    SCHEMA_VERSION, runId, r.qf,
                    r.geometryMode, r.samplingGeometry,
                    r.requiredPayloadBits, r.universeCount,
                    r.dryCount, r.fullyWetCandidates,
                    r.negativeStateCount, r.zeroStateCount,
                    r.positiveStateCount, r.dryPlusDirections,
                    r.dryMinusDirections, r.wetPlusDirections,
                    r.wetMinusDirections,
                    r.targetEngineeringCapacity,
                    s == null ? "" : s.modelVersion(),
                    s == null ? "" : s.mseP95Scale(),
                    s == null ? "" : s.ssimP95Scale(),
                    s == null ? "" : s.klP95Scale(),
                    s == null ? "" : s.entropyP95Scale(),
                    cb == null ? "" : cb.dryCount(),
                    cr == null ? "" : cr.dryCount(),
                    cb == null ? "" : cb.meanStatisticalMultiplier(),
                    cr == null ? "" : cr.meanStatisticalMultiplier(),
                    a == null ? "" : a.model(),
                    a == null ? "" : a.headerGibbsLambda(),
                    a == null ? "" : a.bodyGibbsLambda(),
                    a == null ? "" : a.cbEntropyContributionShare(),
                    a == null ? "" : a.crEntropyContributionShare(),
                    a == null ? "" : a.actualModifiedCb(),
                    a == null ? "" : a.actualModifiedCr(),
                    a == null ? "" : a.actualModifiedCbShare(),
                    a == null ? "" : a.actualModifiedCrShare(),
                    r.capacityPass, r.kernelRan,
                    r.balanceAttemptCount, r.terminalStatus,
                    r.selectedQfCandidate, r.reason));
        }
        writeCsv(path, out);
    }

    private static void writeBalanceAttempts(
            Path path,
            String runId,
            ThreeStateEmbeddingEngine.Profile profile,
            List<ThreeStateEmbeddingEngine.Attempt> attempts
    ) throws IOException {
        List<List<?>> rows = new ArrayList<>();
        rows.add(List.of(
                "schema_version", "run_id", "qf", "profile",
                "iteration", "scale", "multiplier", "step",
                "header_stc_success", "body_stc_success",
                "zero_to_plus", "zero_to_minus",
                "plus_to_zero", "minus_to_zero",
                "zero_activations", "zero_deactivations",
                "delta_h0", "syndrome_verified", "accepted",
                "status", "reason"));
        for (ThreeStateEmbeddingEngine.Attempt a : attempts) {
            ThreeStateEmbeddingEngine.TransitionCounts t =
                    a.transitions();
            rows.add(List.of(
                    SCHEMA_VERSION, runId, a.qf(), profile.code(),
                    a.iteration(), a.scale(), a.multiplier(), a.step(),
                    a.headerStcSuccess(), a.bodyStcSuccess(),
                    t.zeroToPlus(), t.zeroToMinus(),
                    t.plusToZero(), t.minusToZero(),
                    t.activations(), t.deactivations(),
                    a.zeroBinDelta(), a.syndromeVerified(),
                    a.accepted(), a.status(), a.reason()));
        }
        writeCsv(path, rows);
    }

    private static CandidateAuditMetadata writeCandidateAudit(
            Path directory,
            String runId,
            int qf,
            FixedQfEmbeddingKernel.Result kernel,
            AuditProfile auditProfile
    ) throws IOException {
        CandidateSet set = kernel.candidateSet();
        List<Candidate> universe = set.universe();
        Map<Candidate, Integer> dryIndex = identityIndex(
                set.dryBaseCostSorted());
        Map<Candidate, Integer> keyedIndex = identityIndex(
                kernel.keyedDryOrder());
        Map<Candidate, Integer> activeIndex = identityIndex(
                kernel.activeCandidates());

        double[] adjusted = new double[kernel.activeCandidates().size()];
        int[] directions = new int[adjusted.length];
        int headerCount = kernel.carrierPlan().headerCarrierCount();
        double[] headerCosts =
                kernel.balance().headerAdjustedCosts();
        double[] bodyCosts =
                kernel.balance().bodyAdjustedCosts();
        int[] headerDirections =
                kernel.balance().headerDirections();
        int[] bodyDirections =
                kernel.balance().bodyDirections();
        System.arraycopy(
                headerCosts, 0, adjusted, 0, headerCosts.length);
        System.arraycopy(
                bodyCosts, 0, adjusted, headerCount, bodyCosts.length);
        System.arraycopy(
                headerDirections, 0, directions, 0,
                headerDirections.length);
        System.arraycopy(
                bodyDirections, 0, directions, headerCount,
                bodyDirections.length);

        MessageDigest candidateDigest = sha256Digest();
        MessageDigest modificationDigest = sha256Digest();
        List<?> detailHeader = candidateDetailHeader();
        updateDigest(candidateDigest, detailHeader);
        updateDigest(
                modificationDigest,
                List.of(
                        "component", "block_index", "natural_index",
                        "q_reference", "q_stego", "selected_delta"));

        String keyedOrderSha256 = addressStreamSha256(
                kernel.keyedDryOrder());
        String headerAllocationSha256 = addressStreamSha256(
                kernel.activeCandidates().subList(0, headerCount));
        String bodyAllocationSha256 = addressStreamSha256(
                kernel.activeCandidates().subList(
                        headerCount,
                        kernel.activeCandidates().size()));

        BufferedWriter changesWriter = null;
        BufferedWriter fullWriter = null;
        StringBuilder detailFiles = new StringBuilder();

        Map<String, CandidateGroupStats> groups = new TreeMap<>();
        CandidateGroupStats total = new CandidateGroupStats(
                "ALL", "ALL", "ALL");
        JpegImage stego = kernel.intendedStego();
        long changeRows = 0;
        try {
            if (auditProfile.writesChanges()) {
                changesWriter = Files.newBufferedWriter(
                        directory.resolve("candidate_changes.csv"),
                        StandardCharsets.UTF_8);
                writeCsvRow(changesWriter, detailHeader);
                detailFiles.append("candidate_changes.csv");
            }
            if (auditProfile.writesFullCandidateUniverse()) {
                fullWriter = new BufferedWriter(new OutputStreamWriter(
                        new GZIPOutputStream(Files.newOutputStream(
                                directory.resolve(
                                        "candidates_full.csv.gz"))),
                        StandardCharsets.UTF_8));
                writeCsvRow(fullWriter, detailHeader);
                if (!detailFiles.isEmpty()) detailFiles.append('|');
                detailFiles.append("candidates_full.csv.gz");
            }
            for (int universeIndex = 0;
                 universeIndex < universe.size();
                 universeIndex++) {
                Candidate c = universe.get(universeIndex);
                Integer active = activeIndex.get(c);
                int qStego = stego.component(c.component)
                        .getCoeff(c.blockIndex, c.k);
                int delta = qStego - c.qRef;
                int zz0 = JpegMath.zigzagRankFromNaturalIndex(c.k);
                List<?> row = candidateDetailRow(
                        runId, qf, universeIndex,
                        dryIndex.get(c), keyedIndex.get(c), active,
                        c, qStego, delta,
                        active == null ? null : adjusted[active],
                        active == null ? null : directions[active]);
                updateDigest(candidateDigest, row);
                if (fullWriter != null) writeCsvRow(fullWriter, row);
                if (delta != 0) {
                    changeRows++;
                    if (changesWriter != null) {
                        writeCsvRow(changesWriter, row);
                    }
                    updateDigest(
                            modificationDigest,
                            List.of(
                                    c.component, c.blockIndex, c.k,
                                    c.qRef, qStego, delta));
                }
                String state = referenceState(c.qRef);
                String groupKey = String.format(
                        Locale.ROOT, "%03d/%03d/%s",
                        c.component, zz0, state);
                CandidateGroupStats group = groups.computeIfAbsent(
                        groupKey,
                        ignored -> new CandidateGroupStats(
                                componentName(c.component),
                                Integer.toString(zz0), state));
                group.add(c, active != null, delta);
                total.add(c, active != null, delta);
            }
        } finally {
            if (changesWriter != null) changesWriter.close();
            if (fullWriter != null) fullWriter.close();
        }

        String candidateStreamSha256 =
                HexFormat.of().formatHex(candidateDigest.digest());
        String modificationStreamSha256 =
                HexFormat.of().formatHex(modificationDigest.digest());
        CandidateAuditMetadata metadata = new CandidateAuditMetadata(
                candidateStreamSha256, keyedOrderSha256,
                headerAllocationSha256, bodyAllocationSha256,
                modificationStreamSha256, universe.size(), changeRows,
                detailFiles.toString());

        List<List<?>> summaryRows = new ArrayList<>();
        summaryRows.add(candidateSummaryHeader());
        summaryRows.add(total.row(
                runId, qf, auditProfile, metadata));
        for (CandidateGroupStats group : groups.values()) {
            summaryRows.add(group.row(
                    runId, qf, auditProfile, metadata));
        }
        writeCsv(directory.resolve("candidates.csv"), summaryRows);
        return metadata;
    }

    private static List<?> candidateDetailHeader() {
        return List.of(
                "schema_version", "run_id", "qf",
                "universe_index", "base_dry_index", "keyed_index",
                "active_index", "component", "block_index",
                "block_x", "block_y", "u", "v", "natural_index",
                "zz0", "q_reference", "q_stego", "delta",
                "transition", "raw_dct",
                "normalized_precover_f_over_q", "quant_step",
                "base_uniward_cost", "rounding_error",
                "plus_modulation_factor", "minus_modulation_factor",
                "exact_error_delta_plus",
                "exact_error_delta_minus",
                "former_four_mode_near_half_diagnostic",
                "histogram_weight", "channel_weight",
                "zero_crossing_multiplier",
                "rgb_mse_energy_proxy", "ssim_loss_proxy",
                "kl_risk_plus", "kl_risk_minus",
                "entropy_risk_plus", "entropy_risk_minus",
                "statistical_multiplier_plus",
                "statistical_multiplier_minus",
                "minimum_adjusted_cost", "rho_plus", "rho_minus",
                "minimum_adjusted_plus", "minimum_adjusted_minus",
                "base_min_cost", "plus_legal", "minus_legal",
                "fully_wet", "active", "final_adjusted_cost",
                "final_argmin_direction", "selected_delta");
    }

    private static List<?> candidateDetailRow(
            String runId,
            int qf,
            int universeIndex,
            Integer dryIndex,
            Integer keyedIndex,
            Integer activeIndex,
            Candidate c,
            int qStego,
            int delta,
            Double adjusted,
            Integer direction
    ) {
        return List.of(
                SCHEMA_VERSION, runId, qf,
                universeIndex, nullable(dryIndex), nullable(keyedIndex),
                nullable(activeIndex),
                c.component, c.blockIndex, c.blockX, c.blockY,
                c.u, c.v, c.k,
                JpegMath.zigzagRankFromNaturalIndex(c.k),
                c.qRef, qStego, delta, transition(c.qRef, qStego),
                c.rawDct, c.rawDct / c.quantStep,
                c.quantStep, c.baseUniwardCost,
                c.roundingError, c.plusModulationFactor,
                c.minusModulationFactor,
                JpegMath.exactReconstructionErrorIncrement(
                        c.roundingError, 1),
                JpegMath.exactReconstructionErrorIncrement(
                        c.roundingError, -1),
                c.formerFourModeNearHalf,
                c.histogramWeight,
                c.channelWeight, c.zeroCrossingMultiplier,
                c.rgbMseEnergyProxy, c.ssimLossProxy,
                c.klRiskPlus, c.klRiskMinus,
                c.entropyRiskPlus, c.entropyRiskMinus,
                c.statisticalMultiplierPlus,
                c.statisticalMultiplierMinus,
                c.minimumAdjustedCost, c.rhoPlus, c.rhoMinus,
                c.minimumAdjustedPlus, c.minimumAdjustedMinus,
                c.cost, c.plusLegal(), c.minusLegal(),
                c.fullyWet(), activeIndex != null,
                adjusted == null ? "" : adjusted,
                direction == null ? "" : direction,
                delta);
    }

    private static List<?> candidateSummaryHeader() {
        return List.of(
                "schema_version", "audit_profile", "run_id", "qf",
                "component", "zz0", "reference_state",
                "universe_count", "dry_count", "fully_wet_count",
                "active_count", "modified_count",
                "plus_one_count", "minus_one_count",
                "zero_to_nonzero_count", "nonzero_to_zero_count",
                "nonzero_to_nonzero_count", "near_half_count",
                "plus_legal_count", "minus_legal_count",
                "base_min_cost_min", "base_min_cost_p25",
                "base_min_cost_p50", "base_min_cost_p75",
                "base_min_cost_max", "base_min_cost_mean",
                "rho_plus_finite_mean", "rho_minus_finite_mean",
                "abs_rounding_error_mean",
                "candidate_stream_sha256", "keyed_order_sha256",
                "header_allocation_sha256",
                "body_allocation_sha256",
                "modification_stream_sha256", "detail_files");
    }

    private static void writeJpegStatistics(
            Path path,
            String runId,
            int qf,
            Path referencePath,
            Path stegoPath,
            List<Candidate> universe
    ) throws IOException {
        JpegImage reference =
                JpegCore.readCoefficients(referencePath.toString());
        JpegImage stego =
                JpegCore.readCoefficients(stegoPath.toString());
        List<List<?>> rows = new ArrayList<>();
        rows.add(statisticsHeader());

        addStat(
                rows, runId, qf, "file_size", "jpeg", "bytes",
                Files.size(referencePath), Files.size(stegoPath), "");
        addStat(
                rows, runId, qf, "entropy_size", "jpeg",
                "physical_scan_bytes",
                entropyScanBytes(referencePath),
                entropyScanBytes(stegoPath), "");

        for (int value = -1; value <= 1; value++) {
            addStat(
                    rows, runId, qf, "histogram", "universe",
                    "h(" + signed(value) + ")",
                    countUniverseValue(universe, reference, value),
                    countUniverseValue(universe, stego, value), "");
        }

        Map<String, Long> refFlow = new LinkedHashMap<>();
        Map<String, Long> stegoFlow = new LinkedHashMap<>();
        for (Candidate c : universe) {
            int a = reference.component(c.component)
                    .getCoeff(c.blockIndex, c.k);
            int b = stego.component(c.component)
                    .getCoeff(c.blockIndex, c.k);
            String scope = componentName(c.component)
                    + "/ZZ0=" + JpegMath.zigzagRankFromNaturalIndex(c.k);
            refFlow.merge(scope + "/" + a + "->" + a, 1L, Long::sum);
            stegoFlow.merge(scope + "/" + a + "->" + b, 1L, Long::sum);
            if (a != b) {
                addStat(
                        rows, runId, qf, "transition_location",
                        scope,
                        "block=" + c.blockIndex
                                + ",natural=" + c.k,
                        a, b, "delta=" + (b - a));
            }
        }
        Set<String> flowKeys = new java.util.TreeSet<>();
        flowKeys.addAll(refFlow.keySet());
        flowKeys.addAll(stegoFlow.keySet());
        for (String key : flowKeys) {
            int slash = key.lastIndexOf('/');
            addStat(
                    rows, runId, qf, "component_mode_flow",
                    key.substring(0, slash), key.substring(slash + 1),
                    refFlow.getOrDefault(key, 0L),
                    stegoFlow.getOrDefault(key, 0L), "");
        }

        AcStatistics refAc = acStatistics(reference);
        AcStatistics stegoAc = acStatistics(stego);
        addDistribution(
                rows, runId, qf, "zero_run", "all_components",
                refAc.zeroRuns, stegoAc.zeroRuns);
        addDistribution(
                rows, runId, qf, "run_size_symbol",
                "all_components", refAc.symbols, stegoAc.symbols);
        addDistribution(
                rows, runId, qf, "eob_position", "all_components",
                refAc.eobPositions, stegoAc.eobPositions);
        addDistribution(
                rows, runId, qf, "block_zero_ac_count",
                "all_components", refAc.blockZeroAc,
                stegoAc.blockZeroAc);
        addStat(
                rows, runId, qf, "eob", "all_components",
                "count", refAc.eobCount, stegoAc.eobCount, "");
        addStat(
                rows, runId, qf, "zrl", "all_components",
                "count", refAc.zrlCount, stegoAc.zrlCount, "");

        int sharedComponents = Math.min(
                reference.components.length, stego.components.length);
        for (int component = 0; component < sharedComponents;
             component++) {
            AcStatistics refComponent =
                    acStatistics(reference, component);
            AcStatistics stegoComponent =
                    acStatistics(stego, component);
            String scope = componentName(component);
            addDistribution(
                    rows, runId, qf, "zero_run", scope,
                    refComponent.zeroRuns, stegoComponent.zeroRuns);
            addDistribution(
                    rows, runId, qf, "run_size_symbol", scope,
                    refComponent.symbols, stegoComponent.symbols);
            addDistribution(
                    rows, runId, qf, "eob_position", scope,
                    refComponent.eobPositions,
                    stegoComponent.eobPositions);
            addDistribution(
                    rows, runId, qf, "block_zero_ac_count", scope,
                    refComponent.blockZeroAc,
                    stegoComponent.blockZeroAc);
            addStat(
                    rows, runId, qf, "eob", scope, "count",
                    refComponent.eobCount,
                    stegoComponent.eobCount, "");
            addStat(
                    rows, runId, qf, "zrl", scope, "count",
                    refComponent.zrlCount,
                    stegoComponent.zrlCount, "");
        }
        writeCsv(path, rows);
    }

    private static List<?> statisticsHeader() {
        return List.of(
                "schema_version", "run_id", "qf",
                "category", "scope", "key",
                "reference_value", "stego_value", "delta",
                "detail");
    }

    private static AcStatistics acStatistics(JpegImage image) {
        return acStatistics(image, -1);
    }

    private static AcStatistics acStatistics(
            JpegImage image,
            int componentFilter
    ) {
        AcStatistics out = new AcStatistics();
        for (int componentIndex = 0;
             componentIndex < image.components.length;
             componentIndex++) {
            if (componentFilter >= 0
                    && componentIndex != componentFilter) {
                continue;
            }
            JpegComponent component =
                    image.components[componentIndex];
            int blocks = component.blockCount();
            for (int block = 0; block < blocks; block++) {
                int run = 0;
                int zeroAc = 0;
                for (int rank = 1; rank < 64; rank++) {
                    int k = JpegMath.naturalIndexFromZigzagRank(rank);
                    int value = component.getCoeff(block, k);
                    if (value == 0) {
                        run++;
                        zeroAc++;
                        continue;
                    }
                    out.zeroRuns.merge(
                            Integer.toString(run), 1L, Long::sum);
                    while (run >= 16) {
                        out.symbols.merge("F0", 1L, Long::sum);
                        out.zrlCount++;
                        run -= 16;
                    }
                    int size = magnitudeCategory(value);
                    String symbol = String.format(
                            Locale.ROOT, "%X%X", run, size);
                    out.symbols.merge(symbol, 1L, Long::sum);
                    run = 0;
                }
                if (run > 0) {
                    out.symbols.merge("00", 1L, Long::sum);
                    out.eobCount++;
                    out.eobPositions.merge(
                            Integer.toString(64 - run),
                            1L, Long::sum);
                    out.zeroRuns.merge(
                            "EOB:" + run, 1L, Long::sum);
                }
                out.blockZeroAc.merge(
                        Integer.toString(zeroAc), 1L, Long::sum);
            }
        }
        return out;
    }

    private static int magnitudeCategory(int value) {
        long magnitude = Math.abs((long) value);
        int bits = 0;
        while (magnitude != 0) {
            bits++;
            magnitude >>>= 1;
        }
        return bits;
    }

    private static long entropyScanBytes(Path jpeg)
            throws IOException {
        byte[] bytes = Files.readAllBytes(jpeg);
        long total = 0;
        int i = 0;
        while (i + 1 < bytes.length) {
            if ((bytes[i] & 0xff) != 0xff) {
                i++;
                continue;
            }
            int marker = bytes[i + 1] & 0xff;
            if (marker == 0xda) {
                if (i + 4 > bytes.length) break;
                int length = ((bytes[i + 2] & 0xff) << 8)
                        | (bytes[i + 3] & 0xff);
                i += 2 + length;
                while (i < bytes.length) {
                    if ((bytes[i] & 0xff) != 0xff) {
                        total++;
                        i++;
                        continue;
                    }
                    if (i + 1 >= bytes.length) break;
                    int next = bytes[i + 1] & 0xff;
                    if (next == 0x00) {
                        total += 2;
                        i += 2;
                    } else if (next >= 0xd0 && next <= 0xd7) {
                        i += 2;
                    } else {
                        break;
                    }
                }
                continue;
            }
            if (marker == 0xd8 || marker == 0xd9
                    || (marker >= 0xd0 && marker <= 0xd7)
                    || marker == 0x01) {
                i += 2;
                continue;
            }
            if (i + 4 > bytes.length) break;
            int length = ((bytes[i + 2] & 0xff) << 8)
                    | (bytes[i + 3] & 0xff);
            if (length < 2) break;
            i += 2 + length;
        }
        return total;
    }

    private static long countUniverseValue(
            List<Candidate> universe,
            JpegImage image,
            int value
    ) {
        long total = 0;
        for (Candidate c : universe) {
            if (image.component(c.component)
                    .getCoeff(c.blockIndex, c.k) == value) {
                total++;
            }
        }
        return total;
    }

    private static void addDistribution(
            List<List<?>> rows,
            String runId,
            int qf,
            String category,
            String scope,
            Map<String, Long> reference,
            Map<String, Long> stego
    ) {
        Set<String> keys = new java.util.TreeSet<>();
        keys.addAll(reference.keySet());
        keys.addAll(stego.keySet());
        for (String key : keys) {
            addStat(
                    rows, runId, qf, category, scope, key,
                    reference.getOrDefault(key, 0L),
                    stego.getOrDefault(key, 0L), "");
        }
    }

    private static void addStat(
            List<List<?>> rows,
            String runId,
            int qf,
            String category,
            String scope,
            String key,
            long reference,
            long stego,
            String detail
    ) {
        rows.add(List.of(
                SCHEMA_VERSION, runId, qf,
                category, scope, key,
                reference, stego, stego - reference, detail));
    }

    private static Map<Candidate, Integer> identityIndex(
            List<Candidate> candidates
    ) {
        Map<Candidate, Integer> out = new IdentityHashMap<>();
        for (int i = 0; i < candidates.size(); i++) {
            out.put(candidates.get(i), i);
        }
        return out;
    }

    private static String transition(int a, int b) {
        if (a == b) return "UNCHANGED";
        if (a == 0 && b == 1) return "0_TO_PLUS1";
        if (a == 0 && b == -1) return "0_TO_MINUS1";
        if (a == 1 && b == 0) return "PLUS1_TO_0";
        if (a == -1 && b == 0) return "MINUS1_TO_0";
        return (b - a) > 0 ? "OTHER_PLUS1" : "OTHER_MINUS1";
    }

    private static String componentName(int component) {
        return switch (component) {
            case 0 -> "Y";
            case 1 -> "Cb";
            case 2 -> "Cr";
            default -> "C" + component;
        };
    }

    private static String signed(int value) {
        return value > 0 ? "+" + value : Integer.toString(value);
    }

    private static String nullable(Object value) {
        return value == null ? "" : value.toString();
    }

    private static String referenceState(int value) {
        if (value < 0) return "NEGATIVE";
        if (value > 0) return "POSITIVE";
        return "ZERO";
    }

    private static MessageDigest sha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (java.security.NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(
                    "Required SHA-256 implementation is unavailable",
                    impossible);
        }
    }

    private static String addressStreamSha256(
            List<Candidate> candidates
    ) {
        MessageDigest digest = sha256Digest();
        updateDigest(
                digest,
                List.of("component", "block_index", "natural_index"));
        for (Candidate candidate : candidates) {
            updateDigest(
                    digest,
                    List.of(
                            candidate.component,
                            candidate.blockIndex,
                            candidate.k));
        }
        return HexFormat.of().formatHex(digest.digest());
    }

    private static void updateDigest(
            MessageDigest digest,
            List<?> row
    ) {
        digest.update(csvLine(row).getBytes(StandardCharsets.UTF_8));
        digest.update((byte) '\n');
    }

    private static String csvLine(List<?> row) {
        StringBuilder line = new StringBuilder(row.size() * 16);
        for (int i = 0; i < row.size(); i++) {
            if (i > 0) line.append(',');
            line.append(csv(row.get(i)));
        }
        return line.toString();
    }

    private static void writeCsvRow(
            BufferedWriter writer,
            List<?> row
    ) throws IOException {
        writer.write(csvLine(row));
        writer.newLine();
    }

    private static void writeCsv(Path path, List<List<?>> rows)
            throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(
                path, StandardCharsets.UTF_8)) {
            for (List<?> row : rows) {
                writeCsvRow(writer, row);
            }
        }
    }

    private static String csv(Object value) {
        String s;
        if (value == null) s = "";
        else if (value instanceof Double d) {
            s = Double.isFinite(d)
                    ? String.format(Locale.US, "%.17g", d) : "";
        } else if (value instanceof Float f) {
            s = Float.isFinite(f)
                    ? String.format(Locale.US, "%.9g", f) : "";
        } else {
            s = value.toString();
        }
        if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0
                || s.indexOf('\n') >= 0 || s.indexOf('\r') >= 0) {
            return '"' + s.replace("\"", "\"\"") + '"';
        }
        return s;
    }

    private static String absolute(Path path) {
        return path == null ? "" : path.toAbsolutePath().toString();
    }

    private static String hashIfFile(Path path) throws Exception {
        return path != null && Files.isRegularFile(path)
                ? FileFingerprint.sha256(path) : "";
    }

    private static void jsonString(
            StringBuilder b, String name, String value, boolean comma
    ) {
        b.append("  \"").append(jsonEscape(name)).append("\": \"")
                .append(jsonEscape(value == null ? "" : value))
                .append('"').append(comma ? ",\n" : "\n");
    }

    private static void jsonNumber(
            StringBuilder b, String name, long value, boolean comma
    ) {
        b.append("  \"").append(jsonEscape(name)).append("\": ")
                .append(value).append(comma ? ",\n" : "\n");
    }

    private static void jsonNullableNumber(
            StringBuilder b, String name, Number value, boolean comma
    ) {
        b.append("  \"").append(jsonEscape(name)).append("\": ")
                .append(value == null ? "null" : value)
                .append(comma ? ",\n" : "\n");
    }

    private static void jsonDouble(
            StringBuilder b, String name, double value, boolean comma
    ) {
        b.append("  \"").append(jsonEscape(name)).append("\": ")
                .append(String.format(Locale.US, "%.17g", value))
                .append(comma ? ",\n" : "\n");
    }

    private static void jsonBoolean(
            StringBuilder b, String name, boolean value, boolean comma
    ) {
        b.append("  \"").append(jsonEscape(name)).append("\": ")
                .append(value).append(comma ? ",\n" : "\n");
    }

    private static String jsonEscape(String value) {
        return value.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static void deleteTree(Path root) throws IOException {
        if (root == null || !Files.exists(root)) return;
        try (var paths = Files.walk(root)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(path);
            }
        }
    }

    private record CandidateAuditMetadata(
            String candidateStreamSha256,
            String keyedOrderSha256,
            String headerAllocationSha256,
            String bodyAllocationSha256,
            String modificationStreamSha256,
            long universeRows,
            long changeRows,
            String detailFiles
    ) {
        static CandidateAuditMetadata empty() {
            return new CandidateAuditMetadata(
                    "", "", "", "", "", 0L, 0L, "");
        }
    }

    private static final class CandidateGroupStats {
        private final String component;
        private final String zz0;
        private final String referenceState;
        private final DoubleValues baseCosts = new DoubleValues();
        private long universe;
        private long dry;
        private long fullyWet;
        private long active;
        private long modified;
        private long plusOne;
        private long minusOne;
        private long zeroToNonzero;
        private long nonzeroToZero;
        private long nonzeroToNonzero;
        private long nearHalf;
        private long plusLegal;
        private long minusLegal;
        private double rhoPlusSum;
        private long rhoPlusCount;
        private double rhoMinusSum;
        private long rhoMinusCount;
        private double absRoundingErrorSum;
        private long roundingErrorCount;

        CandidateGroupStats(
                String component,
                String zz0,
                String referenceState
        ) {
            this.component = component;
            this.zz0 = zz0;
            this.referenceState = referenceState;
        }

        void add(Candidate candidate, boolean isActive, int delta) {
            universe++;
            if (candidate.fullyWet()) fullyWet++;
            else dry++;
            if (isActive) active++;
            if (candidate.formerFourModeNearHalf) nearHalf++;
            if (candidate.plusLegal()) {
                plusLegal++;
                rhoPlusSum += candidate.rhoPlus;
                rhoPlusCount++;
            }
            if (candidate.minusLegal()) {
                minusLegal++;
                rhoMinusSum += candidate.rhoMinus;
                rhoMinusCount++;
            }
            if (!candidate.fullyWet()
                    && Double.isFinite(candidate.cost)
                    && !JpegMath.isWetCost(candidate.cost)) {
                baseCosts.add(candidate.cost);
            }
            if (Double.isFinite(candidate.roundingError)) {
                absRoundingErrorSum +=
                        Math.abs(candidate.roundingError);
                roundingErrorCount++;
            }
            if (delta == 0) return;
            modified++;
            if (delta > 0) plusOne++;
            else minusOne++;
            int qStego = candidate.qRef + delta;
            if (candidate.qRef == 0 && qStego != 0) {
                zeroToNonzero++;
            } else if (candidate.qRef != 0 && qStego == 0) {
                nonzeroToZero++;
            } else {
                nonzeroToNonzero++;
            }
        }

        List<?> row(
                String runId,
                int qf,
                AuditProfile auditProfile,
                CandidateAuditMetadata metadata
        ) {
            return List.of(
                    SCHEMA_VERSION, auditProfile.name(), runId, qf,
                    component, zz0, referenceState,
                    universe, dry, fullyWet, active, modified,
                    plusOne, minusOne, zeroToNonzero,
                    nonzeroToZero, nonzeroToNonzero, nearHalf,
                    plusLegal, minusLegal,
                    baseCosts.minimum(),
                    baseCosts.quantile(0.25),
                    baseCosts.quantile(0.50),
                    baseCosts.quantile(0.75),
                    baseCosts.maximum(), baseCosts.mean(),
                    mean(rhoPlusSum, rhoPlusCount),
                    mean(rhoMinusSum, rhoMinusCount),
                    mean(absRoundingErrorSum, roundingErrorCount),
                    metadata.candidateStreamSha256(),
                    metadata.keyedOrderSha256(),
                    metadata.headerAllocationSha256(),
                    metadata.bodyAllocationSha256(),
                    metadata.modificationStreamSha256(),
                    metadata.detailFiles());
        }

        private static double mean(double sum, long count) {
            return count == 0L ? Double.NaN : sum / count;
        }
    }

    private static final class DoubleValues {
        private double[] values = new double[16];
        private int size;
        private double sum;
        private double minimum = Double.POSITIVE_INFINITY;
        private double maximum = Double.NEGATIVE_INFINITY;
        private double[] sorted;

        void add(double value) {
            if (!Double.isFinite(value)) return;
            if (size == values.length) {
                values = Arrays.copyOf(values, values.length * 2);
            }
            values[size++] = value;
            sum += value;
            minimum = Math.min(minimum, value);
            maximum = Math.max(maximum, value);
            sorted = null;
        }

        double minimum() {
            return size == 0 ? Double.NaN : minimum;
        }

        double maximum() {
            return size == 0 ? Double.NaN : maximum;
        }

        double mean() {
            return size == 0 ? Double.NaN : sum / size;
        }

        double quantile(double probability) {
            if (size == 0) return Double.NaN;
            if (probability < 0.0 || probability > 1.0) {
                throw new IllegalArgumentException(
                        "probability must be in [0,1]");
            }
            if (sorted == null) {
                sorted = Arrays.copyOf(values, size);
                Arrays.sort(sorted);
            }
            double position = probability * (size - 1);
            int lower = (int) Math.floor(position);
            int upper = (int) Math.ceil(position);
            if (lower == upper) return sorted[lower];
            double fraction = position - lower;
            return sorted[lower]
                    + fraction * (sorted[upper] - sorted[lower]);
        }
    }

    private static final class AcStatistics {
        final Map<String, Long> zeroRuns = new HashMap<>();
        final Map<String, Long> symbols = new HashMap<>();
        final Map<String, Long> eobPositions = new HashMap<>();
        final Map<String, Long> blockZeroAc = new HashMap<>();
        long eobCount;
        long zrlCount;
    }
}
