package thesis.rap.app;

import thesis.rap.alg.JpegMath;
import thesis.rap.alg.SiUniwardCostEngine;
import thesis.rap.alg.StatisticalTransitionCostModel;
import thesis.rap.analysis.ChannelMetrics;
import thesis.rap.analysis.CsvUtil;
import thesis.rap.analysis.EvaluationSummary;
import thesis.rap.analysis.ImageComparisonMetrics;
import thesis.rap.analysis.JpegCoefficientReport;
import thesis.rap.analysis.JpegMetadata;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;
import thesis.rap.payload.PayloadPackage;
import thesis.rap.stc.OriginalStcCodec;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Concurrent one-row-per-pair RESULT_ONLY ledger with a rich terminal-QF
 * snapshot and final-QF aggregate evaluation.
 *
 * <p>The writer never receives or publishes the complete QF history. Failed
 * searches contain only the last tested QF; successful searches contain only
 * the selected verified QF. Unavailable measurements are empty CSV cells,
 * never invented zeroes.</p>
 */
public final class ResultAuditWriter {
    public static final String SCHEMA_VERSION = "1.0";
    public static final Path DEFAULT_CSV =
            thesis.rap.runtime.RuntimePaths.results("audit_final_qf.csv");
    private static final int MAX_TEXT_CHARACTERS = 8_192;
    private static final int MAX_FINAL_DETAIL_CHARACTERS = 4_096;

    private ResultAuditWriter() {}

    /** All inputs required to flatten one terminal pair record. */
    public static final class Entry {
        private final String startedUtc;
        private final String completedUtc;
        private final String runId;
        private final String applicationVersion;
        private final String action;
        private final Path image;
        private final Path message;
        private final Path output;
        private final int qfMin;
        private final int qfMax;
        private final String selectionPolicy;
        private final SamplingGeometryMode geometryMode;
        private final SamplingGeometry samplingGeometry;
        private final String status;
        private final boolean embeddingSuccess;
        private final boolean extractionSuccess;
        private final boolean coefficientWritebackVerified;
        private final boolean nativeStcVerified;
        private final boolean aesGcmVerified;
        private final boolean exactMessageVerified;
        private final String inputPngSha256;
        private final String messageSha256;
        private final String stegoSha256;
        private final String failureReasonCode;
        private final String finalDetail;
        private final FailureDiagnostics.Info exception;
        private final FinalQfAuditState finalQf;
        private final PayloadPackage.PayloadLayout payloadLayout;
        private final EvaluationSummary evaluation;
        private final Long embeddingElapsedMillis;
        private final Long evaluationElapsedMillis;
        private final String evaluationUnavailableReason;
        private final Long inputImageBytes;
        private final Integer inputWidth;
        private final Integer inputHeight;
        private final Long inputMessageBytes;
        private final Integer libjpegVersion;
        private final Long stegoBytes;
        private final Integer outputWidth;
        private final Integer outputHeight;
        private final Integer outputComponents;

        private Entry(Builder builder) {
            startedUtc = safe(builder.startedUtc);
            completedUtc = safe(builder.completedUtc);
            runId = required(builder.runId, "runId");
            applicationVersion = required(
                    builder.applicationVersion, "applicationVersion");
            action = required(builder.action, "action");
            image = builder.image;
            message = builder.message;
            output = builder.output;
            qfMin = builder.qfMin;
            qfMax = builder.qfMax;
            if (qfMin < 1 || qfMax > 100 || qfMin > qfMax) {
                throw new IllegalArgumentException(
                        "Invalid QF range: " + qfMin + "-" + qfMax);
            }
            selectionPolicy = safe(builder.selectionPolicy);
            if (builder.geometryMode == null
                    || builder.samplingGeometry == null) {
                throw new IllegalArgumentException(
                        "geometryMode and samplingGeometry are required");
            }
            geometryMode = builder.geometryMode;
            samplingGeometry = builder.samplingGeometry;
            if (!geometryMode.geometries().contains(
                    samplingGeometry)) {
                throw new IllegalArgumentException(
                        "samplingGeometry is not included in geometryMode");
            }
            status = required(builder.status, "status");
            embeddingSuccess = builder.embeddingSuccess;
            extractionSuccess = builder.extractionSuccess;
            coefficientWritebackVerified =
                    builder.coefficientWritebackVerified;
            nativeStcVerified = builder.nativeStcVerified;
            aesGcmVerified = builder.aesGcmVerified;
            exactMessageVerified = builder.exactMessageVerified;
            inputPngSha256 = safe(builder.inputPngSha256);
            messageSha256 = safe(builder.messageSha256);
            stegoSha256 = safe(builder.stegoSha256);
            exception = FailureDiagnostics.info(builder.exception);
            failureReasonCode = safe(builder.failureReasonCode);
            finalDetail = compact(
                    builder.finalDetail, MAX_FINAL_DETAIL_CHARACTERS);
            finalQf = builder.finalQf == null
                    ? new FinalQfAuditState() : builder.finalQf;
            payloadLayout = builder.payloadLayout;
            evaluation = builder.evaluation;
            embeddingElapsedMillis = builder.embeddingElapsedMillis;
            evaluationElapsedMillis = builder.evaluationElapsedMillis;
            evaluationUnavailableReason = safe(
                    builder.evaluationUnavailableReason);
            inputImageBytes = builder.inputImageBytes;
            inputWidth = builder.inputWidth;
            inputHeight = builder.inputHeight;
            inputMessageBytes = builder.inputMessageBytes;
            libjpegVersion = builder.libjpegVersion;
            stegoBytes = builder.stegoBytes;
            outputWidth = builder.outputWidth;
            outputHeight = builder.outputHeight;
            outputComponents = builder.outputComponents;
        }

        public static Builder builder() {
            return new Builder();
        }
    }

    /** Grouped builder methods keep all terminal paths explicit. */
    public static final class Builder {
        private String startedUtc;
        private String completedUtc;
        private String runId;
        private String applicationVersion;
        private String action;
        private Path image;
        private Path message;
        private Path output;
        private int qfMin = 75;
        private int qfMax = 100;
        private String selectionPolicy = "capacity";
        private SamplingGeometryMode geometryMode =
                SamplingGeometryMode.GEOMETRY_420;
        private SamplingGeometry samplingGeometry =
                SamplingGeometry.JPEG_420;
        private String status;
        private boolean embeddingSuccess;
        private boolean extractionSuccess;
        private boolean coefficientWritebackVerified;
        private boolean nativeStcVerified;
        private boolean aesGcmVerified;
        private boolean exactMessageVerified;
        private String inputPngSha256;
        private String messageSha256;
        private String stegoSha256;
        private String failureReasonCode;
        private String finalDetail;
        private Throwable exception;
        private FinalQfAuditState finalQf;
        private PayloadPackage.PayloadLayout payloadLayout;
        private EvaluationSummary evaluation;
        private Long embeddingElapsedMillis;
        private Long evaluationElapsedMillis;
        private String evaluationUnavailableReason;
        private Long inputImageBytes;
        private Integer inputWidth;
        private Integer inputHeight;
        private Long inputMessageBytes;
        private Integer libjpegVersion;
        private Long stegoBytes;
        private Integer outputWidth;
        private Integer outputHeight;
        private Integer outputComponents;

        private Builder() {}

        public Builder identity(
                String start,
                String completion,
                String id,
                String version,
                String operation
        ) {
            startedUtc = start;
            completedUtc = completion;
            runId = id;
            applicationVersion = version;
            action = operation;
            return this;
        }

        public Builder paths(Path png, Path msg, Path stego) {
            image = png;
            message = msg;
            output = stego;
            return this;
        }

        public Builder geometry(
                SamplingGeometryMode mode,
                SamplingGeometry geometry
        ) {
            geometryMode = mode;
            samplingGeometry = geometry;
            return this;
        }

        public Builder qfSearch(int min, int max, String policy) {
            qfMin = min;
            qfMax = max;
            selectionPolicy = policy;
            return this;
        }

        public Builder outcome(
                String terminalStatus,
                boolean embedded,
                boolean extracted,
                boolean writeback,
                boolean nativeStc,
                boolean aesGcm,
                boolean exactMessage
        ) {
            status = terminalStatus;
            embeddingSuccess = embedded;
            extractionSuccess = extracted;
            coefficientWritebackVerified = writeback;
            nativeStcVerified = nativeStc;
            aesGcmVerified = aesGcm;
            exactMessageVerified = exactMessage;
            return this;
        }

        public Builder hashes(
                String imageHash,
                String messageHash,
                String outputHash
        ) {
            inputPngSha256 = imageHash;
            messageSha256 = messageHash;
            stegoSha256 = outputHash;
            return this;
        }

        public Builder exception(Throwable error) {
            exception = error;
            return this;
        }

        public Builder failure(String reasonCode, String detail) {
            failureReasonCode = reasonCode;
            finalDetail = detail;
            return this;
        }

        Builder finalQf(FinalQfAuditState state) {
            finalQf = state;
            return this;
        }

        public Builder payload(PayloadPackage.PayloadLayout layout) {
            payloadLayout = layout;
            return this;
        }

        public Builder evaluation(
                EvaluationSummary summary,
                long embedMillis,
                long evaluateMillis
        ) {
            evaluation = summary;
            embeddingElapsedMillis = embedMillis;
            evaluationElapsedMillis = evaluateMillis;
            evaluationUnavailableReason = summary == null
                    ? "FINAL_QF_EVALUATION_NOT_AVAILABLE" : "";
            return this;
        }

        public Builder evaluationUnavailable(
                String reason,
                Long embedMillis
        ) {
            evaluation = null;
            evaluationUnavailableReason = reason;
            embeddingElapsedMillis = embedMillis;
            evaluationElapsedMillis = null;
            return this;
        }

        public Builder inputEvidence(
                long imageBytes,
                int width,
                int height,
                long messageBytes,
                int jpegLibraryVersion
        ) {
            inputImageBytes = imageBytes;
            inputWidth = width;
            inputHeight = height;
            inputMessageBytes = messageBytes;
            libjpegVersion = jpegLibraryVersion;
            return this;
        }

        public Builder outputEvidence(
                long bytes,
                int width,
                int height,
                int components
        ) {
            stegoBytes = bytes;
            outputWidth = width;
            outputHeight = height;
            outputComponents = components;
            return this;
        }

        public Entry build() {
            return new Entry(this);
        }
    }

    public static void append(Entry entry) throws IOException {
        append(DEFAULT_CSV, entry);
    }

    public static void append(Path csv, Entry entry) throws IOException {
        if (csv == null || entry == null) {
            throw new IllegalArgumentException("csv and entry cannot be null");
        }
        LinkedHashMap<String, String> values = flatten(entry);
        CsvUtil.append(csv, List.copyOf(values.keySet()),
                List.copyOf(values.values()));
    }

    
    public static void append(
            Path csv,
            String startedUtc,
            String completedUtc,
            String runId,
            String applicationVersion,
            String action,
            Path image,
            Path message,
            Path output,
            String status,
            boolean embeddingSuccess,
            boolean extractionSuccess,
            boolean coefficientWritebackVerified,
            boolean nativeStcVerified,
            boolean aesGcmVerified,
            boolean exactMessageVerified,
            String inputPngSha256,
            String messageSha256,
            String stegoSha256,
            String failureReasonCode,
            String finalDetail
    ) throws IOException {
        append(csv, Entry.builder()
                .identity(startedUtc, completedUtc, runId,
                        applicationVersion, action)
                .paths(image, message, output)
                .outcome(status, embeddingSuccess, extractionSuccess,
                        coefficientWritebackVerified, nativeStcVerified,
                        aesGcmVerified, exactMessageVerified)
                .hashes(inputPngSha256, messageSha256, stegoSha256)
                .failure(failureReasonCode, finalDetail)
                .evaluationUnavailable(
                        "CALLER_DID_NOT_SUPPLY_FINAL_QF_STATE", null)
                .build());
    }

    private static LinkedHashMap<String, String> flatten(Entry entry) {
        LinkedHashMap<String, String> out = new LinkedHashMap<>();
        put(out, "schema_version", SCHEMA_VERSION);
        put(out, "audit_mode", AuditMode.RESULT_ONLY.name());
        put(out, "record_model",
                "ONE_IMAGE_MESSAGE_GEOMETRY_ONE_TERMINAL_QF");
        put(out, "qf_history_retained", false);
        put(out, "balance_history_retained", false);
        put(out, "per_zz_rows_retained", false);
        put(out, "null_policy", "EMPTY_CELL_MEANS_NOT_AVAILABLE");
        put(out, "started_utc", entry.startedUtc);
        put(out, "completed_utc", entry.completedUtc);
        put(out, "total_elapsed_ms", elapsedMillis(
                entry.startedUtc, entry.completedUtc));
        put(out, "run_id", entry.runId);
        put(out, "application_version", entry.applicationVersion);
        put(out, "action", entry.action);
        put(out, "image_name", fileName(entry.image));
        put(out, "image_id_text", fileStem(entry.image));
        put(out, "message_name", fileName(entry.message));
        put(out, "message_id_text", fileStem(entry.message));
        put(out, "output_name", fileName(entry.output));
        put(out, "image_path", absolutePath(entry.image));
        put(out, "message_path", absolutePath(entry.message));
        put(out, "output_path", absolutePath(entry.output));
        put(out, "qf_min", entry.qfMin);
        put(out, "qf_max", entry.qfMax);
        put(out, "selection_policy", entry.selectionPolicy);
        put(out, "geometry_mode", entry.geometryMode.token());
        put(out, "sampling_geometry",
                entry.samplingGeometry.label());
        put(out, "geometry_signature_hex",
                entry.samplingGeometry.signatureHex());
        put(out, "carrier_order_domain", "ahp-order-v1");
        put(out, "sampling_factors_expected",
                "Y=" + entry.samplingGeometry.horizontalSampling(0)
                        + "x"
                        + entry.samplingGeometry.verticalSampling(0)
                        + ";Cb="
                        + entry.samplingGeometry.horizontalSampling(1)
                        + "x"
                        + entry.samplingGeometry.verticalSampling(1)
                        + ";Cr="
                        + entry.samplingGeometry.horizontalSampling(2)
                        + "x"
                        + entry.samplingGeometry.verticalSampling(2));
        put(out, "target_band", JpegMath.targetBandLabel());
        put(out, "high_tail_control_band", JpegMath.highTailLabel());
        put(out, "cost_model_impact_window",
                SiUniwardCostEngine.IMPACT_WINDOW_ALIGNMENT);
        put(out, "cost_model_impact_window_offset",
                SiUniwardCostEngine.IMPACT_WINDOW_OFFSET);
        put(out, "cost_model_sigma", "2^-6");
        put(out, "cost_model_si_offset", "1e-4");
        put(out, "statistical_cost_model",
                StatisticalTransitionCostModel.MODEL_VERSION);
        put(out, "statistical_normalization",
                "JOINT_CB_CR_FIXED_QF_P95_CLIPPED_0_1");
        put(out, "statistical_regularizer_strength",
                StatisticalTransitionCostModel.REGULARIZER_STRENGTH);
        put(out, "statistical_mse_weight",
                StatisticalTransitionCostModel.MSE_WEIGHT);
        put(out, "statistical_ssim_weight",
                StatisticalTransitionCostModel.SSIM_WEIGHT);
        put(out, "statistical_kl_weight",
                StatisticalTransitionCostModel.KL_WEIGHT);
        put(out, "statistical_entropy_weight",
                StatisticalTransitionCostModel.ENTROPY_WEIGHT);
        put(out, "wet_cost_sentinel", JpegMath.WET_COST);
        put(out, "stc_supported_heights", "10,12");
        put(out, "stc_numeric_profile", OriginalStcCodec.NUMERIC_PROFILE);
        put(out, "stc_default_height",
                StcCarrierPlan.DEFAULT_STC_HEIGHT);
        put(out, "stc_header_reserve_bits",
                StcCarrierPlan.DEFAULT_HEADER_RESERVE_BITS);
        put(out, "keys_recorded", false);
        put(out, "security_claim", "NONE_FROM_DIAGNOSTIC_METRICS");

        put(out, "status", entry.status);
        put(out, "embedding_success", entry.embeddingSuccess);
        put(out, "extraction_success", entry.extractionSuccess);
        put(out, "coefficient_writeback_verified",
                entry.coefficientWritebackVerified);
        put(out, "native_stc_verified", entry.nativeStcVerified);
        put(out, "aes_gcm_verified", entry.aesGcmVerified);
        put(out, "exact_message_verified", entry.exactMessageVerified);
        put(out, "input_png_sha256", entry.inputPngSha256);
        put(out, "message_sha256", entry.messageSha256);
        put(out, "stego_sha256", entry.stegoSha256);
        put(out, "failure_reason_code", entry.failureReasonCode);
        put(out, "final_detail", entry.finalDetail);
        put(out, "exception_class", entry.exception.className());
        put(out, "exception_message", compact(entry.exception.message(), MAX_TEXT_CHARACTERS));
        put(out, "exception_stack_hash", entry.exception.stackHash());
        put(out, "exception_origin", entry.exception.origin());
        put(out, "input_image_bytes", entry.inputImageBytes);
        put(out, "input_width", entry.inputWidth);
        put(out, "input_height", entry.inputHeight);
        put(out, "input_message_bytes", entry.inputMessageBytes);
        put(out, "native_libjpeg_version", entry.libjpegVersion);
        put(out, "output_stego_bytes", entry.stegoBytes);
        put(out, "output_width", entry.outputWidth);
        put(out, "output_height", entry.outputHeight);
        put(out, "output_components", entry.outputComponents);
        put(out, "java_version", System.getProperty(
                "java.version", "unknown"));
        put(out, "os_name", System.getProperty("os.name", "unknown"));
        put(out, "os_arch", System.getProperty("os.arch", "unknown"));

        for (Map.Entry<String, Object> terminal
                : entry.finalQf.columns().entrySet()) {
            put(out, terminal.getKey(), terminal.getValue());
        }
        addPayload(out, entry.payloadLayout);
        addEvaluation(out, entry);
        return out;
    }

    private static void addPayload(
            LinkedHashMap<String, String> out,
            PayloadPackage.PayloadLayout payload
    ) {
        put(out, "payload_layout_available", payload != null);
        put(out, "raw_message_bytes",
                payload == null ? null : payload.rawMessageBytes());
        put(out, "huffman_candidate_bytes",
                payload == null ? null : payload.huffmanCandidateBytes());
        put(out, "huffman_used",
                payload == null ? null : payload.huffmanUsed());
        put(out, "payload_content_bytes",
                payload == null ? null : payload.contentBytes());
        put(out, "preprocessing_metadata_bytes",
                payload == null ? null : payload.preprocessingMetadataBytes());
        put(out, "prepared_plaintext_bytes",
                payload == null ? null : payload.preparedPlaintextBytes());
        put(out, "payload_protocol_marker_bytes",
                payload == null ? null : payload.protocolMarkerBytes());
        put(out, "payload_qf_field_bytes",
                payload == null ? null : payload.qfFieldBytes());
        put(out, "payload_cipher_length_field_bytes",
                payload == null ? null : payload.ciphertextLengthFieldBytes());
        put(out, "payload_nonce_bytes",
                payload == null ? null : payload.nonceBytes());
        put(out, "payload_authentication_tag_bytes",
                payload == null ? null : payload.authenticationTagBytes());
        put(out, "payload_aad_bytes",
                payload == null ? null : PayloadPackage.GEOMETRY_AAD_LEN);
        put(out, "payload_fixed_header_bytes",
                payload == null ? null : payload.fixedHeaderBytes());
        put(out, "payload_ciphertext_bytes",
                payload == null ? null : payload.ciphertextBytes());
        put(out, "payload_total_bytes",
                payload == null ? null : payload.totalPayloadBytes());
        put(out, "payload_total_bits",
                payload == null ? null : payload.totalPayloadBits());
        put(out, "payload_fixed_framing_overhead_bytes",
                payload == null ? null
                        : payload.fixedFramingOverheadBytes());
        put(out, "payload_net_expansion_bytes",
                payload == null ? null : payload.netPayloadExpansionBytes());
        put(out, "payload_expansion_ratio",
                payload == null ? null : payload.expansionRatio());
        put(out, "payload_content_compression_ratio",
                payload == null ? null : payload.compressionRatio());
    }

    private static void addEvaluation(
            LinkedHashMap<String, String> out,
            Entry entry
    ) {
        EvaluationSummary summary = entry.evaluation;
        boolean available = summary != null
                && summary.jpegMetadata != null
                && summary.coefficientReport != null
                && summary.referenceVsStego != null
                && summary.pngVsStego != null;
        put(out, "final_qf_evaluation_available", available);
        put(out, "final_qf_evaluation_scope",
                "RUN_AND_CB_CR_CHANNEL_AGGREGATES");
        put(out, "final_qf_evaluation_unavailable_reason",
                available ? "" : entry.evaluationUnavailableReason);
        put(out, "embedding_elapsed_ms", entry.embeddingElapsedMillis);
        put(out, "evaluation_elapsed_ms", entry.evaluationElapsedMillis);
        if (!available) {
            addJpegMetadata(out, null);
            addImageComparison(out, "png_stego", null);
            addImageComparison(out, "reference_stego", null);
            addCoefficientMetrics(out, null);
            addChannelMetrics(out, "cb", null);
            addChannelMetrics(out, "cr", null);
            put(out, "computed_zz_position_metric_rows", null);
            put(out, "computed_channel_metric_rows", null);
            put(out, "computed_channel_zz_position_metric_rows", null);
            return;
        }
        addJpegMetadata(out, summary.jpegMetadata);
        addImageComparison(out, "png_stego", summary.pngVsStego);
        addImageComparison(out, "reference_stego", summary.referenceVsStego);
        addCoefficientMetrics(out, summary.coefficientReport);
        addChannelMetrics(out, "cb", findChannel(
                summary.coefficientReport, 1));
        addChannelMetrics(out, "cr", findChannel(
                summary.coefficientReport, 2));
        put(out, "computed_zz_position_metric_rows",
                summary.coefficientReport.positionMetrics.size());
        put(out, "computed_channel_metric_rows",
                summary.coefficientReport.channelMetrics.size());
        put(out, "computed_channel_zz_position_metric_rows",
                summary.coefficientReport.channelPositionMetrics.size());
    }

    private static void addJpegMetadata(
            LinkedHashMap<String, String> out,
            JpegMetadata metadata
    ) {
        put(out, "jpeg_width", metadata == null ? null : metadata.width());
        put(out, "jpeg_height", metadata == null ? null : metadata.height());
        put(out, "jpeg_component_count",
                metadata == null ? null : metadata.componentCount());
        put(out, "jpeg_subsampling",
                metadata == null ? null : metadata.subsampling());
        put(out, "jpeg_max_h_sampling",
                metadata == null ? null : metadata.maxHSamplingFactor());
        put(out, "jpeg_max_v_sampling",
                metadata == null ? null : metadata.maxVSamplingFactor());
        put(out, "jpeg_mcu_columns",
                metadata == null ? null : metadata.mcuColumns());
        put(out, "jpeg_mcu_rows",
                metadata == null ? null : metadata.mcuRows());
        put(out, "jpeg_mcu_count",
                metadata == null ? null : metadata.mcuCount());
        put(out, "jpeg_y_blocks",
                metadata == null ? null : metadata.yBlockCount());
        put(out, "jpeg_cb_blocks",
                metadata == null ? null : metadata.cbBlockCount());
        put(out, "jpeg_cr_blocks",
                metadata == null ? null : metadata.crBlockCount());
        put(out, "jpeg_sampling_factors",
                metadata == null ? null : metadata.samplingFactors());
        put(out, "jpeg_quant_table_indexes",
                metadata == null ? null : metadata.quantTableIndexes());
        put(out, "jpeg_quant_tables_sha256",
                metadata == null ? null
                        : metadata.quantizationTableSha256());
    }

    private static void addImageComparison(
            LinkedHashMap<String, String> out,
            String prefix,
            ImageComparisonMetrics metrics
    ) {
        put(out, prefix + "_mse", metrics == null ? null : metrics.mse);
        put(out, prefix + "_psnr", metrics == null ? null : metrics.psnr);
        put(out, prefix + "_ssim", metrics == null ? null : metrics.ssim);
        put(out, prefix + "_entropy_a",
                metrics == null ? null : metrics.entropyA);
        put(out, prefix + "_entropy_b",
                metrics == null ? null : metrics.entropyB);
        put(out, prefix + "_entropy_difference",
                metrics == null ? null : metrics.entropyDiff);
        put(out, prefix + "_histogram_l1",
                metrics == null ? null : metrics.histogramL1);
        put(out, prefix + "_kl_divergence",
                metrics == null ? null : metrics.klDivergence);
        put(out, prefix + "_file_size_a_bytes",
                metrics == null ? null : metrics.fileSizeA);
        put(out, prefix + "_file_size_b_bytes",
                metrics == null ? null : metrics.fileSizeB);
        put(out, prefix + "_file_size_difference_bytes",
                metrics == null ? null : metrics.fileSizeDiff);
        put(out, prefix + "_file_size_ratio_b_over_a",
                metrics == null ? null : metrics.fileSizeRatio);
        put(out, prefix + "_file_size_change_percent",
                metrics == null || metrics.fileSizeA == 0
                        ? null
                        : 100.0 * metrics.fileSizeDiff / metrics.fileSizeA);
    }

    private static void addCoefficientMetrics(
            LinkedHashMap<String, String> out,
            JpegCoefficientReport c
    ) {
        put(out, "eval_payload_bits", value(c, x -> x.payloadBits));
        put(out, "eval_selected_qf", value(c, x -> x.selectedQF));
        put(out, "eval_candidate_count", value(c, x -> x.candidateCount));
        put(out, "eval_active_carrier_count", value(c, x -> x.activeCarrierCount));
        put(out, "eval_engineering_capacity_bits", value(c, x -> x.conservativeCapacity));
        put(out, "eval_stc_reserve_bits", value(c, x -> x.stcReserveBits));
        put(out, "eval_stc_constraint_height", value(c, x -> x.stcConstraintHeight));
        put(out, "eval_total_coefficients", value(c, x -> x.totalCoefficients));
        put(out, "eval_total_nonzero_ac", value(c, x -> x.totalNonzeroAC));
        put(out, "eval_target_band_cbcr_coefficients", value(c, x -> x.targetBandCbCrCoefficients));
        put(out, "eval_target_band_nonzero_cover", value(c, x -> x.targetBandCbCrNonzero));
        put(out, "eval_target_band_nonzero_stego", value(c, x -> x.targetBandCbCrNonzeroStego));
        put(out, "eval_target_band_zero_cover", value(c, x -> x.targetBandCbCrZero));
        put(out, "eval_target_band_zero_stego", value(c, x -> x.targetBandCbCrZeroStego));
        put(out, "eval_high_tail_cbcr_coefficients", value(c, x -> x.highTailCbCrCoefficients));
        put(out, "eval_high_tail_nonzero_cover", value(c, x -> x.highTailCbCrNonzero));
        put(out, "eval_high_tail_nonzero_stego", value(c, x -> x.highTailCbCrNonzeroStego));
        put(out, "eval_high_tail_zero_cover", value(c, x -> x.highTailCbCrZero));
        put(out, "eval_high_tail_zero_stego", value(c, x -> x.highTailCbCrZeroStego));
        put(out, "eval_high_tail_modified_coefficients", value(c, x -> x.highTailModifiedCoefficients));
        put(out, "eval_modified_coefficients", value(c, x -> x.modifiedCoefficients));
        put(out, "eval_modified_y", value(c, x -> x.modifiedY));
        put(out, "eval_modified_cb", value(c, x -> x.modifiedCb));
        put(out, "eval_modified_cr", value(c, x -> x.modifiedCr));
        put(out, "eval_plus_one_changes", value(c, x -> x.plusOneChanges));
        put(out, "eval_minus_one_changes", value(c, x -> x.minusOneChanges));
        put(out, "eval_other_delta_changes", value(c, x -> x.otherDeltaChanges));
        put(out, "eval_nonzero_to_nonzero", value(c, x -> x.modifiedNonzeroToNonzero));
        put(out, "eval_zero_to_nonzero", value(c, x -> x.modifiedZeroToNonzero));
        put(out, "eval_nonzero_to_zero", value(c, x -> x.modifiedNonzeroToZero));
        put(out, "eval_candidate_loading", doubleValue(c, x -> x.candidateLoading));
        put(out, "eval_former_four_mode_near_half_candidates", value(c, x -> x.formerFourModeNearHalfCandidateCount));
        put(out, "eval_fully_wet_candidates", value(c, x -> x.fullyWetCandidateCount));
        put(out, "eval_overall_bpnzac", doubleValue(c, x -> x.overallBpnzAC));
        put(out, "eval_bits_per_total_coefficient", doubleValue(c, x -> x.bitsPerTotalCoefficient));
        put(out, "eval_candidate_count_over_nonzero_ac", doubleValue(c, x -> x.candidateFractionOfNonzeroAC));
        put(out, "eval_candidate_fraction_all_coefficients", doubleValue(c, x -> x.candidateFractionOfAllCoefficients));
        put(out, "eval_candidate_vs_global_normalization_ratio", doubleValue(c, x -> x.localGlobalPayloadConcentration));
        put(out, "eval_engineering_utilization", doubleValue(c, x -> x.engineeringUtilization));
        put(out, "eval_active_carrier_payload_rate", doubleValue(c, x -> x.activeCarrierPayloadRate));
        put(out, "eval_global_modification_rate", doubleValue(c, x -> x.modificationRateAllCoefficients));
        put(out, "eval_candidate_modification_rate", doubleValue(c, x -> x.modificationRateCandidates));
        put(out, "eval_active_carrier_modification_rate", doubleValue(c, x -> x.modificationRateActiveCarriers));
        put(out, "eval_modified_per_payload_bit", doubleValue(c, x -> x.modifiedPerPayloadBit));
        put(out, "eval_embedding_efficiency_bits_per_modification", doubleValue(c, x -> x.embeddingEfficiencyBitsPerModification));
        put(out, "eval_cb_modification_share", doubleValue(c, x -> x.cbModificationShare));
        put(out, "eval_cr_modification_share", doubleValue(c, x -> x.crModificationShare));
        put(out, "eval_plus_minus_imbalance", doubleValue(c, x -> x.plusMinusImbalance));
        put(out, "eval_mean_absolute_coefficient_delta", doubleValue(c, x -> x.meanAbsoluteCoefficientDelta));
        put(out, "eval_stc_header_message_bits", value(c, x -> x.stcHeaderMessageBits));
        put(out, "eval_stc_header_cover_elements", value(c, x -> x.stcHeaderCoverElements));
        put(out, "eval_stc_header_changed_elements", value(c, x -> x.stcHeaderChangedElements));
        put(out, "eval_stc_header_relative_payload", doubleValue(c, x -> x.stcHeaderRelativePayload));
        put(out, "eval_stc_header_change_rate", doubleValue(c, x -> x.stcHeaderChangeRate));
        put(out, "eval_stc_body_message_bits", value(c, x -> x.stcBodyMessageBits));
        put(out, "eval_stc_body_cover_elements", value(c, x -> x.stcBodyCoverElements));
        put(out, "eval_stc_body_changed_elements", value(c, x -> x.stcBodyChangedElements));
        put(out, "eval_stc_body_relative_payload", doubleValue(c, x -> x.stcBodyRelativePayload));
        put(out, "eval_stc_body_change_rate", doubleValue(c, x -> x.stcBodyChangeRate));
        put(out, "eval_stc_distinct_column_width_limit", value(c, x -> x.stcDistinctColumnWidthLimit));
        put(out, "eval_stc_header_max_submatrix_width", value(c, x -> x.stcHeaderMaximumSubmatrixWidth));
        put(out, "eval_stc_body_max_submatrix_width", value(c, x -> x.stcBodyMaximumSubmatrixWidth));
        put(out, "eval_stc_duplicate_column_risk", booleanValue(c, x -> x.stcDuplicateColumnRisk));
        put(out, "eval_stc_total_changed_elements", value(c, x -> x.stcTotalChangedElements));
        put(out, "eval_stc_weighted_distortion", doubleValue(c, x -> x.stcWeightedDistortion));
        put(out, "eval_stc_relative_payload", doubleValue(c, x -> x.stcRelativePayload));
        put(out, "eval_stc_change_rate", doubleValue(c, x -> x.stcChangeRate));
        put(out, "eval_stc_coefficient_change_mismatch", value(c, x -> x.stcCoefficientChangeMismatch));
        put(out, "eval_dct_smoothed_histogram_l1", doubleValue(c, x -> x.dctHistogramL1));
        put(out, "eval_dct_smoothed_kl", doubleValue(c, x -> x.dctKLDivergence));
        put(out, "eval_dct_global_exact_l1", doubleValue(c, x -> x.dctExactHistogramL1));
        put(out, "eval_dct_global_exact_total_variation", doubleValue(c, x -> x.dctExactTotalVariation));
        put(out, "eval_dct_global_exact_kl", doubleValue(c, x -> x.dctExactKLDivergence));
        put(out, "eval_dct_global_exact_symmetric_kl", doubleValue(c, x -> x.dctExactSymmetricKL));
        put(out, "eval_dct_global_exact_jensen_shannon", doubleValue(c, x -> x.dctExactJensenShannon));
        put(out, "eval_dct_global_exact_hellinger", doubleValue(c, x -> x.dctExactHellinger));
        put(out, "eval_target_band_histogram_l1", doubleValue(c, x -> x.targetBandHistogramL1));
        put(out, "eval_target_band_total_variation", doubleValue(c, x -> x.targetBandTotalVariation));
        put(out, "eval_target_band_kl", doubleValue(c, x -> x.targetBandKLDivergence));
        put(out, "eval_target_band_symmetric_kl", doubleValue(c, x -> x.targetBandSymmetricKL));
        put(out, "eval_target_band_jensen_shannon", doubleValue(c, x -> x.targetBandJensenShannon));
        put(out, "eval_target_band_hellinger", doubleValue(c, x -> x.targetBandHellinger));
        put(out, "eval_high_tail_histogram_l1", doubleValue(c, x -> x.highTailHistogramL1));
        put(out, "eval_high_tail_total_variation", doubleValue(c, x -> x.highTailTotalVariation));
        put(out, "eval_high_tail_kl", doubleValue(c, x -> x.highTailKLDivergence));
        put(out, "eval_high_tail_symmetric_kl", doubleValue(c, x -> x.highTailSymmetricKL));
        put(out, "eval_high_tail_jensen_shannon", doubleValue(c, x -> x.highTailJensenShannon));
        put(out, "eval_high_tail_hellinger", doubleValue(c, x -> x.highTailHellinger));
        put(out, "eval_embedding_success", booleanValue(c, x -> x.embeddingSuccess));
        put(out, "eval_extraction_success", booleanValue(c, x -> x.extractionSuccess));
        put(out, "eval_aes_gcm_verified", booleanValue(c, x -> x.aesGcmVerified));
        put(out, "eval_recovered_message_matches", booleanValue(c, x -> x.recoveredMessageMatches));
    }

    private static void addChannelMetrics(
            LinkedHashMap<String, String> out,
            String prefix,
            ChannelMetrics c
    ) {
        put(out, prefix + "_component_index", c == null ? null : c.componentIndex);
        put(out, prefix + "_channel", c == null ? null : c.channel);
        put(out, prefix + "_total_coefficients", c == null ? null : c.totalCoefficients);
        put(out, prefix + "_cover_nonzero", c == null ? null : c.coverNonzero);
        put(out, prefix + "_stego_nonzero", c == null ? null : c.stegoNonzero);
        put(out, prefix + "_cover_zero", c == null ? null : c.coverZero);
        put(out, prefix + "_stego_zero", c == null ? null : c.stegoZero);
        put(out, prefix + "_eligible_candidates", c == null ? null : c.eligibleCandidates);
        put(out, prefix + "_active_carriers", c == null ? null : c.activeCarriers);
        put(out, prefix + "_modified_coefficients", c == null ? null : c.modifiedCoefficients);
        put(out, prefix + "_plus_one_changes", c == null ? null : c.plusOneChanges);
        put(out, prefix + "_minus_one_changes", c == null ? null : c.minusOneChanges);
        put(out, prefix + "_other_delta_changes", c == null ? null : c.otherDeltaChanges);
        put(out, prefix + "_zero_to_nonzero", c == null ? null : c.zeroToNonzero);
        put(out, prefix + "_nonzero_to_zero", c == null ? null : c.nonzeroToZero);
        put(out, prefix + "_candidate_selection_rate", c == null ? null : c.candidateSelectionRate);
        put(out, prefix + "_modification_rate_eligible", c == null ? null : c.modificationRateEligible);
        put(out, prefix + "_modification_rate_active", c == null ? null : c.modificationRateActive);
        put(out, prefix + "_modification_share_all_changes", c == null ? null : c.modificationShareAllChanges);
        put(out, prefix + "_plus_minus_imbalance", c == null ? null : c.plusMinusImbalance);
        put(out, prefix + "_mean_absolute_delta", c == null ? null : c.meanAbsoluteDelta);
        put(out, prefix + "_histogram_l1", c == null ? null : c.histogramL1);
        put(out, prefix + "_total_variation", c == null ? null : c.totalVariation);
        put(out, prefix + "_kl_cover_to_stego", c == null ? null : c.klCoverToStego);
        put(out, prefix + "_symmetric_kl", c == null ? null : c.symmetricKl);
        put(out, prefix + "_jensen_shannon", c == null ? null : c.jensenShannon);
        put(out, prefix + "_hellinger", c == null ? null : c.hellinger);
    }

    private static ChannelMetrics findChannel(
            JpegCoefficientReport report,
            int component
    ) {
        for (ChannelMetrics channel : report.channelMetrics) {
            if (channel.componentIndex == component) return channel;
        }
        return null;
    }

    private interface IntMetric {
        int get(JpegCoefficientReport report);
    }

    private interface DoubleMetric {
        double get(JpegCoefficientReport report);
    }

    private interface BooleanMetric {
        boolean get(JpegCoefficientReport report);
    }

    private static Integer value(JpegCoefficientReport c, IntMetric metric) {
        return c == null ? null : metric.get(c);
    }

    private static Double doubleValue(
            JpegCoefficientReport c,
            DoubleMetric metric
    ) {
        return c == null ? null : metric.get(c);
    }

    private static Boolean booleanValue(
            JpegCoefficientReport c,
            BooleanMetric metric
    ) {
        return c == null ? null : metric.get(c);
    }

    private static void put(
            LinkedHashMap<String, String> values,
            String key,
            Object value
    ) {
        if (values.containsKey(key)) {
            throw new IllegalStateException(
                    "Duplicate result-audit column: " + key);
        }
        values.put(key, format(value));
    }

    private static String format(Object value) {
        if (value == null) return "";
        if (value instanceof Double number) {
            if (Double.isNaN(number)) return "";
            if (number == Double.POSITIVE_INFINITY) return "Infinity";
            if (number == Double.NEGATIVE_INFINITY) return "-Infinity";
            return String.format(Locale.US, "%.15g", number);
        }
        if (value instanceof Float number) {
            return format(number.doubleValue());
        }
        return compact(String.valueOf(value), MAX_TEXT_CHARACTERS);
    }

    private static Long elapsedMillis(String start, String end) {
        try {
            return Duration.between(
                    Instant.parse(start), Instant.parse(end)).toMillis();
        } catch (RuntimeException invalidTimestamp) {
            return null;
        }
    }

    private static String fileName(Path path) {
        if (path == null) return "";
        Path name = path.getFileName();
        return name == null ? path.toString() : name.toString();
    }

    private static String fileStem(Path path) {
        String name = fileName(path);
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    private static String absolutePath(Path path) {
        return path == null ? "" : path.toAbsolutePath().normalize().toString();
    }

    private static String required(String value, String label) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(label + " cannot be blank");
        }
        return value;
    }

    private static String safe(String value) {
        return value == null ? "" : value;
    }

    private static String compact(String value, int maximum) {
        String normalized = safe(value).replaceAll("[\\r\\n]+", " ");
        if (normalized.length() <= maximum) return normalized;
        return normalized.substring(0, maximum - 3) + "...";
    }
}
