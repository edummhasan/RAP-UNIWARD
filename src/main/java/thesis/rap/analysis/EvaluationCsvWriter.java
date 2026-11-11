package thesis.rap.analysis;

import thesis.rap.payload.PayloadPackage;
import thesis.rap.stc.OriginalStcCodec;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Writes schema-versioned run, channel, and per-ZZ evaluation datasets. */
public final class EvaluationCsvWriter {
    public static final String SCHEMA_VERSION = "1.0";

    private EvaluationCsvWriter() {}

    public static void append(
            Path runCsv,
            Path positionCsv,
            Path channelCsv,
            Path channelPositionCsv,
            EvaluationRunContext ctx,
            EvaluationSummary s
    ) throws IOException {
        appendRun(runCsv, ctx, s);

        List<List<String>> positionRows = new ArrayList<>();
        for (DctPositionMetrics p : s.coefficientReport.positionMetrics) positionRows.add(positionRow(ctx, s, p));
        CsvUtil.appendRows(positionCsv, positionHeader(), positionRows);

        List<List<String>> channelRows = new ArrayList<>();
        for (ChannelMetrics c : s.coefficientReport.channelMetrics) channelRows.add(channelRow(ctx, s, c));
        CsvUtil.appendRows(channelCsv, channelHeader(), channelRows);

        List<List<String>> channelPositionRows = new ArrayList<>();
        for (DctChannelPositionMetrics p : s.coefficientReport.channelPositionMetrics) {
            channelPositionRows.add(channelPositionRow(ctx, s, p));
        }
        CsvUtil.appendRows(channelPositionCsv, channelPositionHeader(), channelPositionRows);
    }

    /** Appends only the compact per-run table used by THESIS and CHANGES profiles. */
    public static void appendRun(
            Path runCsv,
            EvaluationRunContext ctx,
            EvaluationSummary summary
    ) throws IOException {
        CsvUtil.append(
                runCsv, runHeader(), runRow(ctx, summary));
    }

    private static List<String> runHeader() {
        return List.of(
                "schema_version","run_id","message_group","message_number","message_name",
                "image","message","geometry_mode","sampling_geometry",
                "qf_min","qf_max","selection_policy",
                "max_candidate_loading_gate","max_engineering_utilization_gate","selected_qf",
                "qfs_tested","failed_qfs_before_selection","previous_tested_qf","selected_candidate_count",
                "candidate_headroom_bits","engineering_capacity_bits","capacity_headroom_bits",
                "candidate_headroom_percent_payload","capacity_headroom_percent_payload",
                "timestamp_utc","application_version","input_image_sha256","message_sha256","reference_jpeg_sha256","stego_jpeg_sha256",
                "java_version","os_name","os_arch","libjpeg_version","reference_encoder",

                "jpeg_width","jpeg_height","jpeg_component_count","jpeg_subsampling","jpeg_max_h_sampling","jpeg_max_v_sampling",
                "jpeg_mcu_columns","jpeg_mcu_rows","jpeg_mcu_count","jpeg_y_blocks","jpeg_cb_blocks","jpeg_cr_blocks",
                "jpeg_sampling_factors","jpeg_quant_table_indexes","jpeg_quant_tables_sha256",

                "raw_message_bytes","huffman_candidate_bytes","huffman_used","payload_content_bytes","preprocessing_metadata_bytes",
                "prepared_plaintext_bytes","payload_protocol_marker_bytes","payload_qf_field_bytes","payload_cipher_length_field_bytes",
                "payload_nonce_bytes","payload_auth_tag_bytes","payload_aad_bytes","payload_fixed_header_bytes","payload_ciphertext_bytes",
                "payload_bytes","payload_bits","payload_fixed_framing_overhead_bytes","payload_net_expansion_bytes","payload_expansion_ratio","payload_content_compression_ratio",

                "candidate_count","former_four_mode_near_half_candidate_count","fully_wet_candidate_count","actual_stc_cover_elements","stc_fixed_header_reserve","stc_constraint_height","stc_numeric_profile",
                "engineering_capacity","engineering_utilization","true_stc_relative_payload",
                "stc_header_message_bits","stc_header_cover_elements","stc_header_relative_payload","stc_header_changed_elements","stc_header_change_rate",
                "stc_body_message_bits","stc_body_cover_elements","stc_body_relative_payload","stc_body_changed_elements","stc_body_change_rate",
                "stc_distinct_column_width_limit","stc_header_max_submatrix_width","stc_body_max_submatrix_width","stc_duplicate_column_risk",
                "stc_total_changed_elements","stc_weighted_distortion","stc_change_rate","stc_coefficient_change_mismatch",

                "total_coefficients","total_nonzero_ac","target_band_cbcr_coefficients","target_band_cbcr_nonzero_cover",
                "target_band_cbcr_nonzero_stego","target_band_cbcr_zero_cover","target_band_cbcr_zero_stego",
                "high_tail_cbcr_coefficients","high_tail_nonzero_cover","high_tail_nonzero_stego","high_tail_zero_cover","high_tail_zero_stego","high_tail_modified_coefficients",
                "candidate_loading","overall_bpnzac","bits_per_total_coefficient","eligible_address_count_over_global_nonzero_ac",
                "candidate_fraction_all_coefficients","candidate_vs_global_normalization_ratio",

                "modified_coefficients","modified_y","modified_cb","modified_cr","plus_one_changes","minus_one_changes","other_delta_changes",
                "nonzero_to_nonzero","zero_to_nonzero","nonzero_to_zero","global_modification_rate","candidate_modification_rate",
                "active_carrier_modification_rate","modified_per_payload_bit","embedding_efficiency_bits_per_modification",
                "cb_modification_share","cr_modification_share","plus_minus_imbalance","mean_abs_coefficient_delta",

                "ref_stego_mse","ref_stego_psnr","ref_stego_ssim","ref_stego_entropy_diff","ref_stego_hist_l1","ref_stego_kl",
                "ref_jpeg_bytes","stego_jpeg_bytes","jpeg_size_diff_bytes","jpeg_size_ratio","jpeg_size_change_percent",
                "png_stego_mse_informational","png_stego_psnr_informational","png_stego_ssim_informational",
                "source_png_bytes","png_to_stego_size_ratio_informational",

                "dct_smoothed_l1","dct_smoothed_kl","dct_global_l1","dct_global_tv","dct_global_kl","dct_global_symmetric_kl","dct_global_js","dct_global_hellinger",
                "target_band_l1","target_band_tv","target_band_kl","target_band_symmetric_kl","target_band_js","target_band_hellinger",
                "high_tail_l1","high_tail_tv","high_tail_kl","high_tail_symmetric_kl","high_tail_js","high_tail_hellinger",
                "embedding_success","extraction_success","aes_gcm_verified","recovered_message_matches",
                "embedding_elapsed_ms","evaluation_elapsed_ms"
        );
    }

    private static List<String> runRow(EvaluationRunContext ctx, EvaluationSummary s) {
        JpegCoefficientReport c = s.coefficientReport;
        ImageComparisonMetrics q = s.referenceVsStego;
        ImageComparisonMetrics p = s.pngVsStego;
        PayloadPackage.PayloadLayout pl = ctx.payloadLayout();
        JpegMetadata jm = ctx.jpegMetadata();
        QfSelectionMetrics qm = ctx.qfSelectionMetrics();
        RunReproducibility rr = ctx.reproducibility();
        double jpegSizeChangePercent = q.fileSizeA == 0 ? Double.NaN : 100.0 * q.fileSizeDiff / q.fileSizeA;

        List<String> r = new ArrayList<>();
        MessageGroup group = MessageGroup.from(ctx.messageName());
        add(r,
                SCHEMA_VERSION, ctx.runId(), group.label(), group.number(), group.name(),
                ctx.imageName(), ctx.messageName(), ctx.geometryMode(),
                ctx.samplingGeometry(), ctx.qfMin(), ctx.qfMax(),
                ctx.selectionPolicy(),
                ctx.maxCandidateLoading(), ctx.maxStcUtilization(), c.selectedQF,
                qm.qfsTested(), qm.failedQfsBeforeSelection(), qm.previousTestedQf(), qm.selectedCandidateCount(),
                qm.candidateHeadroomBits(), qm.engineeringCapacityBits(), qm.capacityHeadroomBits(),
                qm.candidateHeadroomPercentOfPayload(), qm.capacityHeadroomPercentOfPayload(),
                rr.timestampUtc(), rr.applicationVersion(), rr.inputImageSha256(), rr.messageSha256(), rr.referenceJpegSha256(), rr.stegoJpegSha256(),
                rr.javaVersion(), rr.osName(), rr.osArch(), rr.libjpegVersion(), rr.referenceEncoder(),

                jm.width(), jm.height(), jm.componentCount(), jm.subsampling(), jm.maxHSamplingFactor(), jm.maxVSamplingFactor(),
                jm.mcuColumns(), jm.mcuRows(), jm.mcuCount(), jm.yBlockCount(), jm.cbBlockCount(), jm.crBlockCount(),
                jm.samplingFactors(), jm.quantTableIndexes(), jm.quantizationTableSha256(),

                pl.rawMessageBytes(), pl.huffmanCandidateBytes(), pl.huffmanUsed(), pl.contentBytes(), pl.preprocessingMetadataBytes(),
                pl.preparedPlaintextBytes(), pl.protocolMarkerBytes(), pl.qfFieldBytes(), pl.ciphertextLengthFieldBytes(),
                pl.nonceBytes(), pl.authenticationTagBytes(), PayloadPackage.GEOMETRY_AAD_LEN, pl.fixedHeaderBytes(), pl.ciphertextBytes(),
                pl.totalPayloadBytes(), pl.totalPayloadBits(),
                pl.fixedFramingOverheadBytes(),
                pl.netPayloadExpansionBytes(),
                pl.expansionRatio(), pl.compressionRatio(),

                c.candidateCount,
                c.formerFourModeNearHalfCandidateCount,
                c.fullyWetCandidateCount,
                c.activeCarrierCount, c.stcReserveBits, c.stcConstraintHeight,
                OriginalStcCodec.NUMERIC_PROFILE,
                c.conservativeCapacity, c.engineeringUtilization, c.activeCarrierPayloadRate,
                c.stcHeaderMessageBits, c.stcHeaderCoverElements, c.stcHeaderRelativePayload, c.stcHeaderChangedElements, c.stcHeaderChangeRate,
                c.stcBodyMessageBits, c.stcBodyCoverElements, c.stcBodyRelativePayload, c.stcBodyChangedElements, c.stcBodyChangeRate,
                c.stcDistinctColumnWidthLimit, c.stcHeaderMaximumSubmatrixWidth, c.stcBodyMaximumSubmatrixWidth, c.stcDuplicateColumnRisk,
                c.stcTotalChangedElements, c.stcWeightedDistortion, c.stcChangeRate, c.stcCoefficientChangeMismatch,

                c.totalCoefficients, c.totalNonzeroAC, c.targetBandCbCrCoefficients, c.targetBandCbCrNonzero,
                c.targetBandCbCrNonzeroStego, c.targetBandCbCrZero, c.targetBandCbCrZeroStego,
                c.highTailCbCrCoefficients, c.highTailCbCrNonzero, c.highTailCbCrNonzeroStego, c.highTailCbCrZero,
                c.highTailCbCrZeroStego, c.highTailModifiedCoefficients,
                c.candidateLoading, c.overallBpnzAC, c.bitsPerTotalCoefficient, c.candidateFractionOfNonzeroAC,
                c.candidateFractionOfAllCoefficients, c.localGlobalPayloadConcentration,

                c.modifiedCoefficients, c.modifiedY, c.modifiedCb, c.modifiedCr, c.plusOneChanges, c.minusOneChanges, c.otherDeltaChanges,
                c.modifiedNonzeroToNonzero, c.modifiedZeroToNonzero, c.modifiedNonzeroToZero, c.modificationRateAllCoefficients,
                c.modificationRateCandidates, c.modificationRateActiveCarriers, c.modifiedPerPayloadBit,
                c.embeddingEfficiencyBitsPerModification, c.cbModificationShare, c.crModificationShare,
                c.plusMinusImbalance, c.meanAbsoluteCoefficientDelta,

                q.mse, q.psnr, q.ssim, q.entropyDiff, q.histogramL1, q.klDivergence,
                q.fileSizeA, q.fileSizeB, q.fileSizeDiff, q.fileSizeRatio, jpegSizeChangePercent,
                p.mse, p.psnr, p.ssim, p.fileSizeA, p.fileSizeRatio,

                c.dctHistogramL1, c.dctKLDivergence, c.dctExactHistogramL1, c.dctExactTotalVariation,
                c.dctExactKLDivergence, c.dctExactSymmetricKL, c.dctExactJensenShannon, c.dctExactHellinger,
                c.targetBandHistogramL1, c.targetBandTotalVariation, c.targetBandKLDivergence, c.targetBandSymmetricKL,
                c.targetBandJensenShannon, c.targetBandHellinger,
                c.highTailHistogramL1, c.highTailTotalVariation, c.highTailKLDivergence, c.highTailSymmetricKL,
                c.highTailJensenShannon, c.highTailHellinger,
                c.embeddingSuccess, c.extractionSuccess, c.aesGcmVerified, c.recoveredMessageMatches,
                ctx.embeddingElapsedMillis(), ctx.evaluationElapsedMillis()
        );
        return r;
    }

    private static List<String> positionHeader() {
        return List.of(
                "schema_version","run_id","image","message",
                "geometry_mode","sampling_geometry","selected_qf","payload_bits","zz_rank","thesis_zz_rank_one_based","u","v","natural_index",
                "total_coefficients","cover_nonzero","stego_nonzero","eligible_candidates","active_carriers","modified_coefficients",
                "candidate_selection_rate","modification_rate_eligible","modification_rate_active","modification_share_all_changes",
                "plus_one_changes","minus_one_changes","other_delta_changes","plus_minus_imbalance","zero_to_nonzero","nonzero_to_zero",
                "mean_abs_delta","hist_l1","total_variation","kl_cover_to_stego","symmetric_kl","jensen_shannon","hellinger",
                "horizontal_corr_cover","horizontal_corr_stego","horizontal_corr_delta",
                "vertical_corr_cover","vertical_corr_stego","vertical_corr_delta"
        );
    }

    private static List<String> positionRow(EvaluationRunContext ctx, EvaluationSummary s, DctPositionMetrics p) {
        List<String> r = new ArrayList<>();
        add(r, SCHEMA_VERSION, ctx.runId(), ctx.imageName(), ctx.messageName(),
                ctx.geometryMode(), ctx.samplingGeometry(),
                s.coefficientReport.selectedQF,
                s.coefficientReport.payloadBits, p.zzRank, p.zzRank + 1, p.u, p.v, p.naturalIndex, p.totalCoefficients,
                p.coverNonzero, p.stegoNonzero, p.eligibleCandidates, p.activeCarriers, p.modifiedCoefficients,
                p.candidateSelectionRate, p.modificationRateEligible, p.modificationRateActive,
                p.modificationShareOfAllChanges, p.plusOneChanges, p.minusOneChanges, p.otherDeltaChanges,
                p.plusMinusImbalance, p.zeroToNonzero, p.nonzeroToZero, p.meanAbsoluteDelta,
                p.histogramL1, p.totalVariation, p.klCoverToStego, p.symmetricKl, p.jensenShannon, p.hellinger,
                p.horizontalCorrelationCover, p.horizontalCorrelationStego, p.horizontalCorrelationDelta,
                p.verticalCorrelationCover, p.verticalCorrelationStego, p.verticalCorrelationDelta);
        return r;
    }

    private static List<String> channelHeader() {
        return List.of(
                "schema_version","run_id","image","message",
                "geometry_mode","sampling_geometry","selected_qf","payload_bits","component_index","channel",
                "total_coefficients","cover_nonzero","stego_nonzero","cover_zero","stego_zero","eligible_candidates","active_carriers","modified_coefficients",
                "candidate_selection_rate","modification_rate_eligible","modification_rate_active","modification_share_all_changes",
                "plus_one_changes","minus_one_changes","other_delta_changes","plus_minus_imbalance","zero_to_nonzero","nonzero_to_zero",
                "mean_abs_delta","hist_l1","total_variation","kl_cover_to_stego","symmetric_kl","jensen_shannon","hellinger"
        );
    }

    private static List<String> channelRow(EvaluationRunContext ctx, EvaluationSummary s, ChannelMetrics c) {
        List<String> r = new ArrayList<>();
        add(r, SCHEMA_VERSION, ctx.runId(), ctx.imageName(), ctx.messageName(),
                ctx.geometryMode(), ctx.samplingGeometry(),
                s.coefficientReport.selectedQF,
                s.coefficientReport.payloadBits, c.componentIndex, c.channel, c.totalCoefficients, c.coverNonzero, c.stegoNonzero,
                c.coverZero, c.stegoZero, c.eligibleCandidates, c.activeCarriers, c.modifiedCoefficients,
                c.candidateSelectionRate, c.modificationRateEligible, c.modificationRateActive, c.modificationShareAllChanges,
                c.plusOneChanges, c.minusOneChanges, c.otherDeltaChanges, c.plusMinusImbalance,
                c.zeroToNonzero, c.nonzeroToZero, c.meanAbsoluteDelta, c.histogramL1, c.totalVariation,
                c.klCoverToStego, c.symmetricKl, c.jensenShannon, c.hellinger);
        return r;
    }

    private static List<String> channelPositionHeader() {
        return List.of(
                "schema_version","run_id","image","message",
                "geometry_mode","sampling_geometry","selected_qf","payload_bits","component_index","channel",
                "zz_rank","thesis_zz_rank_one_based","u","v","natural_index","total_coefficients","cover_nonzero","stego_nonzero",
                "eligible_candidates","active_carriers","modified_coefficients","candidate_selection_rate","modification_rate_eligible",
                "modification_rate_active","modification_share_all_changes","plus_one_changes","minus_one_changes","other_delta_changes",
                "plus_minus_imbalance","zero_to_nonzero","nonzero_to_zero","mean_abs_delta","hist_l1","total_variation",
                "kl_cover_to_stego","symmetric_kl","jensen_shannon","hellinger","horizontal_corr_cover","horizontal_corr_stego",
                "horizontal_corr_delta","vertical_corr_cover","vertical_corr_stego","vertical_corr_delta"
        );
    }

    private static List<String> channelPositionRow(EvaluationRunContext ctx, EvaluationSummary s, DctChannelPositionMetrics p) {
        List<String> r = new ArrayList<>();
        add(r, SCHEMA_VERSION, ctx.runId(), ctx.imageName(), ctx.messageName(),
                ctx.geometryMode(), ctx.samplingGeometry(),
                s.coefficientReport.selectedQF,
                s.coefficientReport.payloadBits, p.componentIndex, p.channel, p.zzRank, p.thesisZzRankOneBased,
                p.u, p.v, p.naturalIndex, p.totalCoefficients, p.coverNonzero, p.stegoNonzero,
                p.eligibleCandidates, p.activeCarriers, p.modifiedCoefficients, p.candidateSelectionRate,
                p.modificationRateEligible, p.modificationRateActive, p.modificationShareAllChanges,
                p.plusOneChanges, p.minusOneChanges, p.otherDeltaChanges, p.plusMinusImbalance,
                p.zeroToNonzero, p.nonzeroToZero, p.meanAbsoluteDelta, p.histogramL1, p.totalVariation,
                p.klCoverToStego, p.symmetricKl, p.jensenShannon, p.hellinger,
                p.horizontalCorrelationCover, p.horizontalCorrelationStego, p.horizontalCorrelationDelta,
                p.verticalCorrelationCover, p.verticalCorrelationStego, p.verticalCorrelationDelta);
        return r;
    }

    private static void add(List<String> out, Object... values) {
        for (Object value : values) {
            if (value == null) out.add("");
            else if (value instanceof Double d) out.add(fmt(d));
            else if (value instanceof Float f) out.add(fmt(f.doubleValue()));
            else out.add(String.valueOf(value));
        }
    }

    private static String fmt(double value) {
        if (!Double.isFinite(value)) return "";
        return String.format(Locale.US, "%.15g", value);
    }
}
