package thesis.rap.app;

import thesis.rap.alg.StatisticalCostSummary;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Bounded in-memory snapshot of only the most recently tested quality factor.
 *
 * <p>This is the RESULT_ONLY evidence model: every new QF replaces the previous
 * snapshot, so the application never accumulates the complete QF or balance
 * trail.  A successful search leaves the selected verified QF in the snapshot;
 * an exhausted search leaves the last tested QF and its last available state.</p>
 */
final class FinalQfAuditState {
    private int qfsTested;
    private Integer qf;
    private String geometryMode = "";
    private String samplingGeometry = "";
    private String role = "NOT_REACHED";
    private String stage = "NOT_STARTED";
    private String terminalStatus = "NOT_RUN";
    private String terminalReason = "";

    private Integer requiredPayloadBits;
    private Boolean referenceBuilt;
    private Boolean candidateStatisticsAvailable;
    private Boolean kernelCompleted;
    private Boolean coefficientWritebackVerified;
    private Boolean authenticatedRoundTripAttempted;
    private Boolean extractionSuccess;
    private Boolean aesGcmVerified;
    private Boolean exactMessageVerified;
    private String verificationDetail = "";
    private Integer configuredStcHeight;

    private Integer targetTotal;
    private Integer targetNonzero;
    private Integer targetZero;
    private Double targetNonzeroRatio;
    private Integer highTailTotal;
    private Integer highTailNonzero;
    private Integer highTailZero;
    private Double highTailNonzeroRatio;
    private Integer universeCount;
    private Integer dryCount;
    private Integer actualCandidateCount;
    private Double candidateLoading;
    private Integer formerFourModeNearHalfCandidates;
    private Integer fullyWetCandidates;
    private Integer negativeStateCount;
    private Integer zeroStateCount;
    private Integer positiveStateCount;
    private Integer dryPlusDirections;
    private Integer dryMinusDirections;
    private Integer wetPlusDirections;
    private Integer wetMinusDirections;
    private Integer engineeringCapacityBits;
    private Integer capacityHeadroomBits;
    private Double engineeringUtilization;
    private Boolean capacityPass;
    private StatisticalCostSummary statisticalCostSummary;
    private ChannelAllocationEstimate channelAllocation;

    private String balanceProfile = "";
    private Integer balanceMaximumAttempts;
    private Integer balanceAttemptCount;
    private Integer finalBalanceIteration;
    private Double finalBalanceScale;
    private Double finalBalanceMultiplier;
    private Double finalBalanceStep;
    private Boolean finalHeaderStcSuccess;
    private Boolean finalBodyStcSuccess;
    private Long finalZeroToPlus;
    private Long finalZeroToMinus;
    private Long finalPlusToZero;
    private Long finalMinusToZero;
    private Long finalZeroActivations;
    private Long finalZeroDeactivations;
    private Long finalZeroBinDelta;
    private Boolean finalSyndromeVerified;
    private Boolean finalBalanceAccepted;
    private String finalBalanceStatus = "";
    private String finalBalanceReason = "";

    private Integer stcReserveBits;
    private Integer stcHeaderMessageBits;
    private Integer stcHeaderCarrierCount;
    private Integer stcBodyMessageBits;
    private Integer stcBodyCarrierCount;
    private Integer stcTotalMessageBits;
    private Integer stcTotalCarrierCount;
    private Integer stcCapacityHeadroomBits;
    private Double stcOverallRelativePayload;
    private Integer stcHeaderChangedElements;
    private Double stcHeaderWeightedDistortion;
    private Double stcHeaderChangeRate;
    private Integer stcBodyChangedElements;
    private Double stcBodyWeightedDistortion;
    private Double stcBodyChangeRate;
    private Integer stcTotalChangedElements;
    private Double stcTotalWeightedDistortion;
    private Double stcOverallChangeRate;
    private Double stcEmbeddingEfficiencyBitsPerChange;
    private Integer stcBodyWidthCap;
    private Long stcNativePathBytesEstimate;
    private Long stcNativePeakPathBytesEstimate;
    private Long stcNativeMetricBytesEstimate;

    void configureStcHeight(int height) {
        StcCarrierPlan.validateSupportedHeight(height);
        configuredStcHeight = height;
    }

    void begin(
            int testedQf,
            int payloadBits,
            ThreeStateEmbeddingEngine.Profile profile
    ) {
        qfsTested++;
        resetPerQf();
        qf = testedQf;
        role = "LAST_TESTED";
        stage = "QF_STARTED";
        terminalStatus = "IN_PROGRESS";
        requiredPayloadBits = payloadBits;
        referenceBuilt = false;
        candidateStatisticsAvailable = false;
        kernelCompleted = false;
        coefficientWritebackVerified = false;
        authenticatedRoundTripAttempted = false;
        extractionSuccess = false;
        aesGcmVerified = false;
        exactMessageVerified = false;
        if (profile != null) {
            balanceProfile = profile.code();
            balanceMaximumAttempts = profile.maxAttempts();
        }
    }

    void markReferenceBuilt() {
        referenceBuilt = true;
        stage = "REFERENCE_BUILT";
    }

    void captureCandidateRow(QfBandCoefficientReport.Row row) {
        if (row == null) return;
        geometryMode = safe(row.geometryMode);
        samplingGeometry = safe(row.samplingGeometry);
        candidateStatisticsAvailable = true;
        stage = "CANDIDATES_ANALYZED";
        requiredPayloadBits = row.requiredPayloadBits;
        targetTotal = row.targetTotal;
        targetNonzero = row.targetNonzero;
        targetZero = row.targetZero;
        targetNonzeroRatio = row.targetNonzeroRatio();
        highTailTotal = row.highTailTotal;
        highTailNonzero = row.highTailNonzero;
        highTailZero = row.highTailZero;
        highTailNonzeroRatio = row.highTailNonzeroRatio();
        universeCount = row.universeCount;
        dryCount = row.dryCount;
        actualCandidateCount = row.actualCandidateCount;
        candidateLoading = row.candidateLoading;
        formerFourModeNearHalfCandidates =
                row.formerFourModeNearHalfCandidates;
        fullyWetCandidates = row.fullyWetCandidates;
        negativeStateCount = row.negativeStateCount;
        zeroStateCount = row.zeroStateCount;
        positiveStateCount = row.positiveStateCount;
        dryPlusDirections = row.dryPlusDirections;
        dryMinusDirections = row.dryMinusDirections;
        wetPlusDirections = row.wetPlusDirections;
        wetMinusDirections = row.wetMinusDirections;
        engineeringCapacityBits = row.targetEngineeringCapacity;
        capacityHeadroomBits = row.targetEngineeringCapacity
                - row.requiredPayloadBits;
        engineeringUtilization = row.engineeringUtilization;
        capacityPass = row.capacityPass;
        balanceAttemptCount = row.balanceAttemptCount;
        statisticalCostSummary = row.statisticalCostSummary;
        channelAllocation = row.channelAllocation;
    }

    void captureBalanceAttempts(
            List<ThreeStateEmbeddingEngine.Attempt> attempts
    ) {
        if (attempts == null) return;
        balanceAttemptCount = attempts.size();
        if (attempts.isEmpty()) return;
        ThreeStateEmbeddingEngine.Attempt last =
                attempts.get(attempts.size() - 1);
        finalBalanceIteration = last.iteration();
        finalBalanceScale = last.scale();
        finalBalanceMultiplier = last.multiplier();
        finalBalanceStep = last.step();
        finalHeaderStcSuccess = last.headerStcSuccess();
        finalBodyStcSuccess = last.bodyStcSuccess();
        ThreeStateEmbeddingEngine.TransitionCounts transitions =
                last.transitions();
        if (transitions != null) {
            finalZeroToPlus = transitions.zeroToPlus();
            finalZeroToMinus = transitions.zeroToMinus();
            finalPlusToZero = transitions.plusToZero();
            finalMinusToZero = transitions.minusToZero();
            finalZeroActivations = transitions.activations();
            finalZeroDeactivations = transitions.deactivations();
        }
        finalZeroBinDelta = last.zeroBinDelta();
        finalSyndromeVerified = last.syndromeVerified();
        finalBalanceAccepted = last.accepted();
        finalBalanceStatus = safe(last.status());
        finalBalanceReason = safe(last.reason());
        stage = last.accepted()
                ? "BALANCE_ACCEPTED" : "BALANCE_REJECTED";
    }

    void captureKernel(FixedQfEmbeddingKernel.Result kernel) {
        if (kernel == null) return;
        kernelCompleted = true;
        stage = "KERNEL_COMPLETED";
        ThreeStateEmbeddingEngine.Result balance = kernel.balance();
        balanceProfile = balance.profile().code();
        balanceMaximumAttempts = balance.profile().maxAttempts();
        captureBalanceAttempts(balance.attempts());
        finalBalanceScale = balance.scale();
        finalBalanceMultiplier = balance.finalMultiplier();
        finalZeroBinDelta = balance.zeroBinDelta();

        StcCarrierPlan plan = kernel.carrierPlan();
        if (configuredStcHeight != null
                && configuredStcHeight != plan.constraintHeight()) {
            throw new IllegalStateException(
                    "Configured and kernel STC heights differ");
        }
        configuredStcHeight = plan.constraintHeight();
        stcBodyWidthCap = plan.bodyWidthCap();
        stcNativePathBytesEstimate = plan.nativePathBytesEstimate();
        stcNativePeakPathBytesEstimate = plan.nativePeakPathBytesEstimate();
        stcNativeMetricBytesEstimate = plan.nativeMetricBytesEstimate();
        channelAllocation = kernel.channelAllocation();
        stcReserveBits = plan.reserveBits();
        stcHeaderMessageBits = plan.headerMessageBits();
        stcHeaderCarrierCount = plan.headerCarrierCount();
        stcBodyMessageBits = plan.bodyMessageBits();
        stcBodyCarrierCount = plan.bodyCarrierCount();
        stcTotalMessageBits = plan.totalMessageBits();
        stcTotalCarrierCount = plan.totalCarrierCount();
        engineeringCapacityBits = plan.engineeringMessageCapacityBits();
        capacityHeadroomBits = plan.capacityHeadroomBits();
        stcCapacityHeadroomBits = plan.capacityHeadroomBits();
        stcOverallRelativePayload = plan.overallRelativePayload();

        StcEmbeddingTrace header = balance.header().trace();
        StcEmbeddingTrace body = balance.body().trace();
        stcHeaderChangedElements = header.changedElementCount();
        stcHeaderWeightedDistortion = header.weightedDistortion();
        stcHeaderChangeRate = header.changeRate();
        stcBodyChangedElements = body.changedElementCount();
        stcBodyWeightedDistortion = body.weightedDistortion();
        stcBodyChangeRate = body.changeRate();
        stcTotalChangedElements = header.changedElementCount()
                + body.changedElementCount();
        stcTotalWeightedDistortion = header.weightedDistortion()
                + body.weightedDistortion();
        stcOverallChangeRate = plan.totalCarrierCount() == 0
                ? Double.NaN
                : stcTotalChangedElements
                / (double) plan.totalCarrierCount();
        stcEmbeddingEfficiencyBitsPerChange = stcTotalChangedElements == 0
                ? Double.POSITIVE_INFINITY
                : plan.totalMessageBits()
                / (double) stcTotalChangedElements;
    }

    void markCoefficientWritebackVerified() {
        coefficientWritebackVerified = true;
        stage = "COEFFICIENT_WRITEBACK_VERIFIED";
    }

    void captureVerification(
            boolean extraction,
            boolean aesGcm,
            boolean exactMessage,
            String detail
    ) {
        authenticatedRoundTripAttempted = true;
        extractionSuccess = extraction;
        aesGcmVerified = aesGcm;
        exactMessageVerified = exactMessage;
        verificationDetail = safe(detail);
        stage = extraction && aesGcm && exactMessage
                ? "AUTHENTICATED_ROUND_TRIP_VERIFIED"
                : "AUTHENTICATED_ROUND_TRIP_REJECTED";
    }

    void finish(String status, String reason, boolean selected) {
        terminalStatus = safe(status);
        terminalReason = safe(reason);
        role = selected ? "SELECTED_VERIFIED" : "LAST_TESTED";
        stage = selected ? "QF_VERIFIED" : terminalStatus;
    }

    void markRunFailure(String status, String reason) {
        if (qf == null) {
            role = "NOT_REACHED";
            stage = "PRE_QF_FAILURE";
        } else if ("IN_PROGRESS".equals(terminalStatus)) {
            stage = "QF_ABORTED";
        }
        if (qf == null || "IN_PROGRESS".equals(terminalStatus)) {
            terminalStatus = safe(status);
            terminalReason = safe(reason);
        }
    }

    Integer qf() {
        return qf;
    }

    int qfsTested() {
        return qfsTested;
    }

    boolean writebackVerified() {
        return Boolean.TRUE.equals(coefficientWritebackVerified);
    }

    boolean nativeStcVerified() {
        return Boolean.TRUE.equals(finalSyndromeVerified)
                && Boolean.TRUE.equals(finalBalanceAccepted);
    }

    boolean extractionVerified() {
        return Boolean.TRUE.equals(extractionSuccess);
    }

    boolean aesGcmVerified() {
        return Boolean.TRUE.equals(aesGcmVerified);
    }

    boolean exactMessageVerified() {
        return Boolean.TRUE.equals(exactMessageVerified);
    }

    String consoleSummary() {
        return String.format(
                java.util.Locale.US,
                "FINAL_QF_STATE qfs_tested=%d role=%s qf=%s "
                        + "geometry_mode=%s sampling_geometry=%s stage=%s "
                        + "status=%s payload_bits=%s dry_candidates=%s "
                        + "stc_height=%s "
                        + "engineering_capacity_bits=%s capacity_pass=%s "
                        + "balance_attempts=%s final_delta_h0=%s "
                        + "writeback_verified=%s extraction_success=%s "
                        + "aes_gcm_verified=%s exact_message_verified=%s "
                        + "reason=%s",
                qfsTested, role, text(qf),
                geometryMode.isBlank() ? "none" : geometryMode,
                samplingGeometry.isBlank() ? "none" : samplingGeometry,
                stage, terminalStatus,
                text(requiredPayloadBits), text(dryCount),
                text(configuredStcHeight),
                text(engineeringCapacityBits), text(capacityPass),
                text(balanceAttemptCount), text(finalZeroBinDelta),
                text(coefficientWritebackVerified), text(extractionSuccess),
                text(aesGcmVerified), text(exactMessageVerified),
                terminalReason.isBlank() ? "none" : terminalReason);
    }

    Map<String, Object> columns() {
        LinkedHashMap<String, Object> values = new LinkedHashMap<>();
        values.put("qfs_tested", qfsTested);
        values.put("terminal_qf_role", role);
        values.put("terminal_qf", qf);
        values.put("terminal_qf_geometry_mode", geometryMode);
        values.put("terminal_qf_sampling_geometry", samplingGeometry);
        values.put("terminal_qf_stage", stage);
        values.put("terminal_qf_status", terminalStatus);
        values.put("terminal_qf_reason", terminalReason);
        values.put("terminal_qf_required_payload_bits", requiredPayloadBits);
        values.put("terminal_qf_reference_built", referenceBuilt);
        values.put("terminal_qf_candidate_statistics_available",
                candidateStatisticsAvailable);
        values.put("terminal_qf_kernel_completed", kernelCompleted);
        values.put("terminal_qf_coefficient_writeback_verified",
                coefficientWritebackVerified);
        values.put("terminal_qf_round_trip_attempted",
                authenticatedRoundTripAttempted);
        values.put("terminal_qf_extraction_success", extractionSuccess);
        values.put("terminal_qf_aes_gcm_verified", aesGcmVerified);
        values.put("terminal_qf_exact_message_verified",
                exactMessageVerified);
        values.put("terminal_qf_verification_detail", verificationDetail);
        values.put("terminal_qf_stc_constraint_height",
                configuredStcHeight);

        values.put("terminal_qf_target_band_total", targetTotal);
        values.put("terminal_qf_target_band_nonzero", targetNonzero);
        values.put("terminal_qf_target_band_zero", targetZero);
        values.put("terminal_qf_target_band_nonzero_ratio",
                targetNonzeroRatio);
        values.put("terminal_qf_high_tail_total", highTailTotal);
        values.put("terminal_qf_high_tail_nonzero", highTailNonzero);
        values.put("terminal_qf_high_tail_zero", highTailZero);
        values.put("terminal_qf_high_tail_nonzero_ratio",
                highTailNonzeroRatio);
        values.put("terminal_qf_universe_count", universeCount);
        values.put("terminal_qf_dry_candidate_count", dryCount);
        values.put("terminal_qf_actual_candidate_count",
                actualCandidateCount);
        values.put("terminal_qf_candidate_loading", candidateLoading);
        values.put("terminal_qf_former_four_mode_near_half_candidates",
                formerFourModeNearHalfCandidates);
        values.put("terminal_qf_fully_wet_candidates", fullyWetCandidates);
        values.put("terminal_qf_negative_state_count", negativeStateCount);
        values.put("terminal_qf_zero_state_count", zeroStateCount);
        values.put("terminal_qf_positive_state_count", positiveStateCount);
        values.put("terminal_qf_dry_plus_directions", dryPlusDirections);
        values.put("terminal_qf_dry_minus_directions", dryMinusDirections);
        values.put("terminal_qf_wet_plus_directions", wetPlusDirections);
        values.put("terminal_qf_wet_minus_directions", wetMinusDirections);
        values.put("terminal_qf_engineering_capacity_bits",
                engineeringCapacityBits);
        values.put("terminal_qf_capacity_headroom_bits",
                capacityHeadroomBits);
        values.put("terminal_qf_engineering_utilization",
                engineeringUtilization);
        values.put("terminal_qf_capacity_pass", capacityPass);

        StatisticalCostSummary cost = statisticalCostSummary;
        StatisticalCostSummary.Channel cb =
                cost == null ? null : cost.cb();
        StatisticalCostSummary.Channel cr =
                cost == null ? null : cost.cr();
        values.put("terminal_qf_statistical_cost_model",
                cost == null ? null : cost.modelVersion());
        values.put("terminal_qf_cost_regularizer_strength",
                cost == null ? null : cost.regularizerStrength());
        values.put("terminal_qf_cost_mse_weight",
                cost == null ? null : cost.mseWeight());
        values.put("terminal_qf_cost_ssim_weight",
                cost == null ? null : cost.ssimWeight());
        values.put("terminal_qf_cost_kl_weight",
                cost == null ? null : cost.klWeight());
        values.put("terminal_qf_cost_entropy_weight",
                cost == null ? null : cost.entropyWeight());
        values.put("terminal_qf_cost_mse_p95_scale",
                cost == null ? null : cost.mseP95Scale());
        values.put("terminal_qf_cost_ssim_p95_scale",
                cost == null ? null : cost.ssimP95Scale());
        values.put("terminal_qf_cost_kl_p95_scale",
                cost == null ? null : cost.klP95Scale());
        values.put("terminal_qf_cost_entropy_p95_scale",
                cost == null ? null : cost.entropyP95Scale());
        addChannelCostColumns(values, "cb", cb);
        addChannelCostColumns(values, "cr", cr);

        ChannelAllocationEstimate allocation = channelAllocation;
        values.put("terminal_qf_allocation_model",
                allocation == null ? null : allocation.model());
        values.put("terminal_qf_allocation_header_gibbs_lambda",
                allocation == null ? null
                        : allocation.headerGibbsLambda());
        values.put("terminal_qf_allocation_body_gibbs_lambda",
                allocation == null ? null
                        : allocation.bodyGibbsLambda());
        values.put("terminal_qf_allocation_active_cb_share",
                allocation == null ? null : allocation.activeCbShare());
        values.put("terminal_qf_allocation_active_cr_share",
                allocation == null ? null : allocation.activeCrShare());
        values.put("terminal_qf_allocation_cb_entropy_contribution_share",
                allocation == null ? null
                        : allocation.cbEntropyContributionShare());
        values.put("terminal_qf_allocation_cr_entropy_contribution_share",
                allocation == null ? null
                        : allocation.crEntropyContributionShare());
        values.put("terminal_qf_allocation_expected_cb_change_share",
                allocation == null ? null
                        : allocation.expectedCbChangeShare());
        values.put("terminal_qf_allocation_expected_cr_change_share",
                allocation == null ? null
                        : allocation.expectedCrChangeShare());
        values.put("terminal_qf_allocation_actual_modified_cb",
                allocation == null ? null : allocation.actualModifiedCb());
        values.put("terminal_qf_allocation_actual_modified_cr",
                allocation == null ? null : allocation.actualModifiedCr());
        values.put("terminal_qf_allocation_actual_modified_cb_share",
                allocation == null ? null
                        : allocation.actualModifiedCbShare());
        values.put("terminal_qf_allocation_actual_modified_cr_share",
                allocation == null ? null
                        : allocation.actualModifiedCrShare());

        values.put("terminal_qf_balance_profile", balanceProfile);
        values.put("terminal_qf_balance_maximum_attempts",
                balanceMaximumAttempts);
        values.put("terminal_qf_balance_attempt_count", balanceAttemptCount);
        values.put("terminal_qf_final_balance_iteration",
                finalBalanceIteration);
        values.put("terminal_qf_final_balance_scale", finalBalanceScale);
        values.put("terminal_qf_final_balance_multiplier",
                finalBalanceMultiplier);
        values.put("terminal_qf_final_balance_step", finalBalanceStep);
        values.put("terminal_qf_final_header_stc_success",
                finalHeaderStcSuccess);
        values.put("terminal_qf_final_body_stc_success",
                finalBodyStcSuccess);
        values.put("terminal_qf_final_zero_to_plus", finalZeroToPlus);
        values.put("terminal_qf_final_zero_to_minus", finalZeroToMinus);
        values.put("terminal_qf_final_plus_to_zero", finalPlusToZero);
        values.put("terminal_qf_final_minus_to_zero", finalMinusToZero);
        values.put("terminal_qf_final_zero_activations",
                finalZeroActivations);
        values.put("terminal_qf_final_zero_deactivations",
                finalZeroDeactivations);
        values.put("terminal_qf_final_zero_bin_delta", finalZeroBinDelta);
        values.put("terminal_qf_final_syndrome_verified",
                finalSyndromeVerified);
        values.put("terminal_qf_final_balance_accepted",
                finalBalanceAccepted);
        values.put("terminal_qf_final_balance_status",
                finalBalanceStatus);
        values.put("terminal_qf_final_balance_reason",
                finalBalanceReason);

        values.put("terminal_qf_stc_reserve_bits", stcReserveBits);
        values.put("terminal_qf_stc_body_width_cap", stcBodyWidthCap);
        values.put("terminal_qf_stc_native_path_bytes_estimate",
                stcNativePathBytesEstimate);
        values.put("terminal_qf_stc_native_peak_path_bytes_estimate",
                stcNativePeakPathBytesEstimate);
        values.put("terminal_qf_stc_native_metric_bytes_estimate",
                stcNativeMetricBytesEstimate);
        values.put("terminal_qf_stc_header_message_bits",
                stcHeaderMessageBits);
        values.put("terminal_qf_stc_header_carrier_count",
                stcHeaderCarrierCount);
        values.put("terminal_qf_stc_body_message_bits",
                stcBodyMessageBits);
        values.put("terminal_qf_stc_body_carrier_count",
                stcBodyCarrierCount);
        values.put("terminal_qf_stc_total_message_bits",
                stcTotalMessageBits);
        values.put("terminal_qf_stc_total_carrier_count",
                stcTotalCarrierCount);
        values.put("terminal_qf_stc_capacity_headroom_bits",
                stcCapacityHeadroomBits);
        values.put("terminal_qf_stc_overall_relative_payload",
                stcOverallRelativePayload);
        values.put("terminal_qf_stc_header_changed_elements",
                stcHeaderChangedElements);
        values.put("terminal_qf_stc_header_weighted_distortion",
                stcHeaderWeightedDistortion);
        values.put("terminal_qf_stc_header_change_rate",
                stcHeaderChangeRate);
        values.put("terminal_qf_stc_body_changed_elements",
                stcBodyChangedElements);
        values.put("terminal_qf_stc_body_weighted_distortion",
                stcBodyWeightedDistortion);
        values.put("terminal_qf_stc_body_change_rate",
                stcBodyChangeRate);
        values.put("terminal_qf_stc_total_changed_elements",
                stcTotalChangedElements);
        values.put("terminal_qf_stc_total_weighted_distortion",
                stcTotalWeightedDistortion);
        values.put("terminal_qf_stc_overall_change_rate",
                stcOverallChangeRate);
        values.put("terminal_qf_stc_embedding_efficiency_bits_per_change",
                stcEmbeddingEfficiencyBitsPerChange);
        return values;
    }

    private void resetPerQf() {
        role = "LAST_TESTED";
        stage = "NOT_STARTED";
        terminalStatus = "NOT_RUN";
        terminalReason = "";
        geometryMode = "";
        samplingGeometry = "";
        requiredPayloadBits = null;
        referenceBuilt = null;
        candidateStatisticsAvailable = null;
        kernelCompleted = null;
        coefficientWritebackVerified = null;
        authenticatedRoundTripAttempted = null;
        extractionSuccess = null;
        aesGcmVerified = null;
        exactMessageVerified = null;
        verificationDetail = "";

        targetTotal = null;
        targetNonzero = null;
        targetZero = null;
        targetNonzeroRatio = null;
        highTailTotal = null;
        highTailNonzero = null;
        highTailZero = null;
        highTailNonzeroRatio = null;
        universeCount = null;
        dryCount = null;
        actualCandidateCount = null;
        candidateLoading = null;
        formerFourModeNearHalfCandidates = null;
        fullyWetCandidates = null;
        negativeStateCount = null;
        zeroStateCount = null;
        positiveStateCount = null;
        dryPlusDirections = null;
        dryMinusDirections = null;
        wetPlusDirections = null;
        wetMinusDirections = null;
        engineeringCapacityBits = null;
        capacityHeadroomBits = null;
        engineeringUtilization = null;
        capacityPass = null;
        statisticalCostSummary = null;
        channelAllocation = null;

        balanceProfile = "";
        balanceMaximumAttempts = null;
        balanceAttemptCount = null;
        finalBalanceIteration = null;
        finalBalanceScale = null;
        finalBalanceMultiplier = null;
        finalBalanceStep = null;
        finalHeaderStcSuccess = null;
        finalBodyStcSuccess = null;
        finalZeroToPlus = null;
        finalZeroToMinus = null;
        finalPlusToZero = null;
        finalMinusToZero = null;
        finalZeroActivations = null;
        finalZeroDeactivations = null;
        finalZeroBinDelta = null;
        finalSyndromeVerified = null;
        finalBalanceAccepted = null;
        finalBalanceStatus = "";
        finalBalanceReason = "";

        stcReserveBits = null;
        stcHeaderMessageBits = null;
        stcHeaderCarrierCount = null;
        stcBodyMessageBits = null;
        stcBodyCarrierCount = null;
        stcTotalMessageBits = null;
        stcTotalCarrierCount = null;
        stcCapacityHeadroomBits = null;
        stcOverallRelativePayload = null;
        stcHeaderChangedElements = null;
        stcHeaderWeightedDistortion = null;
        stcHeaderChangeRate = null;
        stcBodyChangedElements = null;
        stcBodyWeightedDistortion = null;
        stcBodyChangeRate = null;
        stcTotalChangedElements = null;
        stcTotalWeightedDistortion = null;
        stcOverallChangeRate = null;
        stcEmbeddingEfficiencyBitsPerChange = null;
        stcBodyWidthCap = configuredStcHeight == null
                ? null : StcCarrierPlan.bodyWidthCap(configuredStcHeight);
        stcNativePathBytesEstimate = null;
        stcNativePeakPathBytesEstimate = null;
        stcNativeMetricBytesEstimate = null;
    }

    private static void addChannelCostColumns(
            Map<String, Object> values,
            String channel,
            StatisticalCostSummary.Channel summary
    ) {
        String prefix = "terminal_qf_cost_" + channel + "_";
        values.put(prefix + "universe_count",
                summary == null ? null : summary.universeCount());
        values.put(prefix + "dry_count",
                summary == null ? null : summary.dryCount());
        values.put(prefix + "mean_base_uniward_cost",
                summary == null ? null : summary.meanBaseUniwardCost());
        values.put(prefix + "mean_rgb_mse_proxy",
                summary == null ? null : summary.meanRgbMseEnergyProxy());
        values.put(prefix + "mean_ssim_loss_proxy",
                summary == null ? null : summary.meanSsimLossProxy());
        values.put(prefix + "mean_kl_risk",
                summary == null ? null : summary.meanKlRisk());
        values.put(prefix + "mean_entropy_risk",
                summary == null ? null : summary.meanEntropyRisk());
        values.put(prefix + "mean_statistical_multiplier",
                summary == null ? null
                        : summary.meanStatisticalMultiplier());
        values.put(prefix + "median_final_min_cost",
                summary == null ? null : summary.medianFinalMinCost());
    }

    private static String safe(String value) {
        return value == null ? "" : value.replaceAll("[\\r\\n]+", " ");
    }

    private static String text(Object value) {
        return value == null ? "none" : String.valueOf(value);
    }
}
