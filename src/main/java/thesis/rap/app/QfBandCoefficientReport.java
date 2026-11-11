package thesis.rap.app;

import thesis.rap.alg.CandidateSet;
import thesis.rap.alg.JpegMath;
import thesis.rap.alg.StatisticalCostSummary;
import thesis.rap.analysis.CsvUtil;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * QF-by-QF diagnostic report for the statistical ZZ0(16:47) candidate policy.
 *
 * For every examined QF, this report writes:
 * - whether the reproducible dry set passes the engineering-capacity prefilter,
 * - zero/nonzero counts in the target band ZZ0(16:47), and
 * - zero/nonzero counts in the high-frequency tail ZZ0(48:63) for diagnostic comparison only.
 *
 * No embedding is performed here.
 */
public final class QfBandCoefficientReport {
    private static final int STC_RESERVE_BITS = StcCarrierPlan.DEFAULT_HEADER_RESERVE_BITS;

    private QfBandCoefficientReport() {}

    public static final class Row {
        public int qf;
        public String geometryMode = "";
        public String samplingGeometry = "";
        public int requiredPayloadBits;
        public int targetTotal;
        public int targetNonzero;
        public int targetZero;
        public int targetEngineeringCapacity;
        public int universeCount;
        public int dryCount;
        public int actualCandidateCount;
        public double candidateLoading;
        public int formerFourModeNearHalfCandidates;
        public int fullyWetCandidates;
        public int negativeStateCount;
        public int zeroStateCount;
        public int positiveStateCount;
        public int dryPlusDirections;
        public int dryMinusDirections;
        public int wetPlusDirections;
        public int wetMinusDirections;
        public boolean kernelRan;
        public int balanceAttemptCount;
        public String terminalStatus = "NOT_RUN";
        public String reason = "";
        public double engineeringUtilization;
        public boolean capacityPass;
        public boolean pass;
        public boolean selectionPass;
        public String selectionPolicy = "capacity";
        public boolean selectedQfCandidate;
        public int highTailTotal;
        public int highTailNonzero;
        public int highTailZero;
        public StatisticalCostSummary statisticalCostSummary;
        public ChannelAllocationEstimate channelAllocation;

        public double targetNonzeroRatio() {
            return targetTotal == 0 ? 0.0 : targetNonzero / (double) targetTotal;
        }

        public double highTailNonzeroRatio() {
            return highTailTotal == 0 ? 0.0 : highTailNonzero / (double) highTailTotal;
        }
    }

    public static Row analyze(JpegImage img, int requiredPayloadBits, int qf) {
        SamplingGeometry geometry = SamplingGeometry.detect(img);
        SamplingGeometryMode mode = switch (geometry) {
            case JPEG_444 -> SamplingGeometryMode.GEOMETRY_444;
            case JPEG_422 -> SamplingGeometryMode.GEOMETRY_422;
            case JPEG_420 -> SamplingGeometryMode.GEOMETRY_420;
            case JPEG_440 -> SamplingGeometryMode.GEOMETRY_440;
            case JPEG_411 -> SamplingGeometryMode.GEOMETRY_411;
            case JPEG_441 -> SamplingGeometryMode.GEOMETRY_441;
            case JPEG_410 -> SamplingGeometryMode.GEOMETRY_410;
        };
        return analyze(img, requiredPayloadBits, qf, mode, geometry);
    }

    public static Row analyze(
            JpegImage img,
            int requiredPayloadBits,
            int qf,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry
    ) {
        if (geometryMode == null || geometry == null) {
            throw new IllegalArgumentException(
                    "geometryMode and geometry cannot be null");
        }
        geometry.require(img);
        Row row = new Row();
        row.qf = qf;
        row.geometryMode = geometryMode.token();
        row.samplingGeometry = geometry.label();
        row.requiredPayloadBits = requiredPayloadBits;

        int[] target = countBand(img, JpegMath.TARGET_ZZ_START_RANK, JpegMath.TARGET_ZZ_END_RANK);
        int[] highTail = countBand(img, JpegMath.HIGH_TAIL_ZZ_START_RANK, JpegMath.HIGH_TAIL_ZZ_END_RANK);

        row.targetTotal = target[0];
        row.targetNonzero = target[1];
        row.targetZero = target[2];
        // With no PNG-derived cost context this overload can report only the
        // value-independent geometric universe. applyCandidateSet(...) replaces
        // these fields with the actual reproducible dry-set counts.
        row.targetEngineeringCapacity = Math.max(0, row.targetTotal - STC_RESERVE_BITS);
        row.actualCandidateCount = row.targetTotal;
        row.candidateLoading = requiredPayloadBits / Math.max(1.0, row.actualCandidateCount);
        row.engineeringUtilization = requiredPayloadBits / Math.max(1.0, row.targetEngineeringCapacity);
        row.capacityPass = row.targetEngineeringCapacity >= requiredPayloadBits;
        row.pass = row.capacityPass;
        row.selectionPass = row.pass;

        row.highTailTotal = highTail[0];
        row.highTailNonzero = highTail[1];
        row.highTailZero = highTail[2];

        return row;
    }

    public static void applyCandidateSet(Row row, CandidateSet candidateSet, int requiredPayloadBits) {
        row.statisticalCostSummary =
                candidateSet.statisticalCostSummary();
        row.universeCount = candidateSet.universeCount();
        row.dryCount = candidateSet.dryCount();
        row.actualCandidateCount = candidateSet.dryCount();
        row.fullyWetCandidates = candidateSet.fullyWetCount();
        row.formerFourModeNearHalfCandidates =
                (int) candidateSet.universe().stream()
                .filter(candidate -> candidate.formerFourModeNearHalf)
                .count();
        row.targetEngineeringCapacity = Math.max(
                0, candidateSet.dryCount() - STC_RESERVE_BITS);
        row.candidateLoading = requiredPayloadBits
                / Math.max(1.0, candidateSet.dryCount());
        row.engineeringUtilization = requiredPayloadBits
                / Math.max(1.0, row.targetEngineeringCapacity);
        row.capacityPass = row.targetEngineeringCapacity >= requiredPayloadBits;
        row.pass = row.capacityPass;
        row.selectionPass = row.capacityPass;
        for (var candidate : candidateSet.universe()) {
            if (candidate.qRef < 0) row.negativeStateCount++;
            else if (candidate.qRef == 0) row.zeroStateCount++;
            else row.positiveStateCount++;
            if (candidate.plusLegal()) row.dryPlusDirections++;
            else row.wetPlusDirections++;
            if (candidate.minusLegal()) row.dryMinusDirections++;
            else row.wetMinusDirections++;
        }
    }

    public static void applyKernel(
            Row row,
            FixedQfEmbeddingKernel.Result kernel
    ) {
        if (row == null || kernel == null) return;
        row.channelAllocation = kernel.channelAllocation();
    }

    /** Returns {total, nonzero, zero} over Cb/Cr only for a zig-zag-rank interval. */
    private static int[] countBand(JpegImage img, int startRank, int endRank) {
        int total = 0;
        int nonzero = 0;
        int zero = 0;

        for (int componentIndex = 1; componentIndex <= 2 && componentIndex < img.components.length; componentIndex++) {
            JpegComponent component = img.components[componentIndex];
            int blockCount = component.blockCount();

            for (int block = 0; block < blockCount; block++) {
                for (int rank = startRank; rank <= endRank; rank++) {
                    int k = JpegMath.naturalIndexFromZigzagRank(rank);
                    int coeff = component.getCoeff(block, k);
                    total++;
                    if (coeff == 0) zero++;
                    else nonzero++;
                }
            }
        }

        return new int[]{total, nonzero, zero};
    }

    public static void writeCsv(Path csv, List<Row> rows) throws IOException {
        Path parent = csv.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);

        try (BufferedWriter bw = Files.newBufferedWriter(csv, StandardCharsets.UTF_8)) {
            bw.write(String.join(",",
                    "qf",
                    "geometry_mode",
                    "sampling_geometry",
                    "required_payload_bits",
                    "target_band",
                    "target_total_cbcr_coefficients",
                    "target_nonzero_cbcr_coefficients",
                    "target_zero_cbcr_coefficients",
                    "target_nonzero_ratio",
                    "universe_count",
                    "dry_candidate_count",
                    "actual_candidate_count",
                    "candidate_loading",
                    "former_four_mode_near_half_candidates",
                    "fully_wet_candidates",
                    "negative_state_count",
                    "zero_state_count",
                    "positive_state_count",
                    "dry_plus_directions",
                    "dry_minus_directions",
                    "wet_plus_directions",
                    "wet_minus_directions",
                    "cost_model_version",
                    "cost_mse_p95_scale",
                    "cost_ssim_p95_scale",
                    "cost_kl_p95_scale",
                    "cost_entropy_p95_scale",
                    "cb_dry_candidates",
                    "cr_dry_candidates",
                    "cb_mean_mse_proxy",
                    "cr_mean_mse_proxy",
                    "cb_mean_ssim_loss_proxy",
                    "cr_mean_ssim_loss_proxy",
                    "cb_mean_kl_risk",
                    "cr_mean_kl_risk",
                    "cb_mean_entropy_risk",
                    "cr_mean_entropy_risk",
                    "cb_mean_statistical_multiplier",
                    "cr_mean_statistical_multiplier",
                    "cb_median_final_min_cost",
                    "cr_median_final_min_cost",
                    "allocation_model",
                    "allocation_header_gibbs_lambda",
                    "allocation_body_gibbs_lambda",
                    "allocation_active_cb_share",
                    "allocation_active_cr_share",
                    "allocation_cb_entropy_contribution_share",
                    "allocation_cr_entropy_contribution_share",
                    "allocation_actual_modified_cb",
                    "allocation_actual_modified_cr",
                    "allocation_actual_modified_cb_share",
                    "allocation_actual_modified_cr_share",
                    "engineering_message_capacity",
                    "engineering_utilization",
                    "pass_payload_capacity",
                    "pass_selection_policy",
                    "selection_policy",
                    "selected_qf_candidate",
                    "kernel_ran",
                    "balance_attempt_count",
                    "terminal_status",
                    "reason",
                    "high_tail_band",
                    "high_tail_total_cbcr_coefficients",
                    "high_tail_nonzero_cbcr_coefficients",
                    "high_tail_zero_cbcr_coefficients",
                    "high_tail_nonzero_ratio"
            ));
            bw.newLine();

            for (Row r : rows) {
                StatisticalCostSummary s = r.statisticalCostSummary;
                StatisticalCostSummary.Channel cb =
                        s == null ? null : s.cb();
                StatisticalCostSummary.Channel cr =
                        s == null ? null : s.cr();
                ChannelAllocationEstimate a = r.channelAllocation;
                bw.write(CsvUtil.encodeRow(List.of(
                        String.valueOf(r.qf),
                        r.geometryMode,
                        r.samplingGeometry,
                        String.valueOf(r.requiredPayloadBits),
                        JpegMath.targetBandLabel(),
                        String.valueOf(r.targetTotal),
                        String.valueOf(r.targetNonzero),
                        String.valueOf(r.targetZero),
                        fmt(r.targetNonzeroRatio()),
                        String.valueOf(r.universeCount),
                        String.valueOf(r.dryCount),
                        String.valueOf(r.actualCandidateCount),
                        fmt(r.candidateLoading),
                        String.valueOf(
                                r.formerFourModeNearHalfCandidates),
                        String.valueOf(r.fullyWetCandidates),
                        String.valueOf(r.negativeStateCount),
                        String.valueOf(r.zeroStateCount),
                        String.valueOf(r.positiveStateCount),
                        String.valueOf(r.dryPlusDirections),
                        String.valueOf(r.dryMinusDirections),
                        String.valueOf(r.wetPlusDirections),
                        String.valueOf(r.wetMinusDirections),
                        s == null ? "" : s.modelVersion(),
                        s == null ? "" : fmt(s.mseP95Scale()),
                        s == null ? "" : fmt(s.ssimP95Scale()),
                        s == null ? "" : fmt(s.klP95Scale()),
                        s == null ? "" : fmt(s.entropyP95Scale()),
                        cb == null ? "" : String.valueOf(cb.dryCount()),
                        cr == null ? "" : String.valueOf(cr.dryCount()),
                        cb == null ? "" : fmt(cb.meanRgbMseEnergyProxy()),
                        cr == null ? "" : fmt(cr.meanRgbMseEnergyProxy()),
                        cb == null ? "" : fmt(cb.meanSsimLossProxy()),
                        cr == null ? "" : fmt(cr.meanSsimLossProxy()),
                        cb == null ? "" : fmt(cb.meanKlRisk()),
                        cr == null ? "" : fmt(cr.meanKlRisk()),
                        cb == null ? "" : fmt(cb.meanEntropyRisk()),
                        cr == null ? "" : fmt(cr.meanEntropyRisk()),
                        cb == null ? "" : fmt(cb.meanStatisticalMultiplier()),
                        cr == null ? "" : fmt(cr.meanStatisticalMultiplier()),
                        cb == null ? "" : fmt(cb.medianFinalMinCost()),
                        cr == null ? "" : fmt(cr.medianFinalMinCost()),
                        a == null ? "" : a.model(),
                        a == null ? "" : fmt(a.headerGibbsLambda()),
                        a == null ? "" : fmt(a.bodyGibbsLambda()),
                        a == null ? "" : fmt(a.activeCbShare()),
                        a == null ? "" : fmt(a.activeCrShare()),
                        a == null ? "" : fmt(a.cbEntropyContributionShare()),
                        a == null ? "" : fmt(a.crEntropyContributionShare()),
                        a == null ? "" : String.valueOf(a.actualModifiedCb()),
                        a == null ? "" : String.valueOf(a.actualModifiedCr()),
                        a == null ? "" : fmt(a.actualModifiedCbShare()),
                        a == null ? "" : fmt(a.actualModifiedCrShare()),
                        String.valueOf(r.targetEngineeringCapacity),
                        fmt(r.engineeringUtilization),
                        String.valueOf(r.capacityPass),
                        String.valueOf(r.selectionPass),
                        r.selectionPolicy,
                        String.valueOf(r.selectedQfCandidate),
                        String.valueOf(r.kernelRan),
                        String.valueOf(r.balanceAttemptCount),
                        r.terminalStatus,
                        r.reason,
                        JpegMath.highTailLabel(),
                        String.valueOf(r.highTailTotal),
                        String.valueOf(r.highTailNonzero),
                        String.valueOf(r.highTailZero),
                        fmt(r.highTailNonzeroRatio())
                )));
                bw.newLine();
            }
        }
    }

    private static String fmt(double v) {
        return Double.isFinite(v)
                ? String.format(Locale.US, "%.9f", v) : "";
    }
}
