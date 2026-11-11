package thesis.rap.app;

import thesis.rap.analysis.ChannelMetrics;
import thesis.rap.analysis.CsvUtil;
import thesis.rap.analysis.EvaluationSummary;
import thesis.rap.analysis.ImageComparisonMetrics;
import thesis.rap.analysis.JpegCoefficientReport;
import thesis.rap.analysis.JpegMetadata;
import thesis.rap.payload.PayloadPackage;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** Regression coverage for successful and unsuccessful terminal-QF rows. */
public final class ResultAuditWriterSelfTest {
    private ResultAuditWriterSelfTest() {}

    public static void main(String[] args) throws Exception {
        Path directory = Files.createTempDirectory("rap-final-qf-ledger-");
        Path ledger = directory.resolve("final_qf.csv");
        try {
            appendSuccess(ledger);
            appendFailure(ledger);
            List<Map<String, String>> rows = CsvUtil.read(ledger);
            require(rows.size() == 2, "expected exactly two terminal rows");
            verifySuccess(rows.get(0));
            verifyFailure(rows.get(1));
            List<String> physicalLines = Files.readAllLines(ledger);
            System.out.printf(
                    "ResultAuditWriterSelfTest: PASSED, columns=%d, "
                            + "header_bytes=%d, synthetic_success_bytes=%d, "
                            + "synthetic_failure_bytes=%d, total_bytes=%d%n",
                    rows.get(0).size(),
                    utf8Bytes(physicalLines.get(0)),
                    utf8Bytes(physicalLines.get(1)),
                    utf8Bytes(physicalLines.get(2)), Files.size(ledger));
        } finally {
            Files.deleteIfExists(
                    ledger.resolveSibling(ledger.getFileName() + ".lock"));
            Files.deleteIfExists(ledger);
            Files.deleteIfExists(directory);
        }
    }

    private static void appendSuccess(Path ledger) throws Exception {
        FinalQfAuditState state = new FinalQfAuditState();
        state.configureStcHeight(12);
        state.begin(82, 4_096, ThreeStateEmbeddingEngine.Profile.BOUNDED);
        QfBandCoefficientReport.Row qf = qfRow(82, 4_096, 6_000);
        qf.selectedQfCandidate = true;
        qf.terminalStatus = "VERIFIED";
        state.captureCandidateRow(qf);
        state.markCoefficientWritebackVerified();
        state.captureVerification(true, true, true, "OK");
        state.finish("VERIFIED", "", true);

        EvaluationSummary evaluation = evaluation(82, 4_096);
        PayloadPackage.PayloadLayout payload =
                PayloadPackage.inspectLayout(new byte[256]);
        ResultAuditWriter.Entry entry = ResultAuditWriter.Entry.builder()
                .identity(
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:00:02Z",
                        "success-run", "test-version",
                        "EMBED_AND_AUTHENTICATED_EXTRACT")
                .paths(
                        Path.of("000001.png"), Path.of("001.msg"),
                        Path.of("success.jpg"))
                .qfSearch(75, 100, "capacity")
                .outcome(
                        "VERIFIED_SUCCESS", true, true, true, true,
                        true, true)
                .hashes("image-sha", "message-sha", "stego-sha")
                .failure("", "OK")
                .finalQf(state)
                .payload(payload)
                .evaluation(evaluation, 1_500L, 500L)
                .inputEvidence(10_000L, 128, 128, 256L, 80)
                .outputEvidence(8_000L, 128, 128, 3)
                .build();
        ResultAuditWriter.append(ledger, entry);
    }

    private static void appendFailure(Path ledger) throws Exception {
        FinalQfAuditState state = new FinalQfAuditState();
        state.configureStcHeight(10);
        state.begin(100, 8_192, ThreeStateEmbeddingEngine.Profile.BOUNDED);
        QfBandCoefficientReport.Row qf = qfRow(100, 8_192, 7_000);
        qf.capacityPass = false;
        qf.terminalStatus = "CAPACITY_REJECTED";
        qf.reason = "dry_count_minus_32_below_required_bits";
        state.captureCandidateRow(qf);
        state.finish(qf.terminalStatus, qf.reason, false);
        ResultAuditWriter.Entry entry = ResultAuditWriter.Entry.builder()
                .identity(
                        "2026-01-01T00:00:03Z",
                        "2026-01-01T00:00:04Z",
                        "failure-run", "test-version",
                        "EMBED_AND_AUTHENTICATED_EXTRACT")
                .paths(
                        Path.of("000002.png"), Path.of("001.msg"),
                        Path.of("failure.jpg"))
                .qfSearch(75, 100, "capacity")
                .outcome(
                        "UNABLE_TO_EMBED_OR_EXTRACT",
                        false, false, false, false, false, false)
                .hashes("image-sha-2", "message-sha", "")
                .failure("NO_VERIFIED_QF", "No QF verified")
                .finalQf(state)
                .evaluationUnavailable(
                        "NO_VERIFIED_QF_HAS_NO_STEGO_EVALUATION",
                        1_000L)
                .inputEvidence(11_000L, 128, 128, 512L, 80)
                .build();
        ResultAuditWriter.append(ledger, entry);
    }

    private static QfBandCoefficientReport.Row qfRow(
            int qf,
            int payloadBits,
            int dryCandidates
    ) {
        QfBandCoefficientReport.Row row =
                new QfBandCoefficientReport.Row();
        row.qf = qf;
        row.geometryMode = "ALL_GEOMETRY";
        row.samplingGeometry = "4:2:0";
        row.requiredPayloadBits = payloadBits;
        row.targetTotal = 8_000;
        row.targetNonzero = 5_000;
        row.targetZero = 3_000;
        row.highTailTotal = 6_000;
        row.highTailNonzero = 4_000;
        row.highTailZero = 2_000;
        row.universeCount = 8_000;
        row.dryCount = dryCandidates;
        row.actualCandidateCount = dryCandidates;
        row.fullyWetCandidates = 8_000 - dryCandidates;
        row.targetEngineeringCapacity = dryCandidates - 32;
        row.candidateLoading = payloadBits / (double) dryCandidates;
        row.engineeringUtilization = payloadBits
                / (double) row.targetEngineeringCapacity;
        row.capacityPass = row.targetEngineeringCapacity >= payloadBits;
        row.kernelRan = row.capacityPass;
        return row;
    }

    private static EvaluationSummary evaluation(int qf, int payloadBits) {
        EvaluationSummary summary = new EvaluationSummary();
        summary.pngVsStego = imageMetrics(3.0, 43.0, 0.995);
        summary.referenceVsStego = imageMetrics(0.25, 54.0, 0.9995);
        summary.jpegMetadata = new JpegMetadata(
                128, 128, 3, "4:2:0", 2, 2,
                8, 8, 64L, 256, 64, 64,
                "Y:2x2;Cb:1x1;Cr:1x1", "0;1;1", "quant-sha");
        JpegCoefficientReport report = new JpegCoefficientReport();
        report.payloadBits = payloadBits;
        report.selectedQF = qf;
        report.candidateCount = 6_000;
        report.activeCarrierCount = 5_500;
        report.conservativeCapacity = 5_968;
        report.stcReserveBits = 32;
        report.stcConstraintHeight = 10;
        report.totalCoefficients = 24_576;
        report.totalNonzeroAC = 12_000;
        report.modifiedCoefficients = 700;
        report.modifiedCb = 420;
        report.modifiedCr = 280;
        report.candidateLoading = payloadBits / 6_000.0;
        report.overallBpnzAC = payloadBits / 12_000.0;
        report.engineeringUtilization = payloadBits / 5_968.0;
        report.stcWeightedDistortion = 123.456;
        report.targetBandJensenShannon = 0.00012;
        report.embeddingSuccess = true;
        report.extractionSuccess = true;
        report.aesGcmVerified = true;
        report.recoveredMessageMatches = true;
        report.channelMetrics.add(channel(1, "Cb", 420));
        report.channelMetrics.add(channel(2, "Cr", 280));
        summary.coefficientReport = report;
        return summary;
    }

    private static ImageComparisonMetrics imageMetrics(
            double mse,
            double psnr,
            double ssim
    ) {
        ImageComparisonMetrics metrics = new ImageComparisonMetrics();
        metrics.mse = mse;
        metrics.psnr = psnr;
        metrics.ssim = ssim;
        metrics.fileSizeA = 9_000L;
        metrics.fileSizeB = 8_000L;
        metrics.fileSizeDiff = -1_000L;
        metrics.fileSizeRatio = 8.0 / 9.0;
        return metrics;
    }

    private static ChannelMetrics channel(
            int component,
            String name,
            long modified
    ) {
        ChannelMetrics metrics = new ChannelMetrics();
        metrics.componentIndex = component;
        metrics.channel = name;
        metrics.totalCoefficients = 4_000L;
        metrics.modifiedCoefficients = modified;
        metrics.modificationRateEligible = modified / 3_000.0;
        metrics.jensenShannon = 0.0001 * component;
        return metrics;
    }

    private static void verifySuccess(Map<String, String> row) {
        // Literal on purpose: a silent ResultAuditWriter.SCHEMA_VERSION bump
        // must fail here until the ledger consumers are reviewed.
        require("1.0".equals(row.get("schema_version")), "wrong schema");
        require("PORTABLE_SCALAR_BINARY64_STC_V1".equals(
                        row.get("stc_numeric_profile")),
                "STC numeric profile missing");
        require(row.containsKey(
                        "terminal_qf_stc_native_peak_path_bytes_estimate"),
                "peak native path-memory field missing");
        require(row.containsKey(
                        "terminal_qf_stc_native_metric_bytes_estimate"),
                "native metric-memory field missing");
        require(row.containsKey(
                        "terminal_qf_allocation_header_gibbs_lambda"),
                "header Gibbs multiplier field missing");
        require(row.containsKey(
                        "terminal_qf_allocation_body_gibbs_lambda"),
                "body Gibbs multiplier field missing");
        require("12".equals(row.get(
                "terminal_qf_stc_constraint_height")),
                "selected STC height missing");
        require("10,12".equals(row.get("stc_supported_heights")),
                "supported height declaration missing");
        require("82".equals(row.get("terminal_qf")),
                "selected QF missing");
        require("SELECTED_VERIFIED".equals(row.get("terminal_qf_role")),
                "selected role missing");
        require("true".equals(row.get("final_qf_evaluation_available")),
                "evaluation missing");
        require("54.0000000000000".equals(row.get("reference_stego_psnr")),
                "PSNR missing");
        require("420".equals(row.get("cb_modified_coefficients")),
                "Cb metrics missing");
        require("280".equals(row.get("cr_modified_coefficients")),
                "Cr metrics missing");
        require("false".equals(row.get("qf_history_retained")),
                "QF history was retained");
        require("false".equals(row.get("balance_history_retained")),
                "balance history was retained");
    }

    private static void verifyFailure(Map<String, String> row) {
        require("100".equals(row.get("terminal_qf")),
                "last tested QF missing");
        require("LAST_TESTED".equals(row.get("terminal_qf_role")),
                "last-tested role missing");
        require("CAPACITY_REJECTED".equals(
                row.get("terminal_qf_status")),
                "terminal rejection missing");
        require("NO_VERIFIED_QF".equals(row.get("failure_reason_code")),
                "failure code missing");
        require("false".equals(row.get("final_qf_evaluation_available")),
                "failure fabricated an evaluation");
        require(row.get("reference_stego_psnr").isEmpty(),
                "failure fabricated PSNR");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static int utf8Bytes(String value) {
        return value.getBytes(StandardCharsets.UTF_8).length + 1;
    }
}
