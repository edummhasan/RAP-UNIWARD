package thesis.rap.analysis;

import thesis.rap.alg.JpegMath;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Finalizes a completed experiment without re-running embedding.
 *
 * <p>Outputs exact mean and median for every numeric run-level metric, plus
 * per-ZZ-position detector-facing summaries and a human-readable final conclusion.</p>
 */
public final class ExperimentSummaryGenerator {
    private static final Set<String> RUN_NON_NUMERIC = Set.of(
            "schema_version", "run_id", "message_group", "message_name",
            "image", "message", "selection_policy", "timestamp_utc",
            "application_version", "input_image_sha256", "message_sha256",
            "reference_jpeg_sha256", "stego_jpeg_sha256", "java_version",
            "os_name", "os_arch", "reference_encoder", "jpeg_subsampling",
            "jpeg_sampling_factors", "jpeg_quant_table_indexes",
            "jpeg_quant_tables_sha256", "huffman_used",
            "stc_duplicate_column_risk", "embedding_success",
            "extraction_success", "aes_gcm_verified",
            "recovered_message_matches"
    );

    private static final List<String> POSITION_SUMMARY_METRICS = List.of(
            "eligible_candidates","active_carriers","modified_coefficients","candidate_selection_rate",
            "modification_rate_eligible","modification_rate_active","modification_share_all_changes",
            "plus_minus_imbalance","mean_abs_delta","hist_l1","total_variation","kl_cover_to_stego",
            "symmetric_kl","jensen_shannon","hellinger","horizontal_corr_delta","vertical_corr_delta"
    );

    private ExperimentSummaryGenerator() {}

    public static void generate(Path runCsv, Path positionCsv, Path attemptCsv, Path outDir) throws IOException {
        Path parent = runCsv.toAbsolutePath().getParent();
        Path channelCsv = parent == null
                ? Path.of("evaluation_channels.csv")
                : parent.resolve("evaluation_channels.csv");
        Path channelPositionCsv = parent == null
                ? Path.of("evaluation_channel_zz_positions.csv")
                : parent.resolve("evaluation_channel_zz_positions.csv");
        generate(runCsv, positionCsv, channelCsv, channelPositionCsv, attemptCsv, outDir);
    }

    public static void generate(Path runCsv, Path positionCsv, Path channelCsv, Path channelPositionCsv,
                                Path attemptCsv, Path outDir) throws IOException {
        Files.createDirectories(outDir);
        RunAggregate runAggregate = aggregateRuns(runCsv);
        writeMetricSummary(outDir.resolve("run_metric_summary.csv"), runAggregate.metrics);
        writeQfGroupSummary(outDir.resolve("qf_group_summary.csv"), runAggregate.qfGroups);
        writeQfDistribution(runCsv, outDir.resolve("qf_distribution.csv"));
        writeThresholdPassRates(runCsv, outDir.resolve("threshold_pass_rates.csv"));
        writeGroupedRunSummary(runCsv, "image", outDir.resolve("image_group_summary.csv"));
        writeGroupedRunSummary(runCsv, "message", outDir.resolve("message_group_summary.csv"));
        writeGroupedRunSummary(runCsv, "raw_message_bytes", outDir.resolve("payload_size_group_summary.csv"));
        MetricReferenceCatalog.write(outDir.resolve("metric_reference_values.csv"));

        Map<Integer, Map<String, DoubleSeries>> positions = Files.exists(positionCsv)
                ? aggregatePositions(positionCsv, runAggregate.verifiedRunIds)
                : Map.of();
        if (!positions.isEmpty()) {
            writePositionSummary(outDir.resolve("zz_position_summary.csv"), positions);
        }

        AttemptSummary attempts = Files.exists(attemptCsv) ? aggregateAttempts(attemptCsv) : new AttemptSummary();
        writeFinalConclusion(outDir.resolve("final_conclusion.txt"), runAggregate, positions, attempts);
        ExperimentAdvancedAnalyzer.generate(runCsv, positionCsv, channelCsv, channelPositionCsv, attemptCsv, outDir);
    }

    private static final List<String> GROUP_METRICS = List.of(
            "selected_qf","payload_bits","candidate_count","candidate_loading","overall_bpnzac",
            "candidate_vs_global_normalization_ratio","engineering_utilization","true_stc_relative_payload","candidate_modification_rate",
            "global_modification_rate","modified_per_payload_bit","embedding_efficiency_bits_per_modification","ref_stego_psnr","ref_stego_ssim",
            "dct_global_l1","dct_global_kl","dct_global_js","target_band_l1","target_band_kl","target_band_js"
    );

    private static void writeQfDistribution(Path runCsv, Path out) throws IOException {
        Map<Integer, Long> counts = new TreeMap<>();
        long total = 0;
        try (CsvUtil.TableReader table = CsvUtil.openTable(runCsv)) {
            if (!table.header().contains("selected_qf")) {
                throw new IOException(
                        "Run CSV is missing required column selected_qf: "
                                + runCsv);
            }
            Map<String, String> row;
            while ((row = table.readRow()) != null) {
                if (!verifiedRow(row)) continue;
                int qf = parseRequiredIntInRange(
                        row.get("selected_qf"), "selected_qf", 75, 100);
                counts.merge(qf, 1L, Long::sum);
                total++;
            }
        }
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("selected_qf,runs,percentage"); bw.newLine();
            for (Map.Entry<Integer, Long> e : counts.entrySet()) {
                bw.write(String.format(Locale.US, "%d,%d,%.12g%n", e.getKey(), e.getValue(),
                        total == 0 ? 0.0 : 100.0 * e.getValue() / total));
            }
        }
    }

    private static void writeThresholdPassRates(Path runCsv, Path out) throws IOException {
        Map<String, ThresholdCounter> counters = new LinkedHashMap<>();
        try (CsvUtil.TableReader table = CsvUtil.openTable(runCsv)) {
            Map<String, String> row;
            while ((row = table.readRow()) != null) {
                if (!verifiedRow(row)) continue;
                requireSelectedQf(row);
                check(counters, "PSNR > 40 dB", "quality heuristic", parseDouble(row.get("ref_stego_psnr"), "ref_stego_psnr"), v -> v > 40.0);
                check(counters, "PSNR >= 50 dB", "quality heuristic", parseDouble(row.get("ref_stego_psnr"), "ref_stego_psnr"), v -> v >= 50.0);
                check(counters, "SSIM >= 0.98", "quality heuristic", parseDouble(row.get("ref_stego_ssim"), "ref_stego_ssim"), v -> v >= 0.98);
                check(counters, "SSIM >= 0.99", "quality heuristic", parseDouble(row.get("ref_stego_ssim"), "ref_stego_ssim"), v -> v >= 0.99);
                check(counters, "SSIM >= 0.999", "strict diagnostic", parseDouble(row.get("ref_stego_ssim"), "ref_stego_ssim"), v -> v >= 0.999);
                check(counters, "SSIM >= 0.9999", "strict diagnostic", parseDouble(row.get("ref_stego_ssim"), "ref_stego_ssim"), v -> v >= 0.9999);
                check(counters, "Global modification rate < 0.05%", "diagnostic only", parseDouble(row.get("global_modification_rate"), "global_modification_rate"), v -> v < 0.0005);
                check(counters, "Candidate modification rate < 50%", "diagnostic only", parseDouble(row.get("candidate_modification_rate"), "candidate_modification_rate"), v -> v < 0.5);
                check(counters, "Global DCT KL < 1e-6", "diagnostic only", parseDouble(row.get("dct_global_kl"), "dct_global_kl"), v -> v < 1e-6);
                check(counters, "Global DCT L1 < 5e-4", "diagnostic only", parseDouble(row.get("dct_global_l1"), "dct_global_l1"), v -> v < 5e-4);
                check(counters, "Engineering utilization <= 0.95", "diagnostic only", parseDouble(row.get("engineering_utilization"), "engineering_utilization"), v -> v <= 0.95);
            }
        }
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("criterion,threshold_type,passed,total,pass_rate_percent"); bw.newLine();
            for (Map.Entry<String, ThresholdCounter> e : counters.entrySet()) {
                ThresholdCounter c = e.getValue();
                bw.write(CsvUtil.encodeRow(List.of(e.getKey(), c.type, String.valueOf(c.passed), String.valueOf(c.total),
                        String.format(Locale.US, "%.12g", c.total == 0 ? 0.0 : 100.0 * c.passed / c.total))));
                bw.newLine();
            }
        }
    }

    private static void writeGroupedRunSummary(Path runCsv, String groupColumn, Path out) throws IOException {
        Map<String, Map<String, DoubleSeries>> groups = new LinkedHashMap<>();
        try (CsvUtil.TableReader table = CsvUtil.openTable(runCsv)) {
            Map<String, String> row;
            while ((row = table.readRow()) != null) {
                if (!verifiedRow(row)) continue;
                requireSelectedQf(row);
                String groupKey = requiredText(row, groupColumn);
                Map<String, DoubleSeries> metrics = groups.computeIfAbsent(groupKey, k -> new LinkedHashMap<>());
                for (String metric : GROUP_METRICS) {
                    Double value = parseDouble(row.get(metric), metric);
                    if (value != null) metrics.computeIfAbsent(metric, k -> new DoubleSeries()).add(value);
                }
            }
        }
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("group_column,group_value,metric,count,mean,median,stddev,min,p95,max"); bw.newLine();
            for (Map.Entry<String, Map<String, DoubleSeries>> group : groups.entrySet()) {
                for (Map.Entry<String, DoubleSeries> metric : group.getValue().entrySet()) {
                    DoubleSeries d = metric.getValue();
                    bw.write(CsvUtil.encodeRow(List.of(groupColumn, group.getKey(), metric.getKey(), String.valueOf(d.count()),
                            fmt(d.mean()), fmt(d.median()), fmt(d.stddev()), fmt(d.min()), fmt(d.percentile(0.95)), fmt(d.max()))));
                    bw.newLine();
                }
            }
        }
    }

    private static void check(Map<String, ThresholdCounter> counters, String name, String type, Double value, DoublePredicate predicate) {
        if (value == null) return;
        ThresholdCounter c = counters.computeIfAbsent(name, k -> new ThresholdCounter(type));
        c.total++;
        if (predicate.test(value)) c.passed++;
    }

    @FunctionalInterface
    private interface DoublePredicate { boolean test(double value); }

    private static final class ThresholdCounter {
        final String type;
        long passed;
        long total;
        ThresholdCounter(String type) { this.type = type; }
    }

    private static RunAggregate aggregateRuns(Path csv) throws IOException {
        if (!Files.exists(csv)) throw new IOException("Run CSV not found: " + csv);
        RunAggregate out = new RunAggregate();
        try (CsvUtil.TableReader table = CsvUtil.openTable(csv)) {
            List<String> header = table.header();
            Map<String, String> row;
            while ((row = table.readRow()) != null) {
                if (!verifiedRow(row)) continue;
                int qf = requireSelectedQf(row);
                String runId = requiredText(row, "run_id");
                requireSha256(row, "input_image_sha256");
                requireSha256(row, "message_sha256");
                requireSha256(row, "reference_jpeg_sha256");
                requireSha256(row, "stego_jpeg_sha256");
                if (!out.verifiedRunIds.add(runId)) {
                    throw new IOException(
                            "Duplicate verified run_id in run CSV: "
                                    + runId);
                }
                out.runCount++;

                for (String column : header) {
                    if (RUN_NON_NUMERIC.contains(column)) continue;
                    Double value = parseDouble(row.get(column), column);
                    if (value != null) out.metrics.computeIfAbsent(column, k -> new DoubleSeries())
                            .add(value, sourceImageClusterKey(row));
                }

                String band = qfBand(qf);
                Map<String, DoubleSeries> group = out.qfGroups.computeIfAbsent(band, k -> new LinkedHashMap<>());
                for (String metric : List.of("candidate_loading","overall_bpnzac","engineering_utilization",
                        "candidate_modification_rate","global_modification_rate","ref_stego_psnr","ref_stego_ssim",
                        "dct_global_l1","dct_global_kl","target_band_l1","target_band_kl","target_band_js")) {
                    Double value = parseDouble(row.get(metric), metric);
                    if (value != null) group.computeIfAbsent(metric, k -> new DoubleSeries())
                            .add(value, sourceImageClusterKey(row));
                }
            }
        }
        return out;
    }

    private static boolean verifiedRow(Map<String, String> row)
            throws IOException {
        boolean embedded = strictBoolean(row, "embedding_success");
        boolean extracted = strictBoolean(row, "extraction_success");
        boolean authenticated = strictBoolean(row, "aes_gcm_verified");
        boolean messageMatched = strictBoolean(
                row, "recovered_message_matches");
        return embedded && extracted && authenticated && messageMatched;
    }

    private static String sourceImageClusterKey(Map<String, String> row)
            throws IOException {
        return requireSha256(row, "input_image_sha256");
    }

    private static Map<Integer, Map<String, DoubleSeries>> aggregatePositions(Path csv, Set<String> verifiedRunIds) throws IOException {
        Map<Integer, Map<String, DoubleSeries>> out = new LinkedHashMap<>();
        try (CsvUtil.TableReader table = CsvUtil.openTable(csv)) {
            Map<String, String> row;
            while ((row = table.readRow()) != null) {
                if (!verifiedRunIds.contains(row.getOrDefault("run_id", ""))) continue;
                int rank = parseTargetRank(row.get("zz_rank"));
                Map<String, DoubleSeries> metrics = out.computeIfAbsent(rank, k -> new LinkedHashMap<>());
                for (String metric : POSITION_SUMMARY_METRICS) {
                    Double value = parseDouble(row.get(metric), metric);
                    if (value != null) metrics.computeIfAbsent(metric, k -> new DoubleSeries()).add(value);
                }
            }
        }
        return out;
    }

    /** Parse the active study band without duplicating its normative bounds. */
    static int parseTargetRank(String raw) throws IOException {
        return parseRequiredIntInRange(
                raw, "zz_rank",
                JpegMath.TARGET_ZZ_START_RANK,
                JpegMath.TARGET_ZZ_END_RANK);
    }

    private static AttemptSummary aggregateAttempts(Path csv) throws IOException {
        AttemptSummary out = new AttemptSummary();
        try (CsvUtil.TableReader table = CsvUtil.openTable(csv)) {
            Map<String, String> row;
            while ((row = table.readRow()) != null) {
                out.total++;
                String status = requiredText(row, "status");
                out.byStatus.merge(status, 1L, Long::sum);
            }
        }
        return out;
    }

    private static void writeMetricSummary(Path out, Map<String, DoubleSeries> metrics) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("metric,row_count,mean,median,stddev,min,p01,p05,p25,p75,p95,p99,max,source_image_clusters,source_cluster_mean,source_cluster_ci95_low,source_cluster_ci95_high");
            bw.newLine();
            for (Map.Entry<String, DoubleSeries> e : metrics.entrySet()) {
                DoubleSeries s = e.getValue();
                bw.write(String.join(",",
                        e.getKey(), String.valueOf(s.count()), fmt(s.mean()), fmt(s.median()), fmt(s.stddev()),
                        fmt(s.min()), fmt(s.percentile(0.01)), fmt(s.percentile(0.05)), fmt(s.percentile(0.25)),
                        fmt(s.percentile(0.75)), fmt(s.percentile(0.95)), fmt(s.percentile(0.99)), fmt(s.max()),
                        String.valueOf(s.clusterCount()), fmt(s.clusterMean()),
                        fmt(s.clusterCi95Low()), fmt(s.clusterCi95High())));
                bw.newLine();
            }
        }
    }

    private static void writeQfGroupSummary(Path out, Map<String, Map<String, DoubleSeries>> groups) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("qf_band,metric,count,mean,median,stddev,min,p95,max");
            bw.newLine();
            for (Map.Entry<String, Map<String, DoubleSeries>> group : groups.entrySet()) {
                for (Map.Entry<String, DoubleSeries> metric : group.getValue().entrySet()) {
                    DoubleSeries s = metric.getValue();
                    bw.write(String.join(",", group.getKey(), metric.getKey(), String.valueOf(s.count()),
                            fmt(s.mean()), fmt(s.median()), fmt(s.stddev()), fmt(s.min()), fmt(s.percentile(0.95)), fmt(s.max())));
                    bw.newLine();
                }
            }
        }
    }

    private static void writePositionSummary(Path out, Map<Integer, Map<String, DoubleSeries>> positions) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("zz_rank,metric,count,mean,median,stddev,min,p05,p95,p99,max");
            bw.newLine();
            for (Map.Entry<Integer, Map<String, DoubleSeries>> rank : positions.entrySet()) {
                for (String metric : POSITION_SUMMARY_METRICS) {
                    DoubleSeries s = rank.getValue().get(metric);
                    if (s == null) continue;
                    bw.write(String.join(",", String.valueOf(rank.getKey()), metric, String.valueOf(s.count()),
                            fmt(s.mean()), fmt(s.median()), fmt(s.stddev()), fmt(s.min()), fmt(s.percentile(0.05)),
                            fmt(s.percentile(0.95)), fmt(s.percentile(0.99)), fmt(s.max())));
                    bw.newLine();
                }
            }
        }
    }

    private static void writeFinalConclusion(
            Path out,
            RunAggregate runs,
            Map<Integer, Map<String, DoubleSeries>> positions,
            AttemptSummary attempts
    ) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("RAP-UNIWARD " + JpegMath.targetBandLabel()
                    + " FINAL STATISTICAL CONCLUSION"); bw.newLine();
            bw.write("===================================================="); bw.newLine();
            bw.write("Successful run records: " + runs.runCount); bw.newLine();
            if (attempts.total > 0) {
                long completed = attempts.byStatus.getOrDefault("SUCCESS", 0L);
                bw.write(String.format(Locale.US, "All attempted pairs: %d | successful: %d | completion rate: %.6f%%%n",
                        attempts.total, completed, 100.0 * completed / attempts.total));
                bw.write("Attempt status counts: " + attempts.byStatus); bw.newLine();
            }
            bw.newLine();

            bw.write("CORE MEAN / MEDIAN STATUS"); bw.newLine();
            writeCoreLine(bw, runs.metrics, "selected_qf", "Selected QF");
            writeCoreLine(bw, runs.metrics, "ref_stego_psnr", "Reference-stego PSNR (dB)");
            writeCoreLine(bw, runs.metrics, "ref_stego_ssim", "Reference-stego SSIM");
            writeCoreLine(bw, runs.metrics, "candidate_loading", "Candidate loading / bits per candidate");
            writeCoreLine(bw, runs.metrics, "overall_bpnzac", "Global nonzero-AC bpnzAC");
            writeCoreLine(bw, runs.metrics, "candidate_vs_global_normalization_ratio", "Candidate/global normalization ratio (x)");
            writeCoreLine(bw, runs.metrics, "engineering_utilization", "Engineering utilization");
            writeCoreLine(bw, runs.metrics, "true_stc_relative_payload", "True native-STC relative payload");
            writeCoreLine(bw, runs.metrics, "embedding_efficiency_bits_per_modification", "Embedding efficiency (payload bits/modification)");
            writeCoreLine(bw, runs.metrics, "candidate_modification_rate", "Candidate modification rate");
            writeCoreLine(bw, runs.metrics, "global_modification_rate", "Global modification rate");
            writeCoreLine(bw, runs.metrics, "modified_per_payload_bit", "Modified coefficients per payload bit");
            writeCoreLine(bw, runs.metrics, "dct_smoothed_l1", "Smoothed global DCT L1");
            writeCoreLine(bw, runs.metrics, "dct_smoothed_kl", "Smoothed global DCT KL");
            writeCoreLine(bw, runs.metrics, "dct_global_l1", "Exact-support global DCT L1");
            writeCoreLine(bw, runs.metrics, "dct_global_kl", "Exact-support global DCT KL");
            writeCoreLine(bw, runs.metrics, "dct_global_js", "Global DCT JS");
            writeCoreLine(bw, runs.metrics, "target_band_l1", "Target-band DCT L1");
            writeCoreLine(bw, runs.metrics, "target_band_kl", "Target-band DCT KL");
            writeCoreLine(bw, runs.metrics, "target_band_js", "Target-band DCT JS");
            bw.newLine();

            DoubleSeries psnr = runs.metrics.get("ref_stego_psnr");
            DoubleSeries ssim = runs.metrics.get("ref_stego_ssim");
            if (psnr != null && ssim != null) {
                bw.write(String.format(
                        Locale.US,
                        "Visual-fidelity screening means: "
                                + "PSNR=%.9f dB; SSIM=%.12f. "
                                + "These numeric measurements are not "
                                + "steganographic-security grades.",
                        psnr.mean(), ssim.mean()));
                bw.newLine();
            }
            bw.write("Steganalysis interpretation: low global distortion does not by itself establish undetectability. Review target-band and per-ZZ-position divergence, active-carrier concentration, and dependency changes; then report detector AUC and/or P_E from a source-matched held-out test set.");
            bw.newLine();

            if (!positions.isEmpty()) {
                bw.newLine();
                bw.write("PER-ZZ HOTSPOT SUMMARY (rank with largest mean value)"); bw.newLine();
                for (String metric : List.of("modification_rate_eligible","kl_cover_to_stego","jensen_shannon",
                        "horizontal_corr_delta","vertical_corr_delta")) {
                    int bestRank = -1;
                    double best = Double.NEGATIVE_INFINITY;
                    for (Map.Entry<Integer, Map<String, DoubleSeries>> e : positions.entrySet()) {
                        DoubleSeries s = e.getValue().get(metric);
                        if (s == null) continue;
                        double score = metric.endsWith("corr_delta") ? Math.abs(s.mean()) : s.mean();
                        if (score > best) { best = score; bestRank = e.getKey(); }
                    }
                    if (bestRank >= 0) {
                        DoubleSeries s = positions.get(bestRank).get(metric);
                        bw.write(String.format(Locale.US, "%s: ZZ0(%d), mean=%s, median=%s%n",
                                metric, bestRank, fmt(s.mean()), fmt(s.median())));
                    }
                }
            }

            bw.newLine();
            bw.write("ALL RUN-LEVEL NUMERIC METRICS: MEAN AND MEDIAN"); bw.newLine();
            bw.write("------------------------------------------------"); bw.newLine();
            for (Map.Entry<String, DoubleSeries> e : runs.metrics.entrySet()) {
                bw.write(String.format(Locale.US, "%s | mean=%s | median=%s | n=%d%n",
                        e.getKey(), fmt(e.getValue().mean()), fmt(e.getValue().median()), e.getValue().count()));
            }
        }
    }

    private static void writeCoreLine(BufferedWriter bw, Map<String, DoubleSeries> metrics, String key, String label) throws IOException {
        DoubleSeries s = metrics.get(key);
        if (s != null) {
            bw.write(String.format(Locale.US, "%s: mean=%s | median=%s | min=%s | p95=%s | max=%s%n",
                    label, fmt(s.mean()), fmt(s.median()), fmt(s.min()), fmt(s.percentile(0.95)), fmt(s.max())));
        }
    }

    private static String qfBand(int qf) {
        if (qf < 75 || qf > 100) {
            throw new IllegalArgumentException(
                    "selected_qf is outside the actual 75-100 search range: "
                            + qf);
        }
        if (qf >= 75 && qf <= 79) return "75-79";
        if (qf <= 84) return "80-84";
        if (qf <= 89) return "85-89";
        if (qf <= 94) return "90-94";
        return "95-100";
    }

    private static Double parseDouble(String value, String column)
            throws IOException {
        if (value == null || value.isBlank()) return null;
        try {
            double d = Double.parseDouble(value);
            if (!Double.isFinite(d)) {
                throw new IOException(
                        "Non-finite numeric value in " + column + ": "
                                + value);
            }
            return d;
        } catch (NumberFormatException e) {
            throw new IOException(
                    "Invalid numeric value in " + column + ": " + value,
                    e);
        }
    }

    private static int requireSelectedQf(Map<String, String> row)
            throws IOException {
        return parseRequiredIntInRange(
                row.get("selected_qf"), "selected_qf", 75, 100);
    }

    private static int parseRequiredIntInRange(
            String value, String column, int minimum, int maximum
    ) throws IOException {
        final int parsed;
        try {
            parsed = Integer.parseInt(value == null ? "" : value.trim());
        } catch (NumberFormatException invalid) {
            throw new IOException(
                    "Missing or invalid integer in " + column + ": "
                            + value, invalid);
        }
        if (parsed < minimum || parsed > maximum) {
            throw new IOException(
                    column + " must be in " + minimum + ".." + maximum
                            + ", observed " + parsed);
        }
        return parsed;
    }

    private static boolean strictBoolean(
            Map<String, String> row, String column
    ) throws IOException {
        String value = row.get(column);
        if ("true".equalsIgnoreCase(value)) return true;
        if ("false".equalsIgnoreCase(value)) return false;
        throw new IOException(
                "Missing or invalid boolean in " + column + ": " + value);
    }

    private static String requiredText(
            Map<String, String> row, String column
    ) throws IOException {
        String value = row.get(column);
        if (value == null || value.isBlank()) {
            throw new IOException(
                    "Verified run is missing required " + column);
        }
        return value;
    }

    private static String requireSha256(
            Map<String, String> row, String column
    ) throws IOException {
        String value = row.getOrDefault(column, "").trim();
        if (!value.matches("[0-9a-fA-F]{64}")) {
            throw new IOException(
                    "Verified run has no valid " + column);
        }
        return value.toLowerCase(Locale.ROOT);
    }

    private static String fmt(double value) {
        if (!Double.isFinite(value)) return "NA";
        return String.format(Locale.US, "%.12g", value);
    }

    private static final class RunAggregate {
        long runCount;
        final Map<String, DoubleSeries> metrics = new LinkedHashMap<>();
        final Map<String, Map<String, DoubleSeries>> qfGroups = new LinkedHashMap<>();
        final Set<String> verifiedRunIds = new java.util.LinkedHashSet<>();
    }

    private static final class AttemptSummary {
        long total;
        final Map<String, Long> byStatus = new LinkedHashMap<>();
    }

    /** Primitive exact-value series: compact enough for exact medians on large experiments. */
    private static final class DoubleSeries {
        private double[] values = new double[1024];
        private int size;
        private double sum;
        private double sumSquares;
        private double min = Double.POSITIVE_INFINITY;
        private double max = Double.NEGATIVE_INFINITY;
        private double[] sortedCache;
        private final Map<String, ClusterAccumulator> clusters = new LinkedHashMap<>();

        void add(double value) {
            if (!Double.isFinite(value)) return;
            if (size == values.length) values = Arrays.copyOf(values, values.length * 2);
            values[size++] = value;
            sum += value;
            sumSquares += value * value;
            min = Math.min(min, value);
            max = Math.max(max, value);
            sortedCache = null;
        }

        void add(double value, String cluster) {
            add(value);
            if (cluster != null && !cluster.isBlank() && Double.isFinite(value)) {
                clusters.computeIfAbsent(cluster, ignored -> new ClusterAccumulator()).add(value);
            }
        }

        int count() { return size; }
        double mean() { return size == 0 ? Double.NaN : sum / size; }
        double min() { return size == 0 ? Double.NaN : min; }
        double max() { return size == 0 ? Double.NaN : max; }

        double stddev() {
            if (size < 2) return Double.NaN;
            double variance = (sumSquares - sum * sum / size) / (size - 1);
            return Math.sqrt(Math.max(0.0, variance));
        }

        double median() { return percentile(0.5); }

        double percentile(double p) {
            if (size == 0) return Double.NaN;
            double[] sorted = sorted();
            if (size == 1) return sorted[0];
            double index = p * (size - 1);
            int lo = (int) Math.floor(index);
            int hi = (int) Math.ceil(index);
            if (lo == hi) return sorted[lo];
            double w = index - lo;
            return sorted[lo] * (1.0 - w) + sorted[hi] * w;
        }

        int clusterCount() { return clusters.size(); }

        double clusterMean() {
            if (clusters.isEmpty()) return Double.NaN;
            double total = 0.0;
            for (ClusterAccumulator cluster : clusters.values()) total += cluster.mean();
            return total / clusters.size();
        }

        double clusterStddev() {
            if (clusters.size() < 2) return Double.NaN;
            double mean = clusterMean();
            double sum = 0.0;
            for (ClusterAccumulator cluster : clusters.values()) {
                double delta = cluster.mean() - mean;
                sum += delta * delta;
            }
            return Math.sqrt(sum / (clusters.size() - 1));
        }

        double clusterCi95Low() {
            double sd = clusterStddev();
            return Double.isFinite(sd) ? clusterMean() - 1.96 * sd / Math.sqrt(clusters.size()) : Double.NaN;
        }

        double clusterCi95High() {
            double sd = clusterStddev();
            return Double.isFinite(sd) ? clusterMean() + 1.96 * sd / Math.sqrt(clusters.size()) : Double.NaN;
        }

        private double[] sorted() {
            if (sortedCache == null) {
                sortedCache = Arrays.copyOf(values, size);
                Arrays.sort(sortedCache);
            }
            return sortedCache;
        }
    }

    private static final class ClusterAccumulator {
        private double sum;
        private int count;
        void add(double value) { sum += value; count++; }
        double mean() { return count == 0 ? Double.NaN : sum / count; }
    }
}
