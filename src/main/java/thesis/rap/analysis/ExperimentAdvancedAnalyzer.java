package thesis.rap.analysis;

import java.io.BufferedWriter;
import java.io.IOException;
import thesis.rap.alg.JpegMath;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Advanced batch-level analysis over the real run datasets. */
public final class ExperimentAdvancedAnalyzer {
    private ExperimentAdvancedAnalyzer() {}

    private record Pair(String x, String y) {}
    private record CorrelationResult(String x, String y, int n, double pearson, double spearman) {}

    private static final List<Pair> CORRELATION_PAIRS = List.of(
            new Pair("payload_bits", "selected_qf"),
            new Pair("payload_bits", "candidate_count"),
            new Pair("selected_qf", "candidate_count"),
            new Pair("candidate_loading", "target_band_js"),
            new Pair("candidate_modification_rate", "ref_stego_psnr"),
            new Pair("candidate_modification_rate", "target_band_kl"),
            new Pair("payload_bits", "jpeg_size_change_percent"),
            new Pair("true_stc_relative_payload", "embedding_efficiency_bits_per_modification"),
            new Pair("modified_coefficients", "target_band_js"),
            new Pair("cb_modification_share", "target_band_js")
    );

    public static void generate(
            Path runCsv,
            Path positionCsv,
            Path channelCsv,
            Path channelPositionCsv,
            Path attemptCsv,
            Path outDir
    ) throws IOException {
        List<Map<String, String>> allRuns = readCsv(runCsv);
        List<Map<String, String>> runs = allRuns.stream().filter(ExperimentAdvancedAnalyzer::verifiedRow).toList();
        Set<String> verifiedRunIds = new HashSet<>();
        for (Map<String, String> run : runs) {
            String id = run.getOrDefault("run_id", "");
            if (!id.isBlank()) verifiedRunIds.add(id);
        }
        List<CorrelationResult> correlations = writeCorrelations(runs, outDir.resolve("correlation_analysis.csv"));
        writeOutliers(runs, outDir.resolve("outlier_analysis.csv"));
        writeGroupedNumericSummary(channelCsv, List.of("channel"), verifiedRunIds, outDir.resolve("channel_summary.csv"));
        writeGroupedNumericSummary(channelPositionCsv, List.of("channel", "zz_rank"), verifiedRunIds,
                outDir.resolve("channel_zz_position_summary.csv"));
        Audit audit = writeDataQualityAudit(runs, allRuns.size() - runs.size(), verifiedRunIds,
                positionCsv, channelCsv, channelPositionCsv, attemptCsv,
                outDir.resolve("data_quality_audit.csv"));
        writeJsonSummary(runs, correlations, audit, outDir.resolve("experiment_summary.json"));
    }

    private static List<CorrelationResult> writeCorrelations(List<Map<String, String>> rows, Path out) throws IOException {
        List<CorrelationResult> results = new ArrayList<>();
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("x_metric,y_metric,n,pearson_r,spearman_rho");
            bw.newLine();
            for (Pair pair : CORRELATION_PAIRS) {
                List<Double> xs = new ArrayList<>();
                List<Double> ys = new ArrayList<>();
                for (Map<String, String> row : rows) {
                    Double x = parseDouble(row.get(pair.x), pair.x);
                    Double y = parseDouble(row.get(pair.y), pair.y);
                    if (x != null && y != null) { xs.add(x); ys.add(y); }
                }
                double[] xa = toArray(xs), ya = toArray(ys);
                double pearson = StatisticalMath.pearson(xa, ya);
                double spearman = xa.length < 2 ? Double.NaN : StatisticalMath.pearson(ranks(xa), ranks(ya));
                CorrelationResult result = new CorrelationResult(pair.x, pair.y, xa.length, pearson, spearman);
                results.add(result);
                bw.write(CsvUtil.encodeRow(List.of(pair.x, pair.y, String.valueOf(xa.length), fmt(pearson), fmt(spearman))));
                bw.newLine();
            }
        }
        return results;
    }

    private static void writeOutliers(List<Map<String, String>> rows, Path out) throws IOException {
        Set<String> nonNumeric = Set.of(
                "schema_version", "run_id", "message_group",
                "message_name", "image", "message", "selection_policy",
                "timestamp_utc", "application_version",
                "input_image_sha256", "message_sha256",
                "reference_jpeg_sha256", "stego_jpeg_sha256",
                "java_version", "os_name", "os_arch",
                "reference_encoder", "jpeg_subsampling",
                "jpeg_sampling_factors", "jpeg_quant_table_indexes",
                "jpeg_quant_tables_sha256", "huffman_used",
                "stc_duplicate_column_risk", "embedding_success",
                "extraction_success", "aes_gcm_verified",
                "recovered_message_matches");
        Map<String, List<Double>> byMetric = new LinkedHashMap<>();
        if (!rows.isEmpty()) {
            for (String key : rows.get(0).keySet()) {
                if (nonNumeric.contains(key)) continue;
                List<Double> values = new ArrayList<>();
                for (Map<String, String> row : rows) {
                    Double v = parseDouble(row.get(key), key);
                    if (v != null) values.add(v);
                }
                if (!values.isEmpty()) byMetric.put(key, values);
            }
        }
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("metric,n,q1,q3,iqr,lower_fence,upper_fence,outlier_count,outlier_rate_percent,min,max");
            bw.newLine();
            for (Map.Entry<String, List<Double>> e : byMetric.entrySet()) {
                double[] v = e.getValue().stream().mapToDouble(Double::doubleValue).sorted().toArray();
                double q1 = percentile(v, 0.25), q3 = percentile(v, 0.75), iqr = q3 - q1;
                double lower = q1 - 1.5 * iqr, upper = q3 + 1.5 * iqr;
                long count = Arrays.stream(v).filter(x -> x < lower || x > upper).count();
                bw.write(CsvUtil.encodeRow(List.of(e.getKey(), String.valueOf(v.length), fmt(q1), fmt(q3), fmt(iqr),
                        fmt(lower), fmt(upper), String.valueOf(count), fmt(100.0 * count / v.length), fmt(v[0]), fmt(v[v.length - 1]))));
                bw.newLine();
            }
        }
    }

    private static void writeGroupedNumericSummary(
            Path csv, List<String> groupColumns, Set<String> verifiedRunIds, Path out) throws IOException {
        if (!Files.exists(csv)) return;
        List<Map<String, String>> rows = readCsv(csv);
        if (rows.isEmpty()) return;
        Set<String> nonNumeric = new HashSet<>(groupColumns);
        nonNumeric.addAll(Set.of("schema_version","run_id","image","message","channel"));
        Map<String, Map<String, Series>> groups = new LinkedHashMap<>();
        for (Map<String, String> row : rows) {
            String rowRunId = row.get("run_id");
            if (rowRunId == null || rowRunId.isBlank()) {
                throw new IOException(
                        "Grouped dataset row is missing run_id: " + csv);
            }
            if (!verifiedRunIds.contains(rowRunId)) continue;
            StringBuilder key = new StringBuilder();
            for (int i = 0; i < groupColumns.size(); i++) {
                if (i > 0) key.append('|');
                String groupValue = row.get(groupColumns.get(i));
                if (groupValue == null || groupValue.isBlank()) {
                    throw new IOException(
                            "Grouped dataset row is missing "
                                    + groupColumns.get(i) + ": " + csv);
                }
                key.append(groupValue);
            }
            Map<String, Series> metrics = groups.computeIfAbsent(key.toString(), k -> new LinkedHashMap<>());
            for (Map.Entry<String, String> e : row.entrySet()) {
                if (nonNumeric.contains(e.getKey())) continue;
                Double v = parseDouble(e.getValue(), e.getKey());
                if (v != null) metrics.computeIfAbsent(e.getKey(), k -> new Series()).add(v);
            }
        }
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("group,metric,count,mean,median,stddev,min,p25,p75,p95,max");
            bw.newLine();
            for (Map.Entry<String, Map<String, Series>> g : groups.entrySet()) {
                for (Map.Entry<String, Series> e : g.getValue().entrySet()) {
                    Series s = e.getValue();
                    bw.write(CsvUtil.encodeRow(List.of(g.getKey(), e.getKey(), String.valueOf(s.size()), fmt(s.mean()),
                            fmt(s.percentile(0.5)), fmt(s.stddev()), fmt(s.min()), fmt(s.percentile(0.25)),
                            fmt(s.percentile(0.75)), fmt(s.percentile(0.95)), fmt(s.max()))));
                    bw.newLine();
                }
            }
        }
    }

    private record Audit(long runCount, long excludedUnverifiedRunRows, long uniqueRunIds, long duplicateRunIds, long positionRows, long channelRows,
                         long channelPositionRows, long attemptRows, long stcMismatchRuns, long highTailModifiedRuns,
                         long targetOccupancyChangedRuns, long failedVerificationRuns) {}

    private static Audit writeDataQualityAudit(List<Map<String, String>> runs, long excludedUnverifiedRunRows,
                                               Set<String> verifiedRunIds, Path positionCsv, Path channelCsv,
                                               Path channelPositionCsv, Path attemptCsv, Path out) throws IOException {
        Set<String> ids = new HashSet<>();
        long duplicateIds = 0, mismatch = 0, highTail = 0, occupancy = 0, failedVerification = 0;
        for (Map<String, String> row : runs) {
            String id = row.getOrDefault("run_id", "");
            if (!id.isBlank() && !ids.add(id)) duplicateIds++;
            if (nonZero(
                    row.get("stc_coefficient_change_mismatch"),
                    "stc_coefficient_change_mismatch")) mismatch++;
            if (positive(
                    row.get("high_tail_modified_coefficients"),
                    "high_tail_modified_coefficients")) highTail++;
            Double a = parseDouble(
                    row.get("target_band_cbcr_nonzero_cover"),
                    "target_band_cbcr_nonzero_cover");
            Double b = parseDouble(
                    row.get("target_band_cbcr_nonzero_stego"),
                    "target_band_cbcr_nonzero_stego");
            if (a != null && b != null && Double.compare(a, b) != 0) occupancy++;
            if (!truth(row.get("embedding_success")) || !truth(row.get("extraction_success"))
                    || !truth(row.get("aes_gcm_verified")) || !truth(row.get("recovered_message_matches"))) failedVerification++;
        }
        long pRows = countRowsForRunIds(positionCsv, verifiedRunIds);
        long cRows = countRowsForRunIds(channelCsv, verifiedRunIds);
        long cpRows = countRowsForRunIds(channelPositionCsv, verifiedRunIds);
        long aRows = countDataRows(attemptCsv);
        Audit audit = new Audit(runs.size(), excludedUnverifiedRunRows, ids.size(), duplicateIds, pRows, cRows, cpRows, aRows,
                mismatch, highTail, occupancy, failedVerification);
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("check,observed,expected,status,interpretation"); bw.newLine();
            auditRow(bw, "excluded_unverified_run_rows", excludedUnverifiedRunRows, excludedUnverifiedRunRows, true,
                    "Unverified or incomplete run rows are excluded from every summary and correlation.");
            auditRow(bw, "unique_run_ids", ids.size(), runs.size(), ids.size() == runs.size(), "Each successful run should have one unique run ID.");
            auditRow(bw, "duplicate_run_ids", duplicateIds, 0, duplicateIds == 0, "Duplicate run IDs can corrupt joins across datasets.");
            long bandWidth = (long) JpegMath.TARGET_ZZ_END_RANK
                    - JpegMath.TARGET_ZZ_START_RANK + 1L;
            auditRow(bw, "combined_zz_rows", pRows,
                    runs.size() * bandWidth,
                    pRows == runs.size() * bandWidth,
                    "Exactly " + bandWidth
                            + " combined target-band ZZ rows per successful run.");
            auditRow(bw, "channel_rows", cRows, runs.size() * 2L, cRows == runs.size() * 2L, "Exactly two target-band channel rows (Cb and Cr) per successful run.");
            auditRow(bw, "channel_zz_rows", cpRows,
                    runs.size() * 2L * bandWidth,
                    cpRows == runs.size() * 2L * bandWidth,
                    "Exactly " + (2L * bandWidth)
                            + " channel-position rows per successful run.");
            auditRow(bw, "stc_coefficient_change_mismatch_runs", mismatch, 0, mismatch == 0, "Native parity changes should equal observed coefficient changes.");
            auditRow(bw, "high_tail_modified_runs", highTail, 0, highTail == 0, "The high-tail control band is outside the embedding policy and should remain unchanged.");
            auditRow(bw, "target_nonzero_occupancy_changed_runs", occupancy, 0, true,
                    "Three-state embedding permits occupancy changes; inspect this measured count with zero-flow and JPEG run-length metrics rather than treating it as an invariant.");
            auditRow(bw, "failed_verification_success_rows", failedVerification, 0, failedVerification == 0, "Successful run rows should pass extraction, AES-GCM, and message equality checks.");
            bw.write(CsvUtil.encodeRow(List.of("attempt_rows", String.valueOf(aRows), ">= " + runs.size(), aRows >= runs.size() ? "PASS" : "FAIL", "Attempt ledger should include every successful run and all skipped/failed pairs."))); bw.newLine();
        }
        return audit;
    }

    private static void auditRow(BufferedWriter bw, String check, long observed, long expected, boolean pass, String note) throws IOException {
        bw.write(CsvUtil.encodeRow(List.of(check, String.valueOf(observed), String.valueOf(expected), pass ? "PASS" : "FAIL", note)));
        bw.newLine();
    }

    private static void writeJsonSummary(List<Map<String, String>> runs, List<CorrelationResult> correlations, Audit audit, Path out) throws IOException {
        StringBuilder sb = new StringBuilder(8192);
        sb.append("{\n  \"schema_version\": \"3.0\",\n");
        sb.append("  \"successful_runs\": ").append(runs.size()).append(",\n");
        sb.append("  \"data_quality\": {\n");
        sb.append("    \"excluded_unverified_run_rows\": ").append(audit.excludedUnverifiedRunRows).append(",\n");
        sb.append("    \"unique_run_ids\": ").append(audit.uniqueRunIds).append(",\n");
        sb.append("    \"duplicate_run_ids\": ").append(audit.duplicateRunIds).append(",\n");
        sb.append("    \"stc_coefficient_change_mismatch_runs\": ").append(audit.stcMismatchRuns).append(",\n");
        sb.append("    \"high_tail_modified_runs\": ").append(audit.highTailModifiedRuns).append(",\n");
        sb.append("    \"target_occupancy_changed_runs\": ").append(audit.targetOccupancyChangedRuns).append(",\n");
        sb.append("    \"failed_verification_runs\": ").append(audit.failedVerificationRuns).append("\n  },\n");
        sb.append("  \"correlations\": [\n");
        for (int i = 0; i < correlations.size(); i++) {
            CorrelationResult c = correlations.get(i);
            sb.append("    {\"x\":\"").append(json(c.x)).append("\",\"y\":\"").append(json(c.y))
                    .append("\",\"n\":").append(c.n).append(",\"pearson_r\":").append(jsonNumber(c.pearson))
                    .append(",\"spearman_rho\":").append(jsonNumber(c.spearman)).append('}');
            if (i + 1 < correlations.size()) sb.append(',');
            sb.append('\n');
        }
        sb.append("  ]\n}\n");
        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8);
    }

    private static List<Map<String, String>> readCsv(Path path) throws IOException {
        if (path == null || !Files.exists(path)) return List.of();
        return CsvUtil.read(path);
    }

    private static long countDataRows(Path path) throws IOException {
        if (path == null || !Files.exists(path)) return 0;
        return CsvUtil.countDataRows(path);
    }

    private static long countRowsForRunIds(Path path, Set<String> verifiedRunIds) throws IOException {
        if (path == null || !Files.exists(path) || verifiedRunIds.isEmpty()) return 0;
        long count = 0;
        try (CsvUtil.TableReader table = CsvUtil.openTable(path)) {
            Map<String, String> row;
            while ((row = table.readRow()) != null) {
                if (verifiedRunIds.contains(
                        row.getOrDefault("run_id", ""))) count++;
            }
        }
        return count;
    }

    private static boolean verifiedRow(Map<String, String> row) {
        return truth(row.get("embedding_success"))
                && truth(row.get("extraction_success"))
                && truth(row.get("aes_gcm_verified"))
                && truth(row.get("recovered_message_matches"));
    }

    private static double[] ranks(double[] values) {
        Integer[] idx = new Integer[values.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble(i -> values[i]));
        double[] ranks = new double[values.length];
        int i = 0;
        while (i < idx.length) {
            int j = i + 1;
            while (j < idx.length && Double.compare(values[idx[j]], values[idx[i]]) == 0) j++;
            double rank = (i + 1 + j) / 2.0;
            for (int k = i; k < j; k++) ranks[idx[k]] = rank;
            i = j;
        }
        return ranks;
    }

    private static double percentile(double[] sorted, double p) {
        if (sorted.length == 0) return Double.NaN;
        if (sorted.length == 1) return sorted[0];
        double index = p * (sorted.length - 1);
        int lo = (int) Math.floor(index), hi = (int) Math.ceil(index);
        if (lo == hi) return sorted[lo];
        double w = index - lo;
        return sorted[lo] * (1.0 - w) + sorted[hi] * w;
    }

    private static double[] toArray(List<Double> values) {
        return values.stream().mapToDouble(Double::doubleValue).toArray();
    }
    private static Double parseDouble(String s, String column)
            throws IOException {
        if (s == null || s.isBlank()) return null;
        try {
            double value = Double.parseDouble(s);
            if (!Double.isFinite(value)) {
                throw new IOException(
                        "Non-finite numeric value in " + column + ": " + s);
            }
            return value;
        } catch (NumberFormatException invalid) {
            throw new IOException(
                    "Invalid numeric value in " + column + ": " + s,
                    invalid);
        }
    }
    private static boolean truth(String s) { return "true".equalsIgnoreCase(s); }
    private static boolean nonZero(String s, String column)
            throws IOException {
        Double value = parseDouble(s, column);
        return value != null && value != 0.0;
    }
    private static boolean positive(String s, String column)
            throws IOException {
        Double value = parseDouble(s, column);
        return value != null && value > 0.0;
    }
    private static String fmt(double v) { return Double.isFinite(v) ? String.format(Locale.US, "%.15g", v) : ""; }
    private static String jsonNumber(double v) { return Double.isFinite(v) ? String.format(Locale.US, "%.15g", v) : "null"; }
    private static String json(String s) { return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n"); }

    private static final class Series {
        private final List<Double> values = new ArrayList<>();
        void add(double v) { values.add(v); }
        int size() { return values.size(); }
        double mean() { return values.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN); }
        double min() { return values.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN); }
        double max() { return values.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN); }
        double stddev() {
            if (values.size() < 2) return Double.NaN;
            double mean = mean();
            double ss = 0.0;
            for (double v : values) ss += (v - mean) * (v - mean);
            return Math.sqrt(ss / (values.size() - 1));
        }
        double percentile(double p) { return ExperimentAdvancedAnalyzer.percentile(values.stream().mapToDouble(Double::doubleValue).sorted().toArray(), p); }
    }
}
