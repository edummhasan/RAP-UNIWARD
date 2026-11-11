package thesis.rap.analysis;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * Computes empirical steganalysis metrics from detector scores.
 * Expected labels: 0=cover, 1=stego. Higher score must mean "more likely stego".
 */
public final class DetectorMetricsAnalyzer {
    private DetectorMetricsAnalyzer() {}

    public static Result analyze(Path csv, String labelColumn, String scoreColumn, Path outCsv) throws IOException {
        List<Observation> observations = read(csv, labelColumn, scoreColumn);
        Result result = calculate(observations);
        writeSummary(outCsv, result);
        writeRoc(rocPath(outCsv), result.rocPoints());
        return result;
    }

    private static List<Observation> read(Path csv, String labelColumn, String scoreColumn) throws IOException {
        List<Observation> out = new ArrayList<>();
        try (CsvUtil.TableReader table = CsvUtil.openTable(csv)) {
            if (!table.header().contains(labelColumn)
                    || !table.header().contains(scoreColumn)) {
                throw new IOException("Detector CSV must contain columns '" + labelColumn + "' and '" + scoreColumn + "'.");
            }
            java.util.Map<String, String> row;
            long record = 0;
            while ((row = table.readRow()) != null) {
                record++;
                final int label;
                final double score;
                try {
                    label = Integer.parseInt(row.get(labelColumn).trim());
                    score = Double.parseDouble(row.get(scoreColumn).trim());
                } catch (NumberFormatException invalid) {
                    throw new IOException(
                            "Invalid detector value at data record "
                                    + record + " in " + csv, invalid);
                }
                if (label != 0 && label != 1) {
                    throw new IOException(
                            "Detector label must be 0 or 1 at data record "
                                    + record + " in " + csv);
                }
                if (!Double.isFinite(score)) {
                    throw new IOException(
                            "Detector score must be finite at data record "
                                    + record + " in " + csv);
                }
                out.add(new Observation(label, score));
            }
        }
        if (out.isEmpty()) throw new IOException("No valid detector observations found.");
        return out;
    }

    private static Result calculate(List<Observation> observations) {
        int positives = 0;
        int negatives = 0;
        for (Observation o : observations) {
            if (o.label == 1) positives++; else negatives++;
        }
        if (positives == 0 || negatives == 0) {
            throw new IllegalArgumentException("Detector evaluation requires both cover (0) and stego (1) labels.");
        }

        List<Observation> sorted = new ArrayList<>(observations);
        sorted.sort(Comparator.comparingDouble(o -> o.score));

        // Mann-Whitney rank-sum AUC with average ranks for ties.
        double positiveRankSum = 0.0;
        int i = 0;
        while (i < sorted.size()) {
            int j = i + 1;
            while (j < sorted.size() && Double.compare(sorted.get(j).score, sorted.get(i).score) == 0) j++;
            double averageRank = ((i + 1) + j) / 2.0;
            for (int k = i; k < j; k++) if (sorted.get(k).label == 1) positiveRankSum += averageRank;
            i = j;
        }
        double auc = (positiveRankSum - positives * (positives + 1.0) / 2.0) / (positives * (double) negatives);

        List<Double> thresholds = new ArrayList<>();
        thresholds.add(Double.POSITIVE_INFINITY);
        Double previous = null;
        for (int index = sorted.size() - 1; index >= 0; index--) {
            Observation o = sorted.get(index);
            if (previous == null || Double.compare(previous, o.score) != 0) {
                thresholds.add(o.score);
                previous = o.score;
            }
        }
        thresholds.add(Double.NEGATIVE_INFINITY);

        List<RocPoint> roc = new ArrayList<>();
        double bestPe = Double.POSITIVE_INFINITY;
        double bestThreshold = Double.NaN;
        double bestPfa = Double.NaN;
        double bestPmd = Double.NaN;
        double bestAccuracy = Double.NaN;
        double eer = Double.NaN;
        double eerThreshold = Double.NaN;
        double bestEerGap = Double.POSITIVE_INFINITY;

        for (double threshold : thresholds) {
            int fp = 0, fn = 0, tp = 0, tn = 0;
            for (Observation o : observations) {
                boolean predictStego = o.score >= threshold;
                if (o.label == 1 && predictStego) tp++;
                else if (o.label == 1) fn++;
                else if (predictStego) fp++;
                else tn++;
            }
            double pfa = fp / (double) negatives;
            double pmd = fn / (double) positives;
            double tpr = tp / (double) positives;
            double pe = 0.5 * (pfa + pmd);
            double accuracy = (tp + tn) / (double) observations.size();
            roc.add(new RocPoint(threshold, pfa, tpr, pmd, pe, accuracy));

            if (pe < bestPe) {
                bestPe = pe;
                bestThreshold = threshold;
                bestPfa = pfa;
                bestPmd = pmd;
                bestAccuracy = accuracy;
            }
            double gap = Math.abs(pfa - pmd);
            if (gap < bestEerGap) {
                bestEerGap = gap;
                eer = 0.5 * (pfa + pmd);
                eerThreshold = threshold;
            }
        }

        return new Result(observations.size(), negatives, positives, auc, bestPe, bestThreshold,
                bestPfa, bestPmd, bestAccuracy, eer, eerThreshold, List.copyOf(roc));
    }

    private static void writeSummary(Path out, Result r) throws IOException {
        Path parent = out.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("total,cover_count,stego_count,auc,min_detection_error_pe,best_threshold,false_alarm_probability,missed_detection_probability,accuracy_at_best_pe,equal_error_rate,eer_threshold");
            bw.newLine();
            bw.write(String.format(Locale.US, "%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g%n",
                    r.total, r.coverCount, r.stegoCount, r.auc, r.minDetectionErrorPe, r.bestThreshold,
                    r.falseAlarmProbability, r.missedDetectionProbability, r.accuracyAtBestPe,
                    r.equalErrorRate, r.eerThreshold));
        }
    }

    private static void writeRoc(Path out, List<RocPoint> points) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            bw.write("threshold,false_positive_rate,true_positive_rate,missed_detection_probability,detection_error_pe,accuracy");
            bw.newLine();
            for (RocPoint p : points) {
                bw.write(String.format(Locale.US, "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g%n",
                        p.threshold, p.falsePositiveRate, p.truePositiveRate, p.missedDetectionProbability,
                        p.detectionErrorPe, p.accuracy));
            }
        }
    }

    private static Path rocPath(Path summaryPath) {
        String name = summaryPath.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String stem = dot > 0 ? name.substring(0, dot) : name;
        String ext = dot > 0 ? name.substring(dot) : ".csv";
        Path parent = summaryPath.toAbsolutePath().getParent();
        return (parent == null ? Path.of(stem + "_roc" + ext) : parent.resolve(stem + "_roc" + ext));
    }

    public record Result(
            int total,
            int coverCount,
            int stegoCount,
            double auc,
            double minDetectionErrorPe,
            double bestThreshold,
            double falseAlarmProbability,
            double missedDetectionProbability,
            double accuracyAtBestPe,
            double equalErrorRate,
            double eerThreshold,
            List<RocPoint> rocPoints
    ) {
        public void print() {
            System.out.printf(Locale.US,
                    "Detector results: n=%d (cover=%d, stego=%d), AUC=%.6f, min P_E=%.6f, EER=%.6f, best threshold=%.9g, P_FA=%.6f, P_MD=%.6f, accuracy=%.6f%n",
                    total, coverCount, stegoCount, auc, minDetectionErrorPe, equalErrorRate, bestThreshold,
                    falseAlarmProbability, missedDetectionProbability, accuracyAtBestPe);
            System.out.println("Interpretation direction: AUC -> 0.5 and P_E -> 0.5 indicate chance-level separation.");
        }
    }

    public record RocPoint(
            double threshold,
            double falsePositiveRate,
            double truePositiveRate,
            double missedDetectionProbability,
            double detectionErrorPe,
            double accuracy
    ) {}

    private record Observation(int label, double score) {}
}
