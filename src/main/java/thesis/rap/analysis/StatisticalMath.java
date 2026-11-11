package thesis.rap.analysis;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Numerically stable statistical helpers used by the detector-facing evaluation layer.
 *
 * <p>No external numerical library is required. Distribution metrics operate on the
 * union of observed integer coefficient values and therefore avoid the clipping that
 * can occur with a fixed histogram range.</p>
 */
public final class StatisticalMath {
    private static final double EPS = 1e-12;

    private StatisticalMath() {}

    public static Map<Integer, Long> histogram(int[] values) {
        Map<Integer, Long> out = new HashMap<>();
        for (int value : values) {
            out.merge(value, 1L, Long::sum);
        }
        return out;
    }

    public static DistributionMetrics compareHistograms(Map<Integer, Long> cover, Map<Integer, Long> stego) {
        Set<Integer> support = new TreeSet<>();
        support.addAll(cover.keySet());
        support.addAll(stego.keySet());

        if (support.isEmpty()) {
            return new DistributionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        double coverTotal = cover.values().stream().mapToDouble(Long::doubleValue).sum();
        double stegoTotal = stego.values().stream().mapToDouble(Long::doubleValue).sum();
        int bins = support.size();

        // Add the same tiny pseudo-count to each observed support bin before KL.
        double coverDenom = coverTotal + EPS * bins;
        double stegoDenom = stegoTotal + EPS * bins;

        double l1 = 0.0;
        double klCoverStego = 0.0;
        double klStegoCover = 0.0;
        double js = 0.0;
        double hellingerSq = 0.0;

        for (int value : support) {
            double p = (cover.getOrDefault(value, 0L) + EPS) / coverDenom;
            double q = (stego.getOrDefault(value, 0L) + EPS) / stegoDenom;
            double m = 0.5 * (p + q);

            l1 += Math.abs(p - q);
            klCoverStego += p * StrictMath.log(p / q);
            klStegoCover += q * StrictMath.log(q / p);
            js += 0.5 * p * StrictMath.log(p / m)
                    + 0.5 * q * StrictMath.log(q / m);
            double rootDiff = StrictMath.sqrt(p) - StrictMath.sqrt(q);
            hellingerSq += rootDiff * rootDiff;
        }

        double totalVariation = 0.5 * l1;
        double symmetricKl = 0.5 * (klCoverStego + klStegoCover);
        double hellinger = StrictMath.sqrt(0.5 * hellingerSq);
        return new DistributionMetrics(l1, totalVariation, klCoverStego, symmetricKl, js, hellinger);
    }

    public static double mean(List<Double> values) {
        if (values.isEmpty()) return Double.NaN;
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / values.size();
    }

    public static double median(List<Double> values) {
        return percentile(values, 0.5);
    }

    public static double percentile(List<Double> values, double p) {
        if (values.isEmpty()) return Double.NaN;
        if (p < 0.0 || p > 1.0) throw new IllegalArgumentException("Percentile must be in [0,1].");
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        if (sorted.size() == 1) return sorted.get(0);
        double index = p * (sorted.size() - 1);
        int lo = (int) Math.floor(index);
        int hi = (int) Math.ceil(index);
        if (lo == hi) return sorted.get(lo);
        double w = index - lo;
        return sorted.get(lo) * (1.0 - w) + sorted.get(hi) * w;
    }

    public static double pearson(double[] x, double[] y) {
        if (x.length != y.length || x.length < 2) return Double.NaN;
        double meanX = 0.0;
        double meanY = 0.0;
        for (int i = 0; i < x.length; i++) {
            meanX += x[i];
            meanY += y[i];
        }
        meanX /= x.length;
        meanY /= y.length;

        double sxx = 0.0;
        double syy = 0.0;
        double sxy = 0.0;
        for (int i = 0; i < x.length; i++) {
            double dx = x[i] - meanX;
            double dy = y[i] - meanY;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
        if (sxx <= 0.0 || syy <= 0.0) return Double.NaN;
        return sxy / StrictMath.sqrt(sxx * syy);
    }

    public record DistributionMetrics(
            double l1,
            double totalVariation,
            double klCoverToStego,
            double symmetricKl,
            double jensenShannon,
            double hellinger
    ) {}
}
