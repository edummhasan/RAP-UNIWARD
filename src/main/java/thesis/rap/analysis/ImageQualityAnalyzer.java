package thesis.rap.analysis;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

public final class ImageQualityAnalyzer {

    private static final double EPS = 1e-12;

    private ImageQualityAnalyzer() {}

    public static ImageComparisonMetrics compare(
            String label,
            String imageAPath,
            String imageBPath
    ) throws Exception {

        BufferedImage a = ImageIO.read(new File(imageAPath));
        BufferedImage b = ImageIO.read(new File(imageBPath));

        if (a == null) {
            throw new IllegalArgumentException("Cannot read image A: " + imageAPath);
        }

        if (b == null) {
            throw new IllegalArgumentException("Cannot read image B: " + imageBPath);
        }

        a = toRgb(a);
        b = toRgb(b);

        // Rescaling to force a comparison would produce a fidelity number that
        // describes an interpolation, not the embedding.  Two images of
        // different size are simply not comparable in this pipeline.
        if (a.getWidth() != b.getWidth() || a.getHeight() != b.getHeight()) {
            throw new IllegalArgumentException(
                    "Cannot compare images of different dimensions: "
                            + imageAPath + " is " + a.getWidth() + "x" + a.getHeight()
                            + ", " + imageBPath + " is " + b.getWidth() + "x" + b.getHeight());
        }

        int[] histA = grayscaleHistogram(a);
        int[] histB = grayscaleHistogram(b);

        double[] pA = normalize(histA);
        double[] pB = normalize(histB);

        ImageComparisonMetrics r = new ImageComparisonMetrics();

        r.label = label;
        r.mse = mse(a, b);
        r.psnr = psnr(r.mse);
        r.ssim = windowedSsim(a, b);

        r.entropyA = entropy(pA);
        r.entropyB = entropy(pB);
        r.entropyDiff = Math.abs(r.entropyA - r.entropyB);

        r.histogramL1 = histogramL1(pA, pB);
        r.klDivergence = klDivergence(pA, pB);

        File fA = new File(imageAPath);
        File fB = new File(imageBPath);

        r.fileSizeA = fA.length();
        r.fileSizeB = fB.length();
        r.fileSizeDiff = r.fileSizeB - r.fileSizeA;
        r.fileSizeRatio = r.fileSizeA == 0 ? 0.0 : (double) r.fileSizeB / r.fileSizeA;

        return r;
    }

    private static BufferedImage toRgb(BufferedImage src) {
        BufferedImage out = new BufferedImage(
                src.getWidth(),
                src.getHeight(),
                BufferedImage.TYPE_INT_RGB
        );

        Graphics2D g = out.createGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();

        return out;
    }

    private static double mse(BufferedImage a, BufferedImage b) {
        int w = a.getWidth();
        int h = a.getHeight();

        double total = 0.0;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgbA = a.getRGB(x, y);
                int rgbB = b.getRGB(x, y);

                int rA = (rgbA >> 16) & 0xff;
                int gA = (rgbA >> 8) & 0xff;
                int bA = rgbA & 0xff;

                int rB = (rgbB >> 16) & 0xff;
                int gB = (rgbB >> 8) & 0xff;
                int bB = rgbB & 0xff;

                total += sq(rA - rB);
                total += sq(gA - gB);
                total += sq(bA - bB);
            }
        }

        return total / (w * h * 3.0);
    }

    /**
     * Peak signal-to-noise ratio in dB.
     *
     * <p>Identical images have zero error and therefore infinite PSNR.  A
     * finite stand-in such as 99 dB would be an invented measurement that
     * silently passes quality thresholds, so the true value is reported and
     * consumers must handle it explicitly.</p>
     */
    private static double psnr(double mse) {
        if (mse <= 0.0) {
            return Double.POSITIVE_INFINITY;
        }

        return 10.0 * Math.log10((255.0 * 255.0) / mse);
    }

    /**
     * Mean SSIM as defined by Wang et al. (2004): an 11x11 circular-symmetric
     * Gaussian window with sigma = 1.5, evaluated at every valid position, with
     * K1 = 0.01 and K2 = 0.03 on an 8-bit dynamic range.
     *
*/
    private static double windowedSsim(BufferedImage a, BufferedImage b) {
        int w = a.getWidth();
        int h = a.getHeight();

        int window = 11;
        int half = window / 2;

        if (w < window || h < window) {
            throw new IllegalArgumentException(
                    "SSIM requires images of at least " + window + "x" + window
                            + " pixels; got " + w + "x" + h);
        }

        double[][] weight = gaussianWindow(window, 1.5);

        // Cache the grayscale planes: getRGB() per window sample dominates cost.
        double[][] grayA = grayPlane(a);
        double[][] grayB = grayPlane(b);

        double c1 = sq(0.01 * 255.0);
        double c2 = sq(0.03 * 255.0);

        double total = 0.0;
        long count = 0;

        for (int y = half; y < h - half; y++) {
            for (int x = half; x < w - half; x++) {
                double meanA = 0.0;
                double meanB = 0.0;
                for (int dy = -half; dy <= half; dy++) {
                    for (int dx = -half; dx <= half; dx++) {
                        double gw = weight[dy + half][dx + half];
                        meanA += gw * grayA[y + dy][x + dx];
                        meanB += gw * grayB[y + dy][x + dx];
                    }
                }

                double varA = 0.0;
                double varB = 0.0;
                double cov = 0.0;
                for (int dy = -half; dy <= half; dy++) {
                    for (int dx = -half; dx <= half; dx++) {
                        double gw = weight[dy + half][dx + half];
                        double dA = grayA[y + dy][x + dx] - meanA;
                        double dB = grayB[y + dy][x + dx] - meanB;
                        varA += gw * dA * dA;
                        varB += gw * dB * dB;
                        cov += gw * dA * dB;
                    }
                }

                double numerator = (2 * meanA * meanB + c1) * (2 * cov + c2);
                double denominator =
                        (meanA * meanA + meanB * meanB + c1) * (varA + varB + c2);

                total += numerator / denominator;
                count++;
            }
        }

        return count == 0 ? Double.NaN : total / count;
    }

    /** Normalized circular-symmetric Gaussian window. */
    private static double[][] gaussianWindow(int size, double sigma) {
        int half = size / 2;
        double[][] weight = new double[size][size];
        double sum = 0.0;
        for (int y = -half; y <= half; y++) {
            for (int x = -half; x <= half; x++) {
                double value = Math.exp(-(x * x + y * y) / (2.0 * sigma * sigma));
                weight[y + half][x + half] = value;
                sum += value;
            }
        }
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) weight[y][x] /= sum;
        }
        return weight;
    }

    private static double[][] grayPlane(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        double[][] out = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) out[y][x] = gray(img.getRGB(x, y));
        }
        return out;
    }

    private static int[] grayscaleHistogram(BufferedImage img) {
        int[] hist = new int[256];

        int w = img.getWidth();
        int h = img.getHeight();

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int g = (int) Math.round(gray(img.getRGB(x, y)));
                g = clamp(g, 0, 255);
                hist[g]++;
            }
        }

        return hist;
    }

    private static double[] normalize(int[] hist) {
        double total = 0.0;

        for (int v : hist) {
            total += v;
        }

        double[] p = new double[hist.length];

        for (int i = 0; i < hist.length; i++) {
            p[i] = hist[i] / Math.max(total, EPS);
        }

        return p;
    }

    private static double entropy(double[] p) {
        double h = 0.0;

        for (double v : p) {
            if (v > 0.0) {
                h -= v * log2(v);
            }
        }

        return h;
    }

    private static double histogramL1(double[] a, double[] b) {
        double sum = 0.0;

        for (int i = 0; i < a.length; i++) {
            sum += Math.abs(a[i] - b[i]);
        }

        return sum;
    }

    private static double klDivergence(double[] p, double[] q) {
        if (p.length != q.length) throw new IllegalArgumentException("Histogram lengths differ");
        double pTotal = 0.0;
        double qTotal = 0.0;
        for (int i = 0; i < p.length; i++) {
            pTotal += p[i] + EPS;
            qTotal += q[i] + EPS;
        }
        double sum = 0.0;
        for (int i = 0; i < p.length; i++) {
            double pi = (p[i] + EPS) / pTotal;
            double qi = (q[i] + EPS) / qTotal;
            sum += pi * Math.log(pi / qi);
        }
        return Math.max(0.0, sum);
    }

    private static double gray(int rgb) {
        int r = (rgb >> 16) & 0xff;
        int g = (rgb >> 8) & 0xff;
        int b = rgb & 0xff;

        return 0.299 * r + 0.587 * g + 0.114 * b;
    }

    private static double sq(double x) {
        return x * x;
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
