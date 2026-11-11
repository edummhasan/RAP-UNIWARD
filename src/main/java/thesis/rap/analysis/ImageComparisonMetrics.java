package thesis.rap.analysis;

public class ImageComparisonMetrics {
    public String label;

    public double mse;
    public double psnr;
    public double ssim;

    public double entropyA;
    public double entropyB;
    public double entropyDiff;

    public double histogramL1;
    public double klDivergence;

    public long fileSizeA;
    public long fileSizeB;
    public long fileSizeDiff;

    public double fileSizeRatio;

    public void print() {
        System.out.println("========== IMAGE QUALITY / STATISTICAL REPORT ==========");
        System.out.println("Comparison: " + label);
        System.out.printf("MSE: %.6f%n", mse);
        System.out.printf("PSNR: %.6f dB%n", psnr);
        System.out.printf("SSIM: %.6f%n", ssim);
        System.out.printf("Entropy A: %.6f%n", entropyA);
        System.out.printf("Entropy B: %.6f%n", entropyB);
        System.out.printf("Entropy Difference: %.6f%n", entropyDiff);
        System.out.printf("Histogram L1 Divergence: %.6f%n", histogramL1);
        System.out.printf("KL Divergence: %.9f%n", klDivergence);
        System.out.printf("File Size A: %d bytes%n", fileSizeA);
        System.out.printf("File Size B: %d bytes%n", fileSizeB);
        System.out.printf("File Size Difference: %+d bytes%n", fileSizeDiff);
        System.out.printf("File Size Ratio B/A: %.6f%n", fileSizeRatio);
        System.out.println("========================================================");
    }
}
