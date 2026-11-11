package thesis.rap.jni;

import java.util.Objects;

/** Quantized DCT component data and codec metadata read directly from libjpeg. */
public class JpegComponent {
    public final int componentIndex;
    public final int widthBlocks;
    public final int heightBlocks;
    public final int hSamplingFactor;
    public final int vSamplingFactor;
    public final int quantTableIndex;
    public final int[] coefficients; // blocks * 64 in JPEG natural order
    public final int[] quantTable;    // 64 entries

    public JpegComponent(int componentIndex, int widthBlocks, int heightBlocks,
                         int hSamplingFactor, int vSamplingFactor, int quantTableIndex,
                         int[] coefficients, int[] quantTable) {
        if (componentIndex < 0) {
            throw new IllegalArgumentException(
                    "componentIndex must be nonnegative");
        }
        if (widthBlocks <= 0 || heightBlocks <= 0) {
            throw new IllegalArgumentException(
                    "JPEG component dimensions must be positive");
        }
        if (hSamplingFactor <= 0 || vSamplingFactor <= 0) {
            throw new IllegalArgumentException(
                    "JPEG sampling factors must be positive");
        }
        if (quantTableIndex < 0) {
            throw new IllegalArgumentException(
                    "quantTableIndex must be nonnegative");
        }
        Objects.requireNonNull(coefficients, "coefficients");
        Objects.requireNonNull(quantTable, "quantTable");
        int expectedCoefficients = Math.multiplyExact(
                Math.multiplyExact(widthBlocks, heightBlocks), 64);
        if (coefficients.length != expectedCoefficients) {
            throw new IllegalArgumentException(
                    "Coefficient array length mismatch: expected "
                            + expectedCoefficients + ", got "
                            + coefficients.length);
        }
        if (quantTable.length != 64) {
            throw new IllegalArgumentException(
                    "Quantization table must contain 64 entries");
        }
        for (int value : quantTable) {
            if (value < 1 || value > 255) {
                throw new IllegalArgumentException(
                        "Baseline quantization values must be in 1..255");
            }
        }
        this.componentIndex = componentIndex;
        this.widthBlocks = widthBlocks;
        this.heightBlocks = heightBlocks;
        this.hSamplingFactor = hSamplingFactor;
        this.vSamplingFactor = vSamplingFactor;
        this.quantTableIndex = quantTableIndex;
        this.coefficients = coefficients;
        this.quantTable = quantTable;
    }

    public int blockCount() {
        return Math.multiplyExact(widthBlocks, heightBlocks);
    }
    public int getCoeff(int blockIndex, int k) { return coefficients[blockIndex * 64 + k]; }
    public void setCoeff(int blockIndex, int k, int v) { coefficients[blockIndex * 64 + k] = v; }
}
