package thesis.rap.image;

import thesis.rap.alg.JpegMath;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;
import thesis.rap.gpu.CudaDctNative;
import thesis.rap.gpu.CudaImagePreprocessNative;
import thesis.rap.runtime.ExecutionBackend;
import thesis.rap.runtime.GpuStageStats;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.HashMap;
import java.util.Map;
import javax.imageio.ImageIO;

/**
 * Run-scoped immutable source-image context.
 *
 * <p>The PNG decode and RGB-to-YCbCr conversion are independent of QF.  RAP
 * scans many QFs for the same image, so rebuilding them for every QF adds CPU
 * work without changing any algorithmic input.  Component resampling/padding
 * is cached by the exact JPEG layout it serves.</p>
 */
public final class SourceImageContext {
    private final Path sourcePath;
    private final String sourceSha256;
    private final long sourceByteCount;
    private final BufferedImage sourceImage;
    private final double[][][] fullPlanes;
    private final double[][] flatFullPlanes;
    private static final int[] ALL_NATURAL_INDICES = allNaturalIndices();

    private final Map<ComponentKey, double[][]> componentSamples = new HashMap<>();
    private final Map<ComponentKey, double[][]> paddedPlanes = new HashMap<>();
    private final Map<ComponentKey, double[]> rawDctCoefficients = new HashMap<>();
    private final Map<ComponentKey, double[]> blockVariances = new HashMap<>();
    private final Map<ComponentKey, CudaDctNative.Result> gpuDctVariance = new HashMap<>();
    private final Map<ComponentKey, ComponentPlanePair> gpuComponentPlanes = new HashMap<>();

    private SourceImageContext(
            Path sourcePath,
            String sourceSha256,
            long sourceByteCount,
            BufferedImage sourceImage,
            double[][][] fullPlanes,
            double[][] flatFullPlanes
    ) {
        this.sourcePath = sourcePath;
        this.sourceSha256 = sourceSha256;
        this.sourceByteCount = sourceByteCount;
        this.sourceImage = sourceImage;
        this.fullPlanes = fullPlanes;
        this.flatFullPlanes = flatFullPlanes;
    }

    public static SourceImageContext load(Path png) throws Exception {
        if (png == null) throw new IllegalArgumentException("png cannot be null");
        // Decode and fingerprint the same byte snapshot. File size and mtime
        // cannot establish content identity, especially after file replacement.
        byte[] bytes = Files.readAllBytes(png);
        byte[] signature = {(byte) 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
        if (bytes.length < signature.length
                || !Arrays.equals(signature, Arrays.copyOf(bytes, signature.length))) {
            throw new IllegalArgumentException("Input image is not an actual PNG file: " + png);
        }
        BufferedImage decoded;
        try (var stream = new ByteArrayInputStream(bytes)) {
            decoded = ImageIO.read(stream);
        }
        if (decoded == null) throw new IOException("Cannot read image: " + png);
        BufferedImage rgb = ImageUtil.toRGB(decoded);
        FullPlaneData data = buildFullPlanes(rgb);
        return new SourceImageContext(
                png.toAbsolutePath().normalize(),
                HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)),
                bytes.length, rgb, data.planes(), data.flatPlanes());
    }

    /**
     * Creates a context for an already decoded image. Its byte provenance is
     * unknown, so it cannot enter the shared content-addressed cache. Production
     * callers must use {@link #load(Path)} to bind pixels to a PNG snapshot.
     */
    public static SourceImageContext fromDecoded(
            Path png, BufferedImage decoded
    ) {
        if (png == null) throw new IllegalArgumentException("png cannot be null");
        if (decoded == null) {
            throw new IllegalArgumentException("decoded image cannot be null");
        }
        FullPlaneData data = buildFullPlanes(decoded);
        return new SourceImageContext(
                png.toAbsolutePath().normalize(), null, -1, decoded,
                data.planes(), data.flatPlanes());
    }

    public Path sourcePath() {
        return sourcePath;
    }

    public String sourceSha256() {
        if (sourceSha256 == null) {
            throw new IllegalStateException("Source byte identity is unavailable; use SourceImageContext.load");
        }
        return sourceSha256;
    }

    public long sourceByteCount() {
        sourceSha256(); // Reject contexts whose PNG bytes were never observed.
        return sourceByteCount;
    }

    /** The already-normalized RGB image used to create JPEG templates. */
    public BufferedImage sourceImage() {
        return sourceImage;
    }

    /** Full-resolution Y/Cb/Cr planes.  Internal algorithms treat these as read-only. */
    public double[][][] fullPlanes() {
        return fullPlanes;
    }

    /** Exact unpadded component sample plane for the supplied JPEG layout. */
    public synchronized double[][] componentSamples(JpegImage jpeg, int componentIndex) {
        ComponentKey key = ComponentKey.of(jpeg, componentIndex);
        double[][] existing = componentSamples.get(key);
        if (existing != null) return existing;
        ComponentPlanePair accelerated = acceleratedComponentPlanes(
                key, jpeg, componentIndex);
        if (accelerated != null) {
            componentSamples.put(key, accelerated.samples());
            paddedPlanes.put(key, accelerated.padded());
            return accelerated.samples();
        }
        double[][] computed = ImageUtil.jpegComponentPlane(
                fullPlanes, jpeg, componentIndex);
        componentSamples.put(key, computed);
        return computed;
    }

    /** Exact component plane padded to the reference JPEG block grid. */
    public synchronized double[][] paddedComponentPlane(
            JpegImage jpeg,
            int componentIndex
    ) {
        ComponentKey key = ComponentKey.of(jpeg, componentIndex);
        double[][] existing = paddedPlanes.get(key);
        if (existing != null) return existing;
        ComponentPlanePair accelerated = acceleratedComponentPlanes(
                key, jpeg, componentIndex);
        if (accelerated != null) {
            componentSamples.put(key, accelerated.samples());
            paddedPlanes.put(key, accelerated.padded());
            return accelerated.padded();
        }
        JpegComponent c = jpeg.component(componentIndex);
        double[][] computed = ImageUtil.padToBlockGrid(
                componentSamples(jpeg, componentIndex),
                c.widthBlocks * 8,
                c.heightBlocks * 8);
        paddedPlanes.put(key, computed);
        return computed;
    }

    /**
     * QF-independent raw forward-DCT grid in block-major natural order.
     * Entry {@code block*64+k} is exactly the unquantized coefficient used by
     * both canonical JPEG construction and side-information rounding error.
     */
    public synchronized double[] rawDctCoefficients(
            JpegImage jpeg, int componentIndex
    ) {
        ComponentKey key = ComponentKey.of(jpeg, componentIndex);
        return rawDctCoefficients.computeIfAbsent(key, ignored -> {
            JpegComponent component = jpeg.component(componentIndex);
            double[][] plane = paddedComponentPlane(jpeg, componentIndex);
            CudaDctNative.Result accelerated = acceleratedDctVariance(
                    key, plane, component);
            if (accelerated != null) {
                blockVariances.putIfAbsent(key, accelerated.variancesView());
                return accelerated.dctView();
            }
            int blockCount = Math.multiplyExact(
                    component.widthBlocks, component.heightBlocks);
            double[] coefficients = new double[Math.multiplyExact(blockCount, 64)];
            for (int by = 0; by < component.heightBlocks; by++) {
                for (int bx = 0; bx < component.widthBlocks; bx++) {
                    int block = by * component.widthBlocks + bx;
                    double[] raw = JpegMath.forwardDctCoefficients(
                            plane, bx * 8, by * 8, ALL_NATURAL_INDICES);
                    System.arraycopy(raw, 0, coefficients, block * 64, 64);
                }
            }
            return coefficients;
        });
    }

    /**
     * QF-independent variance of every 8x8 source-component block.  The exact
     * arithmetic matches the former per-QF/per-block calculation: one mean
     * pass followed by one squared-deviation pass, each in y/x order.
     */
    public synchronized double[] blockVariances(
            JpegImage jpeg, int componentIndex
    ) {
        ComponentKey key = ComponentKey.of(jpeg, componentIndex);
        return blockVariances.computeIfAbsent(key, ignored -> {
            JpegComponent component = jpeg.component(componentIndex);
            double[][] plane = paddedComponentPlane(jpeg, componentIndex);
            CudaDctNative.Result accelerated = acceleratedDctVariance(
                    key, plane, component);
            if (accelerated != null) {
                rawDctCoefficients.putIfAbsent(key, accelerated.dctView());
                return accelerated.variancesView();
            }
            int blockCount = Math.multiplyExact(
                    component.widthBlocks, component.heightBlocks);
            double[] variances = new double[blockCount];
            for (int by = 0; by < component.heightBlocks; by++) {
                for (int bx = 0; bx < component.widthBlocks; bx++) {
                    int block = by * component.widthBlocks + bx;
                    int pixelX = bx * 8;
                    int pixelY = by * 8;
                    double mean = 0.0;
                    for (int y = 0; y < 8; y++) {
                        for (int x = 0; x < 8; x++) {
                            mean += plane[pixelY + y][pixelX + x];
                        }
                    }
                    mean /= 64.0;
                    double variance = 0.0;
                    for (int y = 0; y < 8; y++) {
                        for (int x = 0; x < 8; x++) {
                            double centered =
                                    plane[pixelY + y][pixelX + x] - mean;
                            variance += centered * centered;
                        }
                    }
                    variances[block] = variance / 64.0;
                }
            }
            return variances;
        });
    }

    private ComponentPlanePair acceleratedComponentPlanes(
            ComponentKey key, JpegImage jpeg, int componentIndex
    ) {
        if (!gpuPreprocessEnabled()) return null;
        ComponentPlanePair existing = gpuComponentPlanes.get(key);
        if (existing != null) return existing;
        if (!CudaImagePreprocessNative.isAvailable()) {
            if (strictGpu()) {
                throw new IllegalStateException(
                        "Strict CUDA mode requires GPU image preprocessing: "
                                + CudaImagePreprocessNative.diagnostic());
            }
            return null;
        }
        JpegComponent component = jpeg.component(componentIndex);
        int maxH = 1;
        int maxV = 1;
        for (JpegComponent c : jpeg.components) {
            maxH = Math.max(maxH, c.hSamplingFactor);
            maxV = Math.max(maxV, c.vSamplingFactor);
        }
        if (maxH % component.hSamplingFactor != 0
                || maxV % component.vSamplingFactor != 0) {
            throw new IllegalArgumentException(
                    "Non-integral JPEG component sampling ratio");
        }
        int ratioX = maxH / component.hSamplingFactor;
        int ratioY = maxV / component.vSamplingFactor;
        int sourceHeight = fullPlanes[componentIndex].length;
        int sourceWidth = fullPlanes[componentIndex][0].length;
        int outputWidth = (sourceWidth + ratioX - 1) / ratioX;
        int outputHeight = (sourceHeight + ratioY - 1) / ratioY;
        int paddedWidth = component.widthBlocks * 8;
        int paddedHeight = component.heightBlocks * 8;
        try {
            double[] flat = CudaImagePreprocessNative.resampleAndPad(
                    flatFullPlanes[componentIndex],
                    sourceWidth, sourceHeight,
                    ratioX, ratioY,
                    paddedWidth, paddedHeight);
            double[][] padded = reshape(flat, paddedWidth, paddedHeight);
            double[][] samples = new double[outputHeight][outputWidth];
            for (int y = 0; y < outputHeight; y++) {
                System.arraycopy(padded[y], 0, samples[y], 0, outputWidth);
            }
            ComponentPlanePair computed = new ComponentPlanePair(samples, padded);
            gpuComponentPlanes.put(key, computed);
            GpuStageStats.gpu(GpuStageStats.Stage.RESAMPLE);
            return computed;
        } catch (RuntimeException | LinkageError failure) {
            if (strictGpu()) throw failure;
            GpuStageStats.fallback(GpuStageStats.Stage.RESAMPLE, failure);
            return null;
        }
    }

    private static FullPlaneData buildFullPlanes(BufferedImage decoded) {
        if (gpuPreprocessEnabled() && CudaImagePreprocessNative.isAvailable()) {
            try {
                int width = decoded.getWidth();
                int height = decoded.getHeight();
                // Same white-composited TYPE_INT_RGB raster the CPU path
                // (ImageUtil.yCbCr -> toRGB) converts; a no-op for RGB input.
                int[] rgb = ImageUtil.toRGB(decoded).getRGB(
                        0, 0, width, height, null, 0, width);
                double[] combined = CudaImagePreprocessNative.rgbToYCbCr(
                        rgb, width, height);
                int pixels = Math.multiplyExact(width, height);
                double[][] flat = new double[3][pixels];
                double[][][] planes = new double[3][][];
                for (int component = 0; component < 3; component++) {
                    System.arraycopy(
                            combined, component * pixels,
                            flat[component], 0, pixels);
                    planes[component] = reshape(
                            flat[component], width, height);
                }
                GpuStageStats.gpu(GpuStageStats.Stage.RGB_TO_YCBCR);
                return new FullPlaneData(planes, flat);
            } catch (RuntimeException | LinkageError failure) {
                if (strictGpu()) throw failure;
                GpuStageStats.fallback(GpuStageStats.Stage.RGB_TO_YCBCR, failure);
            }
        }
        double[][][] planes = ImageUtil.yCbCr(decoded);
        double[][] flat = new double[3][];
        for (int component = 0; component < 3; component++) {
            flat[component] = flatten(planes[component]);
        }
        return new FullPlaneData(planes, flat);
    }

    private static boolean gpuPreprocessEnabled() {
        return ExecutionBackend.cudaPermitted()
                && Boolean.parseBoolean(System.getProperty(
                "rap.gpu.preprocess",
                System.getenv().getOrDefault(
                        "RAP_GPU_PREPROCESS", "true")));
    }

    private static double[] flatten(double[][] plane) {
        int height = plane.length;
        int width = plane[0].length;
        double[] flat = new double[Math.multiplyExact(width, height)];
        for (int y = 0; y < height; y++) {
            System.arraycopy(plane[y], 0, flat, y * width, width);
        }
        return flat;
    }

    private static double[][] reshape(double[] flat, int width, int height) {
        if (flat.length != Math.multiplyExact(width, height)) {
            throw new IllegalArgumentException("Flat plane size mismatch");
        }
        double[][] out = new double[height][width];
        for (int y = 0; y < height; y++) {
            System.arraycopy(flat, y * width, out[y], 0, width);
        }
        return out;
    }

    private record FullPlaneData(double[][][] planes, double[][] flatPlanes) {}
    private record ComponentPlanePair(double[][] samples, double[][] padded) {}

    private CudaDctNative.Result acceleratedDctVariance(
            ComponentKey key, double[][] plane, JpegComponent component
    ) {
        if (!gpuDctEnabled()) return null;
        CudaDctNative.Result existing = gpuDctVariance.get(key);
        if (existing != null) return existing;
        if (!CudaDctNative.isAvailable()) {
            if (strictGpu()) {
                throw new IllegalStateException(
                        "Strict true-GPU mode requires CUDA DCT: "
                                + CudaDctNative.diagnostic());
            }
            return null;
        }
        try {
            CudaDctNative.Result computed = CudaDctNative.compute(
                    plane, component.widthBlocks, component.heightBlocks);
            gpuDctVariance.put(key, computed);
            GpuStageStats.gpu(GpuStageStats.Stage.DCT_VARIANCE);
            return computed;
        } catch (RuntimeException | LinkageError failure) {
            if (strictGpu()) throw failure;
            GpuStageStats.fallback(GpuStageStats.Stage.DCT_VARIANCE, failure);
            return null;
        }
    }

    private static boolean gpuDctEnabled() {
        return ExecutionBackend.cudaPermitted()
                && Boolean.parseBoolean(System.getProperty(
                "rap.gpu.dct",
                System.getenv().getOrDefault("RAP_GPU_DCT", "true")));
    }

    private static boolean strictGpu() {
        return ExecutionBackend.cudaFailureIsFatal();
    }

    public double rawDctCoefficient(
            JpegImage jpeg, int componentIndex, int block, int naturalIndex
    ) {
        if (naturalIndex < 0 || naturalIndex >= 64) {
            throw new IllegalArgumentException(
                    "natural coefficient index out of range: " + naturalIndex);
        }
        JpegComponent component = jpeg.component(componentIndex);
        if (block < 0 || block >= component.blockCount()) {
            throw new IllegalArgumentException("block index out of range: " + block);
        }
        return rawDctCoefficients(jpeg, componentIndex)[block * 64 + naturalIndex];
    }

    private static int[] allNaturalIndices() {
        int[] indices = new int[64];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        return indices;
    }

    private record ComponentKey(
            int componentIndex,
            int width,
            int height,
            int widthBlocks,
            int heightBlocks,
            int hSampling,
            int vSampling,
            int maxH,
            int maxV
    ) {
        static ComponentKey of(JpegImage jpeg, int componentIndex) {
            if (jpeg == null) throw new IllegalArgumentException("jpeg cannot be null");
            if (componentIndex < 0 || componentIndex >= jpeg.components.length) {
                throw new IllegalArgumentException(
                        "Invalid JPEG component index: " + componentIndex);
            }
            JpegComponent c = jpeg.component(componentIndex);
            return new ComponentKey(
                    componentIndex,
                    jpeg.width,
                    jpeg.height,
                    c.widthBlocks,
                    c.heightBlocks,
                    c.hSamplingFactor,
                    c.vSamplingFactor,
                    ImageUtil.maxHorizontalSampling(jpeg),
                    ImageUtil.maxVerticalSampling(jpeg));
        }
    }
}
