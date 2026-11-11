package thesis.rap.image;

import javax.imageio.*;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.spi.ImageWriterSpi;
import javax.imageio.stream.ImageOutputStream;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import org.w3c.dom.Node;
import thesis.rap.alg.JpegMath;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;
import thesis.rap.gpu.CudaCanonicalLumaNative;
import thesis.rap.runtime.ExecutionBackend;
import thesis.rap.runtime.GpuStageStats;
import java.util.Arrays;

public final class ImageUtil {
    private ImageUtil() {}
    public static BufferedImage read(Path p) throws IOException {
        BufferedImage img = ImageIO.read(p.toFile());
        if (img == null) throw new IOException("Cannot read image: " + p);
        return toRGB(img);
    }
    public static BufferedImage toRGB(BufferedImage in) {
        if (in.getType() == BufferedImage.TYPE_INT_RGB) return in;
        BufferedImage out = new BufferedImage(in.getWidth(), in.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setColor(Color.WHITE); g.fillRect(0,0,out.getWidth(),out.getHeight()); g.drawImage(in,0,0,null); g.dispose();
        return out;
    }
    public static String writeJpeg(
            BufferedImage img,
            Path out,
            int qf
    ) throws IOException {
        return writeJpeg(img, out, qf, SamplingGeometry.JPEG_420);
    }

    /**
     * Writes the structural JPEG template with an explicit supported sampling
     * geometry.  The native coefficient layer later preserves this exact
     * component layout while replacing quantization tables and coefficients.
     */
    public static String writeJpeg(
            BufferedImage img,
            Path out,
            int qf,
            SamplingGeometry geometry
    ) throws IOException {
        if (geometry == null) {
            throw new IllegalArgumentException("geometry cannot be null");
        }
        ImageWriter writer = selectJpegWriter();
        String provenance = jpegWriterDescription(writer);
        ImageWriteParam param = writer.getDefaultWriteParam();
        param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
        param.setCompressionQuality(Math.max(0f, Math.min(1f, qf / 100f)));
        BufferedImage rgb = toRGB(img);
        IIOMetadata metadata = geometryMetadata(
                writer, rgb, param, geometry);
        try (OutputStream os = new FileOutputStream(out.toFile()); ImageOutputStream ios = ImageIO.createImageOutputStream(os)) {
            writer.setOutput(ios);
            writer.write(
                    null,
                    new IIOImage(rgb, null, metadata),
                    param);
        } finally { writer.dispose(); }
        return provenance;
    }

    private static IIOMetadata geometryMetadata(
            ImageWriter writer,
            BufferedImage image,
            ImageWriteParam param,
            SamplingGeometry geometry
    ) throws IOException {
        ImageTypeSpecifier type =
                ImageTypeSpecifier.createFromRenderedImage(image);
        IIOMetadata metadata = writer.getDefaultImageMetadata(type, param);
        String format = "javax_imageio_jpeg_image_1.0";
        if (metadata == null
                || !List.of(metadata.getMetadataFormatNames())
                .contains(format)) {
            throw new IOException(
                    "Selected JPEG writer does not expose the required "
                            + format + " metadata format");
        }
        Node root = metadata.getAsTree(format);
        Node sof = firstNodeNamed(root, "sof");
        if (sof == null) {
            throw new IOException(
                    "JPEG metadata has no SOF node for geometry control");
        }
        int componentIndex = 0;
        for (Node child = sof.getFirstChild(); child != null;
             child = child.getNextSibling()) {
            if (!"componentSpec".equals(child.getNodeName())) continue;
            if (!(child instanceof IIOMetadataNode component)
                    || componentIndex >= 3) {
                throw new IOException(
                        "JPEG SOF component metadata is not a supported "
                                + "three-component tree");
            }
            component.setAttribute(
                    "HsamplingFactor",
                    String.valueOf(
                            geometry.horizontalSampling(componentIndex)));
            component.setAttribute(
                    "VsamplingFactor",
                    String.valueOf(
                            geometry.verticalSampling(componentIndex)));
            componentIndex++;
        }
        if (componentIndex != 3) {
            throw new IOException(
                    "JPEG writer exposed " + componentIndex
                            + " SOF components; exactly three are required");
        }
        metadata.setFromTree(format, root);
        return metadata;
    }

    private static Node firstNodeNamed(Node root, String name) {
        if (root == null) return null;
        if (name.equals(root.getNodeName())) return root;
        for (Node child = root.getFirstChild(); child != null;
             child = child.getNextSibling()) {
            Node found = firstNodeNamed(child, name);
            if (found != null) return found;
        }
        return null;
    }

    /** Describes the same deterministically selected template writer used above. */
    public static String jpegWriterDescription() throws IOException {
        ImageWriter writer = selectJpegWriter();
        try {
            return jpegWriterDescription(writer);
        } finally {
            writer.dispose();
        }
    }

    private static ImageWriter selectJpegWriter() throws IOException {
        Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("jpg");
        List<ImageWriter> available = new ArrayList<>();
        writers.forEachRemaining(available::add);
        if (available.isEmpty()) throw new IOException("No JPEG writer");
        available.sort(Comparator.comparing(ImageUtil::jpegWriterDescription));
        ImageWriter selected = available.remove(0);
        for (ImageWriter writer : available) writer.dispose();
        return selected;
    }

    private static String jpegWriterDescription(ImageWriter writer) {
        ImageWriterSpi provider = writer.getOriginatingProvider();
        String vendor = provider == null ? "unknown" : safe(provider.getVendorName());
        String version = provider == null ? "unknown" : safe(provider.getVersion());
        String plugin = provider == null ? writer.getClass().getName()
                : safe(provider.getPluginClassName());
        return "ImageIO JPEG writer class=" + writer.getClass().getName()
                + ", provider=" + plugin
                + ", vendor=" + vendor
                + ", version=" + version;
    }

    private static String safe(String value) {
        return value == null || value.isBlank() ? "unknown" : value;
    }
    public static double[][][] yCbCr(BufferedImage img) {
        img = toRGB(img); int w=img.getWidth(), h=img.getHeight();
        double[][] Y=new double[h][w], Cb=new double[h][w], Cr=new double[h][w];
        for(int y=0;y<h;y++) for(int x=0;x<w;x++){
            int rgb=img.getRGB(x,y); int r=(rgb>>16)&255,g=(rgb>>8)&255,b=rgb&255;
            Y[y][x]=0.299*r+0.587*g+0.114*b;
            Cb[y][x]=128-0.168736*r-0.331264*g+0.5*b;
            Cr[y][x]=128+0.5*r-0.418688*g-0.081312*b;
        }
        return new double[][][]{Y,Cb,Cr};
    }
    public static double[] histogram(double[][] plane) {
        double[] h=new double[256]; int n=plane.length*plane[0].length;
        for(double[] row: plane) for(double v: row) h[clamp((int)Math.round(v))]++;
        for(int i=0;i<256;i++) h[i]/=n; return h;
    }

    public record CanonicalLuminanceStats(
            double[][] luminance,
            double[] histogram
    ) {
        public CanonicalLuminanceStats {
            if (luminance == null || histogram == null
                    || histogram.length != 256) {
                throw new IllegalArgumentException(
                        "Canonical luminance statistics cannot be null");
            }
        }
    }

    /**
     * GPU-aware combined reconstruction used by candidate construction.  The
     * CPU method below remains the reference/parity implementation.
     */
    public static CanonicalLuminanceStats canonicalReferenceLuminanceStats(
            JpegImage jpeg
    ) {
        if (gpuCanonicalLuminanceEnabled()
                && CudaCanonicalLumaNative.isAvailable()) {
            try {
                if (jpeg == null || jpeg.components.length < 3) {
                    throw new IllegalArgumentException(
                            "RAP requires a three-component Y/Cb/Cr reference JPEG");
                }
                JpegComponent yComponent = jpeg.component(0);
                int maxH = maxHorizontalSampling(jpeg);
                int maxV = maxVerticalSampling(jpeg);
                if (yComponent.hSamplingFactor != maxH
                        || yComponent.vSamplingFactor != maxV) {
                    throw new IllegalArgumentException(
                            "RAP requires component 0 (Y) at full JPEG sampling resolution");
                }
                double[] basis = new double[64 * 64];
                for (int k = 0; k < 64; k++) {
                    int u = JpegMath.uFromNaturalIndex(k);
                    int v = JpegMath.vFromNaturalIndex(k);
                    double[][] blockBasis = JpegMath.idctBasis(
                            u, v, yComponent.quantTable[k]);
                    for (int y = 0; y < 8; y++) {
                        for (int x = 0; x < 8; x++) {
                            basis[k * 64 + y * 8 + x] = blockBasis[y][x];
                        }
                    }
                }
                double[] combined = CudaCanonicalLumaNative.reconstructAndHistogram(
                        yComponent.coefficients,
                        yComponent.widthBlocks, yComponent.heightBlocks,
                        jpeg.width, jpeg.height, basis);
                int pixels = Math.multiplyExact(jpeg.width, jpeg.height);
                double[][] luminance = new double[jpeg.height][jpeg.width];
                for (int y = 0; y < jpeg.height; y++) {
                    System.arraycopy(
                            combined, y * jpeg.width,
                            luminance[y], 0, jpeg.width);
                }
                double[] histogram = Arrays.copyOfRange(
                        combined, pixels, pixels + 256);
                GpuStageStats.gpu(GpuStageStats.Stage.CANONICAL_LUMA);
                return new CanonicalLuminanceStats(luminance, histogram);
            } catch (RuntimeException | LinkageError failure) {
                if (ExecutionBackend.cudaFailureIsFatal()) throw failure;
                GpuStageStats.fallback(GpuStageStats.Stage.CANONICAL_LUMA, failure);
            }
        }
        double[][] luminance = canonicalReferenceLuminance(jpeg);
        return new CanonicalLuminanceStats(
                luminance, histogram(luminance));
    }

    private static boolean gpuCanonicalLuminanceEnabled() {
        return ExecutionBackend.cudaPermitted()
                && Boolean.parseBoolean(System.getProperty(
                "rap.gpu.canonical.luma",
                System.getenv().getOrDefault(
                        "RAP_GPU_CANONICAL_LUMA", "true")));
    }

    /**
     * Reconstructs the full-resolution reference luminance directly from the
     * canonical quantized Y coefficients using the declared ideal inverse DCT.
     *
     * <p>This is the normative RAP histogram source. It deliberately avoids a
     * runtime JPEG-to-RGB decoder, whose IDCT approximation and colour pipeline
     * could otherwise change histogram bins and therefore the base-cost sort.</p>
     */
    public static double[][] canonicalReferenceLuminance(JpegImage jpeg) {
        if (jpeg == null || jpeg.components.length < 3) {
            throw new IllegalArgumentException(
                    "RAP requires a three-component Y/Cb/Cr reference JPEG");
        }
        JpegComponent yComponent = jpeg.component(0);
        int maxH = maxHorizontalSampling(jpeg);
        int maxV = maxVerticalSampling(jpeg);
        if (yComponent.hSamplingFactor != maxH
                || yComponent.vSamplingFactor != maxV) {
            throw new IllegalArgumentException(
                    "RAP requires component 0 (Y) at the full JPEG sampling resolution");
        }
        if (yComponent.widthBlocks * 8 < jpeg.width
                || yComponent.heightBlocks * 8 < jpeg.height) {
            throw new IllegalArgumentException(
                    "Y coefficient grid is smaller than the declared image dimensions");
        }

        double[][][] basis = new double[64][][];
        for (int k = 0; k < 64; k++) {
            int u = JpegMath.uFromNaturalIndex(k);
            int v = JpegMath.vFromNaturalIndex(k);
            basis[k] = JpegMath.idctBasis(
                    u, v, yComponent.quantTable[k]);
        }

        double[][] luminance = new double[jpeg.height][jpeg.width];
        for (int by = 0; by < yComponent.heightBlocks; by++) {
            for (int bx = 0; bx < yComponent.widthBlocks; bx++) {
                int block = by * yComponent.widthBlocks + bx;
                for (int localY = 0; localY < 8; localY++) {
                    int pixelY = by * 8 + localY;
                    if (pixelY >= jpeg.height) continue;
                    for (int localX = 0; localX < 8; localX++) {
                        int pixelX = bx * 8 + localX;
                        if (pixelX >= jpeg.width) continue;
                        double sample = 128.0;
                        for (int k = 0; k < 64; k++) {
                            sample += yComponent.getCoeff(block, k)
                                    * basis[k][localY][localX];
                        }
                        luminance[pixelY][pixelX] =
                                Math.max(0.0, Math.min(255.0, sample));
                    }
                }
            }
        }
        return luminance;
    }

    /**
     * Builds the spatial sample grid used by one JPEG component.  ImageIO exposes
     * an RGB raster, while libjpeg exposes component DCT grids.  For subsampled
     * chroma, box averaging produces one component sample for each
     * (maxH/h) x (maxV/v) footprint: 2x1 (4:2:2), 2x2 (4:2:0), 1x2 (4:4:0),
     * 4x1 (4:1:1), 1x4 (4:4:1) or 4x2 (4:1:0); 4:4:4 has unit ratios.  Edge
     * cells clamp the source coordinate.  This keeps block coordinates in the
     * same component grid as the quantized DCT coefficients.
     */
    public static double[][] jpegComponentPlane(double[][][] fullResolutionPlanes,
                                                JpegImage jpeg,
                                                int componentIndex) {
        if (componentIndex < 0 || componentIndex >= jpeg.components.length) {
            throw new IllegalArgumentException("Invalid JPEG component index: " + componentIndex);
        }
        JpegComponent component = jpeg.component(componentIndex);
        int maxH = 1;
        int maxV = 1;
        for (JpegComponent c : jpeg.components) {
            maxH = Math.max(maxH, c.hSamplingFactor);
            maxV = Math.max(maxV, c.vSamplingFactor);
        }
        if (component.hSamplingFactor <= 0
                || component.vSamplingFactor <= 0
                || maxH % component.hSamplingFactor != 0
                || maxV % component.vSamplingFactor != 0) {
            throw new IllegalArgumentException(
                    "RAP requires positive integer component sampling ratios");
        }
        int ratioX = maxH / component.hSamplingFactor;
        int ratioY = maxV / component.vSamplingFactor;
        double[][] source = fullResolutionPlanes[componentIndex];
        int outW = (source[0].length + ratioX - 1) / ratioX;
        int outH = (source.length + ratioY - 1) / ratioY;
        double[][] out = new double[outH][outW];
        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                double sum = 0.0;
                for (int dy = 0; dy < ratioY; dy++) {
                    int sy = Math.min(
                            source.length - 1, oy * ratioY + dy);
                    for (int dx = 0; dx < ratioX; dx++) {
                        int sx = Math.min(
                                source[0].length - 1,
                                ox * ratioX + dx);
                        sum += source[sy][sx];
                    }
                }
                out[oy][ox] = sum / (ratioX * (double) ratioY);
            }
        }
        return out;
    }


    /**
     * Pads a component plane to the full JPEG 8x8 block grid by repeating the
     * last valid row/column. This is the same edge completion implied by the
     * canonical forward-DCT helper for partial edge blocks.
     */
    public static double[][] padToBlockGrid(double[][] plane, int paddedWidth, int paddedHeight) {
        if (plane == null || plane.length == 0 || plane[0].length == 0) {
            throw new IllegalArgumentException("plane must be nonempty");
        }
        if (paddedWidth < plane[0].length || paddedHeight < plane.length
                || (paddedWidth & 7) != 0 || (paddedHeight & 7) != 0) {
            throw new IllegalArgumentException("invalid JPEG block-grid dimensions");
        }
        double[][] out = new double[paddedHeight][paddedWidth];
        for (int y = 0; y < paddedHeight; y++) {
            int sy = Math.min(y, plane.length - 1);
            for (int x = 0; x < paddedWidth; x++) {
                int sx = Math.min(x, plane[0].length - 1);
                out[y][x] = plane[sy][sx];
            }
        }
        return out;
    }

    public static int maxHorizontalSampling(JpegImage jpeg) {
        int max = 1;
        for (JpegComponent c : jpeg.components) max = Math.max(max, c.hSamplingFactor);
        return max;
    }

    public static int maxVerticalSampling(JpegImage jpeg) {
        int max = 1;
        for (JpegComponent c : jpeg.components) max = Math.max(max, c.vSamplingFactor);
        return max;
    }
    public static int clamp(int v){ return Math.max(0, Math.min(255,v)); }
}
