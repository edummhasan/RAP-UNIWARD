package thesis.rap.image;

import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import java.util.Arrays;

/**
 * Exact JPEG component-sampling profiles supported by the multi-geometry
 * research method.
 *
 * <p>The public labels describe the seven YCbCr chroma-subsampling geometries
 * declared for this thesis implementation. Grayscale is intentionally excluded
 * because the RAP carrier model requires Cb and Cr. These labels are never
 * interpreted as Cb:Cr payload ratios.  Cb and Cr remain one
 * joint statistical-cost pool inside each fixed geometry/QF trial.</p>
 */
public enum SamplingGeometry {
    JPEG_444("444GEOMETRY", "4:4:4", "g444",
            new int[][]{{1, 1}, {1, 1}, {1, 1}}),
    JPEG_422("422GEOMETRY", "4:2:2", "g422",
            new int[][]{{2, 1}, {1, 1}, {1, 1}}),
    JPEG_420("420GEOMETRY", "4:2:0", "g420",
            new int[][]{{2, 2}, {1, 1}, {1, 1}}),
    JPEG_440("440GEOMETRY", "4:4:0", "g440",
            new int[][]{{1, 2}, {1, 1}, {1, 1}}),
    JPEG_411("411GEOMETRY", "4:1:1", "g411",
            new int[][]{{4, 1}, {1, 1}, {1, 1}}),
    JPEG_441("441GEOMETRY", "4:4:1", "g441",
            new int[][]{{1, 4}, {1, 1}, {1, 1}}),
    JPEG_410("410GEOMETRY", "4:1:0", "g410",
            new int[][]{{4, 2}, {1, 1}, {1, 1}});

    private final String modeToken;
    private final String label;
    private final String outputTag;
    private final int[][] sampling;

    SamplingGeometry(
            String modeToken,
            String label,
            String outputTag,
            int[][] sampling
    ) {
        this.modeToken = modeToken;
        this.label = label;
        this.outputTag = outputTag;
        this.sampling = deepCopy(sampling);
    }

    public String modeToken() {
        return modeToken;
    }

    public String label() {
        return label;
    }

    public String outputTag() {
        return outputTag;
    }

    /**
     * Stable 24-bit geometry signature derived only from the six JPEG component
     * sampling factors: Y(H,V), Cb(H,V), Cr(H,V).  Each factor occupies one
     * nibble.  This is a protocol context value, not a secret and not a claim
     * that the geometry is hidden from a JPEG observer.
     */
    public int signatureCode() {
        return ((horizontalSampling(0) & 0xF) << 20)
                | ((verticalSampling(0) & 0xF) << 16)
                | ((horizontalSampling(1) & 0xF) << 12)
                | ((verticalSampling(1) & 0xF) << 8)
                | ((horizontalSampling(2) & 0xF) << 4)
                | (verticalSampling(2) & 0xF);
    }

    public String signatureHex() {
        return String.format(java.util.Locale.ROOT, "%06X", signatureCode());
    }

    public int horizontalSampling(int componentIndex) {
        requireComponentIndex(componentIndex);
        return sampling[componentIndex][0];
    }

    public int verticalSampling(int componentIndex) {
        requireComponentIndex(componentIndex);
        return sampling[componentIndex][1];
    }

    public boolean matches(JpegImage jpeg) {
        if (jpeg == null || jpeg.components.length != 3) return false;
        for (int i = 0; i < 3; i++) {
            JpegComponent component = jpeg.component(i);
            if (component.componentIndex != i
                    || component.hSamplingFactor
                    != horizontalSampling(i)
                    || component.vSamplingFactor
                    != verticalSampling(i)) {
                return false;
            }
        }
        return true;
    }

    public void require(JpegImage jpeg) {
        if (!matches(jpeg)) {
            throw new IllegalArgumentException(
                    "Geometry " + label + " requires component order Y/Cb/Cr "
                            + "with sampling " + expectedDescription()
                            + "; observed " + samplingDescription(jpeg));
        }
    }

    public static SamplingGeometry detect(JpegImage jpeg) {
        for (SamplingGeometry geometry : values()) {
            if (geometry.matches(jpeg)) return geometry;
        }
        throw new IllegalArgumentException(
                "JPEG does not match a supported geometry; observed "
                        + samplingDescription(jpeg));
    }

    public static void requireSupported(JpegImage jpeg) {
        detect(jpeg);
    }

    private String expectedDescription() {
        return "Y=" + horizontalSampling(0) + "x" + verticalSampling(0)
                + ",Cb=" + horizontalSampling(1) + "x"
                + verticalSampling(1)
                + ",Cr=" + horizontalSampling(2) + "x"
                + verticalSampling(2);
    }

    private static String samplingDescription(JpegImage jpeg) {
        if (jpeg == null) return "null";
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < jpeg.components.length; i++) {
            if (i > 0) out.append(';');
            JpegComponent component = jpeg.component(i);
            out.append(component.componentIndex)
                    .append('=')
                    .append(component.hSamplingFactor)
                    .append('x')
                    .append(component.vSamplingFactor);
        }
        return out.toString();
    }

    private static void requireComponentIndex(int componentIndex) {
        if (componentIndex < 0 || componentIndex >= 3) {
            throw new IllegalArgumentException(
                    "Sampling component index must be 0, 1, or 2: "
                            + componentIndex);
        }
    }

    private static int[][] deepCopy(int[][] source) {
        return Arrays.stream(source)
                .map(row -> Arrays.copyOf(row, row.length))
                .toArray(int[][]::new);
    }
}
