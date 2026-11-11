package thesis.rap.image;

import thesis.rap.app.QfBandCoefficientReport;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.stream.ImageInputStream;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;

/** Non-JNI regression coverage for all public color geometry-mode inputs. */
public final class SamplingGeometrySelfTest {
    private SamplingGeometrySelfTest() {}

    public static void main(String[] args) throws Exception {
        assert SamplingGeometryMode.parse("ALL_GEOMETRY")
                == SamplingGeometryMode.ALL_GEOMATRY;
        assert SamplingGeometryMode.parse("444GEOMATRY")
                == SamplingGeometryMode.GEOMETRY_444;
        assert SamplingGeometryMode.parse("422GEOMATRY")
                == SamplingGeometryMode.GEOMETRY_422;
        assert SamplingGeometryMode.parse("420GEOMATRY") == SamplingGeometryMode.GEOMETRY_420;
        assert SamplingGeometryMode.parse("440GEOMATRY") == SamplingGeometryMode.GEOMETRY_440;
        assert SamplingGeometryMode.parse("411GEOMATRY") == SamplingGeometryMode.GEOMETRY_411;
        assert SamplingGeometryMode.parse("441GEOMATRY") == SamplingGeometryMode.GEOMETRY_441;
        assert SamplingGeometryMode.parse("410GEOMATRY") == SamplingGeometryMode.GEOMETRY_410;
        assert SamplingGeometryMode.parse(null)
                == SamplingGeometryMode.ALL_GEOMATRY;
        assert SamplingGeometryMode.parse("   ")
                == SamplingGeometryMode.ALL_GEOMATRY;
        assert SamplingGeometryMode.ALL_GEOMATRY.geometries().equals(
                java.util.List.of(SamplingGeometry.values()));

        java.util.Set<Integer> signatures = new java.util.HashSet<>();
        for (SamplingGeometry geometry : SamplingGeometry.values()) {
            JpegImage exact = synthetic(geometry);
            geometry.require(exact);
            assert SamplingGeometry.detect(exact) == geometry;
            assert signatures.add(geometry.signatureCode())
                    : "duplicate geometry signature for " + geometry.label();
            assert geometry.signatureHex().length() == 6;
        }
        assertRejected(
                synthetic(SamplingGeometry.JPEG_444),
                SamplingGeometry.JPEG_420);

        verifyCandidatePopulationScaling();

        verifyWriterRealizesEveryGeometry();
        System.out.println(
                "SamplingGeometrySelfTest: ALL TESTS PASSED");
    }

    /**
     * For an exact 16x16 image and a 32-rank Cb/Cr band, 4:4:4 has four
     * chroma blocks per channel, 4:2:2 has two, and 4:2:0 has one.  The
     * resulting address counts must therefore be 256, 128, and 64.
     */
    private static void verifyCandidatePopulationScaling() {
        for (SamplingGeometry geometry : SamplingGeometry.values()) {
            JpegImage image = synthetic(geometry);
            QfBandCoefficientReport.Row row =
                    QfBandCoefficientReport.analyze(image, 1, 75);
            int chromaBlocks = image.component(1).widthBlocks
                    * image.component(1).heightBlocks
                    + image.component(2).widthBlocks
                    * image.component(2).heightBlocks;
            int expected = 32 * chromaBlocks;
            assert row.targetTotal == expected
                    : geometry.label() + " target population was "
                    + row.targetTotal + ", expected " + expected;
            assert row.samplingGeometry.equals(geometry.label());
            assert row.geometryMode.equals(geometry.modeToken());
        }
    }

    private static void verifyWriterRealizesEveryGeometry()
            throws Exception {
        Path root = Files.createTempDirectory(
                "rap-sampling-geometry-");
        try {
            BufferedImage image = new BufferedImage(
                    23, 19, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    image.setRGB(x, y, new Color(
                            (x * 11) & 255,
                            (y * 13) & 255,
                            ((x + y) * 7) & 255).getRGB());
                }
            }
            for (SamplingGeometry geometry
                    : SamplingGeometry.values()) {
                Path jpeg = root.resolve(
                        geometry.outputTag() + ".jpg");
                ImageUtil.writeJpeg(image, jpeg, 75, geometry);
                int[][] observed = readSampling(jpeg);
                for (int component = 0; component < 3; component++) {
                    assert observed[component][0]
                            == geometry.horizontalSampling(component)
                            : geometry.label() + " horizontal factor mismatch";
                    assert observed[component][1]
                            == geometry.verticalSampling(component)
                            : geometry.label() + " vertical factor mismatch";
                }
            }
        } finally {
            try (var paths = Files.walk(root)) {
                for (Path path : paths.sorted(
                        Comparator.reverseOrder()).toList()) {
                    Files.deleteIfExists(path);
                }
            }
        }
    }

    private static int[][] readSampling(Path jpeg) throws Exception {
        Iterator<ImageReader> readers =
                ImageIO.getImageReadersByFormatName("jpg");
        if (!readers.hasNext()) {
            throw new IllegalStateException("No JPEG reader");
        }
        ImageReader reader = readers.next();
        try (ImageInputStream input =
                     ImageIO.createImageInputStream(jpeg.toFile())) {
            reader.setInput(input, true, false);
            IIOMetadata metadata = reader.getImageMetadata(0);
            var root = metadata.getAsTree(
                    "javax_imageio_jpeg_image_1.0");
            var sof = firstNodeNamed(root, "sof");
            if (sof == null) {
                throw new IllegalStateException("No JPEG SOF metadata");
            }
            int[][] result = new int[3][2];
            int index = 0;
            for (var child = sof.getFirstChild(); child != null;
                 child = child.getNextSibling()) {
                if (!"componentSpec".equals(child.getNodeName())) {
                    continue;
                }
                if (index >= 3) {
                    throw new IllegalStateException(
                            "More than three JPEG components");
                }
                var attributes = child.getAttributes();
                result[index][0] = Integer.parseInt(
                        attributes.getNamedItem(
                                "HsamplingFactor").getNodeValue());
                result[index][1] = Integer.parseInt(
                        attributes.getNamedItem(
                                "VsamplingFactor").getNodeValue());
                index++;
            }
            if (index != 3) {
                throw new IllegalStateException(
                        "Expected three JPEG components, observed " + index);
            }
            return result;
        } finally {
            reader.dispose();
        }
    }

    private static org.w3c.dom.Node firstNodeNamed(
            org.w3c.dom.Node root,
            String name
    ) {
        if (root == null) return null;
        if (name.equals(root.getNodeName())) return root;
        for (var child = root.getFirstChild(); child != null;
             child = child.getNextSibling()) {
            var found = firstNodeNamed(child, name);
            if (found != null) return found;
        }
        return null;
    }

    private static void assertRejected(
            JpegImage image,
            SamplingGeometry required
    ) {
        boolean rejected = false;
        try {
            required.require(image);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected : "mismatched geometry must be rejected";
    }

    private static JpegImage synthetic(SamplingGeometry geometry) {
        int[] quant = new int[64];
        Arrays.fill(quant, 1);
        JpegComponent[] components = new JpegComponent[3];
        int width = 16;
        int height = 16;
        int maxH = geometry.horizontalSampling(0);
        int maxV = geometry.verticalSampling(0);
        for (int component = 0; component < 3; component++) {
            int h = geometry.horizontalSampling(component);
            int v = geometry.verticalSampling(component);
            int blocksWide = (width * h + 8 * maxH - 1)
                    / (8 * maxH);
            int blocksHigh = (height * v + 8 * maxV - 1)
                    / (8 * maxV);
            components[component] = new JpegComponent(
                    component, blocksWide, blocksHigh,
                    h, v, component == 0 ? 0 : 1,
                    new int[blocksWide * blocksHigh * 64],
                    Arrays.copyOf(quant, quant.length));
        }
        return new JpegImage(width, height, components);
    }
}
