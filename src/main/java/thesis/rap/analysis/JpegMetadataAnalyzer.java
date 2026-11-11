package thesis.rap.analysis;

import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.HexFormat;

/** Extracts stable JPEG metadata from the coefficient object returned by native libjpeg. */
public final class JpegMetadataAnalyzer {
    private JpegMetadataAnalyzer() {}

    public static JpegMetadata analyze(JpegImage image) {
        int maxH = 1;
        int maxV = 1;
        StringBuilder sampling = new StringBuilder();
        StringBuilder qIndexes = new StringBuilder();
        for (int i = 0; i < image.components.length; i++) {
            JpegComponent c = image.components[i];
            maxH = Math.max(maxH, c.hSamplingFactor);
            maxV = Math.max(maxV, c.vSamplingFactor);
            if (i > 0) { sampling.append(';'); qIndexes.append(';'); }
            sampling.append(componentName(i)).append(':').append(c.hSamplingFactor).append('x').append(c.vSamplingFactor);
            qIndexes.append(componentName(i)).append(':').append(c.quantTableIndex);
        }
        int mcuWidth = 8 * maxH;
        int mcuHeight = 8 * maxV;
        int mcuCols = (image.width + mcuWidth - 1) / mcuWidth;
        int mcuRows = (image.height + mcuHeight - 1) / mcuHeight;

        return new JpegMetadata(
                image.width,
                image.height,
                image.components.length,
                subsamplingLabel(image, maxH, maxV),
                maxH,
                maxV,
                mcuCols,
                mcuRows,
                (long) mcuCols * mcuRows,
                blockCount(image, 0),
                blockCount(image, 1),
                blockCount(image, 2),
                sampling.toString(),
                qIndexes.toString(),
                quantTableHash(image)
        );
    }

    private static int blockCount(JpegImage image, int component) {
        return component < image.components.length ? image.components[component].blockCount() : 0;
    }

    private static String subsamplingLabel(JpegImage image, int maxH, int maxV) {
        if (image.components.length < 3) return image.components.length == 1 ? "grayscale" : "unknown";
        JpegComponent cb = image.components[1];
        JpegComponent cr = image.components[2];
        if (cb.hSamplingFactor != cr.hSamplingFactor || cb.vSamplingFactor != cr.vSamplingFactor) return "nonstandard";
        int hRatio = maxH / Math.max(1, cb.hSamplingFactor);
        int vRatio = maxV / Math.max(1, cb.vSamplingFactor);
        if (hRatio == 1 && vRatio == 1) return "4:4:4";
        if (hRatio == 2 && vRatio == 1) return "4:2:2";
        if (hRatio == 2 && vRatio == 2) return "4:2:0";
        if (hRatio == 1 && vRatio == 2) return "4:4:0";
        return "ratio-" + hRatio + "x" + vRatio;
    }

    private static String quantTableHash(JpegImage image) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            ByteBuffer bb = ByteBuffer.allocate(4);
            for (JpegComponent c : image.components) {
                md.update((byte) c.quantTableIndex);
                for (int v : c.quantTable) {
                    bb.clear();
                    bb.putInt(v);
                    md.update(bb.array());
                }
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (Exception e) {
            throw new IllegalStateException("Unable to hash JPEG quantization tables", e);
        }
    }

    private static String componentName(int i) {
        return switch (i) { case 0 -> "Y"; case 1 -> "Cb"; case 2 -> "Cr"; default -> "C" + i; };
    }
}
