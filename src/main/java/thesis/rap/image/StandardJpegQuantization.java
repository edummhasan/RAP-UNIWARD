package thesis.rap.image;

/**
 * Deterministic RAP scaling of the JPEG Annex-K base tables for quality 1..100.
 *
 * <p>For Q >= 50, scale = 200 - 2Q; for Q < 50, scale = floor(5000/Q).
 * Each Annex-K base-table entry is converted as (base*scale + 50)/100 and
 * clamped to the baseline 1..255 range.</p>
 */
public final class StandardJpegQuantization {
    private StandardJpegQuantization() {}

    private static final int[] LUMA_BASE = {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
    };

    private static final int[] CHROMA_BASE = {
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99
    };

    public static int[] luminance(int quality) {
        return scaled(LUMA_BASE, quality);
    }

    public static int[] chrominance(int quality) {
        return scaled(CHROMA_BASE, quality);
    }

    public static int qualityScale(int quality) {
        int q = Math.max(1, Math.min(100, quality));
        return q < 50 ? 5000 / q : 200 - 2 * q;
    }

    private static int[] scaled(int[] base, int quality) {
        int scale = qualityScale(quality);
        int[] out = new int[64];
        for (int i = 0; i < base.length; i++) {
            long value = ((long) base[i] * scale + 50L) / 100L;
            if (value <= 0L) value = 1L;
            if (value > 255L) value = 255L;
            out[i] = (int) value;
        }
        return out;
    }
}
