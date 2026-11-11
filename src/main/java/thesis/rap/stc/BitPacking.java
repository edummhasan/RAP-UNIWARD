package thesis.rap.stc;

import java.util.Objects;

public final class BitPacking {
    private BitPacking() {}

    public static byte[] bytesToBits(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        byte[] bits = new byte[Math.multiplyExact(bytes.length, 8)];
        for (int i = 0; i < bits.length; i++) {
            bits[i] = (byte) ((bytes[i / 8] >> (7 - (i % 8))) & 1);
        }
        return bits;
    }

    public static byte[] bitsToBytes(byte[] bits) {
        Objects.requireNonNull(bits, "bits");
        int n = Math.addExact(bits.length, 7) / 8;
        byte[] out = new byte[n];
        for (int i = 0; i < bits.length; i++) {
            if (bits[i] != 0 && bits[i] != 1) {
                throw new IllegalArgumentException(
                        "bits contains a non-binary value at index " + i);
            }
            out[i / 8] |= (byte) ((bits[i] & 1) << (7 - (i % 8)));
        }
        return out;
    }
}
