package thesis.rap.stc;

import java.util.Objects;

/** Java wrapper for the audited binary64 adaptation of the bundled STC construction. */
public final class OriginalStcCodec implements StcCodec {
    public static final String NUMERIC_PROFILE =
            "PORTABLE_SCALAR_BINARY64_STC_V1";
    public static final double MAX_COST = 1.0e13;

    @Override
    public byte[] embed(byte[] coverBits, double[] costs, byte[] messageBits, int constraintHeight) {
        validateBits(coverBits, "coverBits");
        validateBits(messageBits, "messageBits");
        Objects.requireNonNull(costs, "costs");

        if (coverBits.length != costs.length) {
            throw new IllegalArgumentException("coverBits and costs must have the same length.");
        }
        for (int i = 0; i < costs.length; i++) {
            if (!Double.isFinite(costs[i]) || costs[i] < 0.0
                    || costs[i] > MAX_COST) {
                throw new IllegalArgumentException(
                        "costs contains a value outside [0, 1e13] at index "
                                + i);
            }
        }
        if (messageBits.length == 0) {
            throw new IllegalArgumentException("messageBits cannot be empty.");
        }
        if (messageBits.length > coverBits.length) {
            throw new IllegalArgumentException(
                    "messageBits cannot be longer than coverBits.");
        }
        if (messageBits.length < constraintHeight) {
            throw new IllegalArgumentException(
                    "messageBits length must be at least constraintHeight.");
        }
        if (constraintHeight < 4 || constraintHeight > 20) {
            throw new IllegalArgumentException("constraintHeight should normally be between 4 and 20.");
        }

        byte[] stego = OriginalStcNative.embedBinary(
                coverBits, costs, messageBits, constraintHeight);
        validateBits(stego, "native stegoBits");
        if (stego.length != coverBits.length) {
            throw new IllegalStateException(
                    "Native STC returned a stego vector with the wrong length");
        }
        return stego;
    }

    @Override
    public byte[] extract(byte[] stegoBits, int messageBitLength, int constraintHeight) {
        validateBits(stegoBits, "stegoBits");

        if (messageBitLength <= 0) {
            throw new IllegalArgumentException("messageBitLength must be positive.");
        }
        if (messageBitLength > stegoBits.length) {
            throw new IllegalArgumentException(
                    "messageBitLength cannot exceed stegoBits length.");
        }
        if (messageBitLength < constraintHeight) {
            throw new IllegalArgumentException(
                    "messageBitLength must be at least constraintHeight.");
        }
        if (constraintHeight < 4 || constraintHeight > 20) {
            throw new IllegalArgumentException("constraintHeight should normally be between 4 and 20.");
        }

        byte[] message = OriginalStcNative.extractBinary(
                stegoBits, messageBitLength, constraintHeight);
        validateBits(message, "native messageBits");
        if (message.length != messageBitLength) {
            throw new IllegalStateException(
                    "Native STC returned a message vector with the wrong length");
        }
        return message;
    }

    private static void validateBits(byte[] bits, String name) {
        Objects.requireNonNull(bits, name);
        for (int i = 0; i < bits.length; i++) {
            if (bits[i] != 0 && bits[i] != 1) {
                throw new IllegalArgumentException(name + " contains non-binary value at index " + i);
            }
        }
    }
}
