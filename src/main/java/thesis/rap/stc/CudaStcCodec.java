package thesis.rap.stc;

import thesis.rap.gpu.CudaStcNative;

import java.util.Objects;

/**
 * CUDA implementation of the exact binary64 STC recurrence used by
 * {@link OriginalStcCodec}.  Only the thesis-declared heights 10 and 12 are
 * accepted so the GPU kernel can keep both price vectors in shared memory.
 */
public final class CudaStcCodec implements StcCodec {
    public static final String BACKEND_PROFILE =
            "CUDA_BINARY64_STC_TRUE_PARALLEL_V1";

    public static boolean isAvailable() {
        return CudaStcNative.isAvailable();
    }

    @Override
    public byte[] embed(
            byte[] coverBits,
            double[] costs,
            byte[] messageBits,
            int constraintHeight) {
        validateHeight(constraintHeight);
        validateBits(coverBits, "coverBits");
        validateBits(messageBits, "messageBits");
        Objects.requireNonNull(costs, "costs");
        if (coverBits.length != costs.length) {
            throw new IllegalArgumentException(
                    "coverBits and costs must have the same length.");
        }
        if (messageBits.length == 0 || messageBits.length > coverBits.length) {
            throw new IllegalArgumentException(
                    "messageBits length must be in 1..coverBits.length.");
        }
        if (messageBits.length < constraintHeight) {
            throw new IllegalArgumentException(
                    "messageBits length must be at least constraintHeight.");
        }
        for (int i = 0; i < costs.length; i++) {
            double cost = costs[i];
            if (!Double.isFinite(cost) || cost < 0.0
                    || cost > OriginalStcCodec.MAX_COST) {
                throw new IllegalArgumentException(
                        "costs contains a value outside [0, 1e13] at index " + i);
            }
        }
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "CUDA STC backend requested but unavailable: "
                            + CudaStcNative.diagnostic());
        }
        byte[] stego = CudaStcNative.embedBinary(
                coverBits, costs, messageBits, constraintHeight);
        validateBits(stego, "CUDA stegoBits");
        if (stego.length != coverBits.length) {
            throw new IllegalStateException(
                    "CUDA STC returned a stego vector with the wrong length.");
        }
        return stego;
    }

    @Override
    public byte[] extract(
            byte[] stegoBits,
            int messageBitLength,
            int constraintHeight) {
        validateHeight(constraintHeight);
        validateBits(stegoBits, "stegoBits");
        if (messageBitLength <= 0 || messageBitLength > stegoBits.length) {
            throw new IllegalArgumentException(
                    "messageBitLength must be in 1..stegoBits.length.");
        }
        if (messageBitLength < constraintHeight) {
            throw new IllegalArgumentException(
                    "messageBitLength must be at least constraintHeight.");
        }
        if (!isAvailable()) {
            throw new IllegalStateException(
                    "CUDA STC backend requested but unavailable: "
                            + CudaStcNative.diagnostic());
        }
        byte[] message = CudaStcNative.extractBinary(
                stegoBits, messageBitLength, constraintHeight);
        validateBits(message, "CUDA messageBits");
        if (message.length != messageBitLength) {
            throw new IllegalStateException(
                    "CUDA STC returned a message vector with the wrong length.");
        }
        return message;
    }

    private static void validateHeight(int height) {
        if (height != 10 && height != 12) {
            throw new IllegalArgumentException(
                    "RAP-UNIWARD CUDA STC supports the thesis-declared heights 10 and 12 only.");
        }
    }

    private static void validateBits(byte[] bits, String name) {
        Objects.requireNonNull(bits, name);
        for (int i = 0; i < bits.length; i++) {
            if (bits[i] != 0 && bits[i] != 1) {
                throw new IllegalArgumentException(
                        name + " contains non-binary value at index " + i);
            }
        }
    }
}
