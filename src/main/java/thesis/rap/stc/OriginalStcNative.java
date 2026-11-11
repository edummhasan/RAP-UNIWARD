package thesis.rap.stc;

final class OriginalStcNative {
    static {
        System.loadLibrary("originalstcjni");
    }

    private OriginalStcNative() {}

    static native byte[] embedBinary(
            byte[] coverBits,
            double[] costs,
            byte[] messageBits,
            int constraintHeight
    );

    static native byte[] extractBinary(
            byte[] stegoBits,
            int messageBitLength,
            int constraintHeight
    );
}
