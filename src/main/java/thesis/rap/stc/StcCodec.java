package thesis.rap.stc;

/** Common STC codec interface used by the RAP pipeline. */
public interface StcCodec {
    byte[] embed(byte[] coverBits, double[] costs, byte[] messageBits, int constraintHeight);
    byte[] extract(byte[] stegoBits, int messageBitLength, int constraintHeight);
}
