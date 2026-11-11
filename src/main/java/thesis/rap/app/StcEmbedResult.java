package thesis.rap.app;

import java.util.Arrays;

/** Stego parity vector plus actual measured STC operation statistics. */
public record StcEmbedResult(byte[] stegoBits, StcEmbeddingTrace trace) {
    public StcEmbedResult {
        stegoBits = Arrays.copyOf(stegoBits, stegoBits.length);
    }

    @Override public byte[] stegoBits() {
        return Arrays.copyOf(stegoBits, stegoBits.length);
    }

    /** Package-private zero-copy view for the validated in-process pipeline. */
    byte[] stegoBitsView() {
        return stegoBits;
    }
}
