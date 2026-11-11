package thesis.rap.app;

/**
 * Small testable boundary around the STC implementation used by the synchronized
 * header/body embedding loop.
 */
public interface StcEmbeddingBackend {
    /** Constraint height used to construct the reproducible carrier plan. */
    default int constraintHeight() {
        return StcCarrierPlan.DEFAULT_STC_HEIGHT;
    }

    StcEmbedResult embedPayloadWithTrace(byte[] coverBits, double[] costs, byte[] payloadBytes);
    byte[] extractPayload(byte[] stegoBits, int payloadByteLength);
}
