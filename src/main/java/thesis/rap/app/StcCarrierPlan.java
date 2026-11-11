package thesis.rap.app;

/**
 * Deterministic carrier allocation used by both embedding and extraction.
 *
 * <p>The fixed header reserve gives the header STC code redundancy while keeping
 * the capacity rule exactly reproducible: payloadBits <= candidateCount - reserveBits.</p>
 */
public record StcCarrierPlan(
        int constraintHeight,
        int bodyWidthCap,
        int candidateCount,
        int reserveBits,
        int headerMessageBits,
        int headerCarrierCount,
        int bodyMessageBits,
        int bodyCarrierCount,
        int totalMessageBits,
        int totalCarrierCount,
        int engineeringMessageCapacityBits,
        int capacityHeadroomBits
) {
    public static final int DEFAULT_HEADER_RESERVE_BITS = 32;
    public static final int DEFAULT_STC_HEIGHT = 10;
    public static final int RESEARCH_STC_HEIGHT = 12;
    /** Fixed research-control cap derived once from the height-10 baseline. */
    public static final int BODY_WIDTH_CAP = 1 << (DEFAULT_STC_HEIGHT - 2);

    public static StcCarrierPlan create(int candidateCount, int headerMessageBits, int bodyMessageBits) {
        return createForHeight(candidateCount, headerMessageBits,
                bodyMessageBits, DEFAULT_STC_HEIGHT,
                DEFAULT_HEADER_RESERVE_BITS);
    }

    public static StcCarrierPlan create(int candidateCount, int headerMessageBits, int bodyMessageBits, int reserveBits) {
        return createForHeight(candidateCount, headerMessageBits,
                bodyMessageBits, DEFAULT_STC_HEIGHT, reserveBits);
    }

    public static StcCarrierPlan createForHeight(
            int candidateCount,
            int headerMessageBits,
            int bodyMessageBits,
            int constraintHeight
    ) {
        return createForHeight(candidateCount, headerMessageBits,
                bodyMessageBits, constraintHeight,
                DEFAULT_HEADER_RESERVE_BITS);
    }

    public static StcCarrierPlan createForHeight(
            int candidateCount,
            int headerMessageBits,
            int bodyMessageBits,
            int constraintHeight,
            int reserveBits
    ) {
        validateSupportedHeight(constraintHeight);
        if (candidateCount < 0 || headerMessageBits <= 0 || bodyMessageBits < 0 || reserveBits < 0) {
            throw new IllegalArgumentException("Invalid STC carrier-plan dimensions.");
        }
        int totalMessageBits = Math.addExact(headerMessageBits, bodyMessageBits);
        int capacity = Math.max(0, candidateCount - reserveBits);
        if (totalMessageBits > capacity) {
            throw new IllegalArgumentException("Payload exceeds RAP engineering capacity C_eng: payload=" + totalMessageBits
                    + ", capacity=" + capacity + ", candidates=" + candidateCount + ", reserve=" + reserveBits);
        }

        int headerCarrierCount = Math.addExact(headerMessageBits, reserveBits);
        if (headerCarrierCount > candidateCount) {
            throw new IllegalArgumentException("Insufficient candidates for the reserved header carrier pool.");
        }
        int availableBodyCandidates = candidateCount - headerCarrierCount;
        int bodyWidthCap = bodyWidthCap(constraintHeight);
        long cappedWidth = Math.multiplyExact(
                (long) bodyMessageBits, (long) bodyWidthCap);
        int bodyCarrierCount = (int) Math.min((long) availableBodyCandidates, cappedWidth);
        if (bodyMessageBits > bodyCarrierCount) {
            throw new IllegalArgumentException("Insufficient body carriers after reserving header redundancy.");
        }
        int totalCarrierCount = Math.addExact(headerCarrierCount, bodyCarrierCount);

        return new StcCarrierPlan(
                constraintHeight,
                bodyWidthCap,
                candidateCount,
                reserveBits,
                headerMessageBits,
                headerCarrierCount,
                bodyMessageBits,
                bodyCarrierCount,
                totalMessageBits,
                totalCarrierCount,
                capacity,
                capacity - totalMessageBits
        );
    }

    
    public static void validateSupportedHeight(int constraintHeight) {
        if (constraintHeight != DEFAULT_STC_HEIGHT
                && constraintHeight != RESEARCH_STC_HEIGHT) {
            throw new IllegalArgumentException(
                    "Supported STC heights are 10 (baseline) and 12 "
                            + "(research); received " + constraintHeight);
        }
    }

    public static int bodyWidthCap(int constraintHeight) {
        validateSupportedHeight(constraintHeight);
        // This is an independent engineering throttle, not an STC theorem.
        // Keeping it fixed removes carrier-capacity as one h=10/h=12
        // confounder.  A controlled comparison must separately control or
        // model payload randomness, runtime conditions, and repeated runs.
        return BODY_WIDTH_CAP;
    }

    /**
     * Exact byte count of the native backtracking path allocation for one STC
     * call: n * (2^h / 32) * sizeof(uint32_t) = n * 2^(h-3).
     * Other trellis arrays and JVM/native overhead are not included.
     */
    public static long nativePathBytesEstimate(
            int carrierCount,
            int constraintHeight
    ) {
        if (carrierCount < 0) {
            throw new IllegalArgumentException(
                    "carrierCount cannot be negative");
        }
        validateSupportedHeight(constraintHeight);
        return Math.multiplyExact(
                (long) carrierCount, 1L << (constraintHeight - 3));
    }

    /**
     * Sum of the exact backtracking-path allocations for the sequential header
     * and body calls. This is a conservative aggregate traffic/allocation
     * figure, not the simultaneous peak, because the two native calls do not
     * retain their path arrays at the same time.
     */
    public long nativePathBytesEstimate() {
        return nativePathBytesEstimate(
                totalCarrierCount, constraintHeight);
    }

    /** Exact larger of the two sequential header/body path allocations. */
    public long nativePeakPathBytesEstimate() {
        return nativePathBytesEstimate(
                Math.max(headerCarrierCount, bodyCarrierCount),
                constraintHeight);
    }

    /**
     * Exact payload-independent storage for the active and next binary64
     * Viterbi metric arrays: 2 arrays * 2^h states * 8 bytes.
     * Vector/container bookkeeping and matrix arrays are not included.
     */
    public static long nativeMetricBytesEstimate(int constraintHeight) {
        validateSupportedHeight(constraintHeight);
        return 1L << (constraintHeight + 4);
    }

    public long nativeMetricBytesEstimate() {
        return nativeMetricBytesEstimate(constraintHeight);
    }

    public double overallRelativePayload() {
        return totalCarrierCount == 0 ? Double.NaN : totalMessageBits / (double) totalCarrierCount;
    }

    public double headerRelativePayload() {
        return headerCarrierCount == 0 ? Double.NaN : headerMessageBits / (double) headerCarrierCount;
    }

    public double bodyRelativePayload() {
        return bodyCarrierCount == 0 ? Double.NaN : bodyMessageBits / (double) bodyCarrierCount;
    }
}
