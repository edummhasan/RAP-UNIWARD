package thesis.rap.app;

/**
 * Regression proof that final QF selection is evaluated independently for each
 * payload size. Capacity values here are controlled fixtures, not research data.
 */
public final class MessageSpecificQfSelectionSelfTest {
    private MessageSpecificQfSelectionSelfTest() {}

    public static void main(String[] args) throws Exception {
        int smallPayloadBits = 900;
        int largePayloadBits = 1500;

        AdaptiveQfSearch.Result<String> small = selectForPayload(smallPayloadBits);
        AdaptiveQfSearch.Result<String> large = selectForPayload(largePayloadBits);

        assert small.selectedQf() == 80 : small.selectedQf();
        assert large.selectedQf() == 90 : large.selectedQf();
        assert !small.selectedQf().equals(large.selectedQf());

        // Repeating the smaller payload after the larger one must reproduce its
        // own QF rather than reuse the previous message's selected QF.
        AdaptiveQfSearch.Result<String> smallAgain = selectForPayload(smallPayloadBits);
        assert smallAgain.selectedQf().equals(small.selectedQf());

        System.out.println("MessageSpecificQfSelectionSelfTest: ALL TESTS PASSED");
    }

    private static AdaptiveQfSearch.Result<String> selectForPayload(int requiredBits)
            throws Exception {
        return AdaptiveQfSearch.firstVerified(75, 100, qf -> {
            // Synthetic monotonic fixture: 60 additional bits per QF.
            int engineeringCapacity = 600 + (qf - 75) * 60;
            if (engineeringCapacity < requiredBits) {
                return AdaptiveQfSearch.Trial.capacityRejected(
                        qf, "controlled_fixture_capacity");
            }
            return AdaptiveQfSearch.Trial.verified(
                    qf, "payload_bits=" + requiredBits);
        });
    }
}
