package thesis.rap.analysis;

/** Auditable adaptive-QF selection margin for one successful run. */
public record QfSelectionMetrics(
        int qfsTested,
        int failedQfsBeforeSelection,
        Integer previousTestedQf,
        int selectedCandidateCount,
        int candidateHeadroomBits,
        int engineeringCapacityBits,
        int capacityHeadroomBits,
        double candidateHeadroomPercentOfPayload,
        double capacityHeadroomPercentOfPayload
) {}
