package thesis.rap.alg;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Separates the value-independent fixed-band universe from the reproducible dry
 * set supplied to STC.
 */
public record CandidateSet(
        List<Candidate> universe,
        List<Candidate> dryBaseCostSorted,
        StatisticalCostSummary statisticalCostSummary
) {
    /** Compatibility constructor for synthetic tests that do not run the model. */
    public CandidateSet(
            List<Candidate> universe,
            List<Candidate> dryBaseCostSorted
    ) {
        this(universe, dryBaseCostSorted,
                compatibilitySummary(universe, dryBaseCostSorted));
    }

    public CandidateSet {
        if (universe == null || dryBaseCostSorted == null
                || statisticalCostSummary == null) {
            throw new IllegalArgumentException("Candidate lists cannot be null");
        }
        universe = Collections.unmodifiableList(new ArrayList<>(universe));
        dryBaseCostSorted = Collections.unmodifiableList(new ArrayList<>(dryBaseCostSorted));
        Set<Candidate> universeIdentity =
                Collections.newSetFromMap(new IdentityHashMap<>());
        long[] universeAddresses = new long[universe.size()];
        int addressIndex = 0;
        for (Candidate candidate : universe) {
            if (candidate == null) {
                throw new IllegalArgumentException(
                        "Candidate universe contains null");
            }
            int rank = JpegMath.zigzagRankFromNaturalIndex(candidate.k);
            if ((candidate.component != 1 && candidate.component != 2)
                    || !JpegMath.isTargetExpandedMidBandRank(rank)
                    || candidate.blockIndex < 0
                    || candidate.blockX < 0
                    || candidate.blockY < 0) {
                throw new IllegalArgumentException(
                        "Candidate lies outside the fixed Cb/Cr "
                                + JpegMath.targetBandLabel() + " universe");
            }
            long address = ((long) candidate.component << 37)
                    | ((long) candidate.blockIndex << 6) | candidate.k;
            universeAddresses[addressIndex++] = address;
            if (!universeIdentity.add(candidate)) {
                throw new IllegalArgumentException(
                        "Candidate universe contains duplicate object identities");
            }
        }
        Arrays.sort(universeAddresses);
        for (int i = 1; i < universeAddresses.length; i++) {
            if (universeAddresses[i - 1] == universeAddresses[i]) {
                throw new IllegalArgumentException(
                        "Candidate universe contains duplicate address " + universeAddresses[i]);
            }
        }
        for (int i = 1; i < dryBaseCostSorted.size(); i++) {
            if (dryBaseCostSorted.get(i - 1).compareTo(dryBaseCostSorted.get(i)) > 0) {
                throw new IllegalArgumentException("Dry candidates are not in the frozen base-cost order");
            }
        }
        for (Candidate candidate : dryBaseCostSorted) {
            if (!universeIdentity.remove(candidate)) {
                throw new IllegalArgumentException(
                        "Dry candidate is duplicated or not a member of the fixed universe");
            }
            if (candidate.fullyWet()) {
                throw new IllegalArgumentException("Fully wet address entered the dry candidate set");
            }
        }
        for (Candidate candidate : universeIdentity) {
            if (!candidate.fullyWet()) {
                throw new IllegalArgumentException(
                        "Legal universe address is missing from the dry set");
            }
        }
    }

    public int universeCount() {
        return universe.size();
    }

    public int dryCount() {
        return dryBaseCostSorted.size();
    }

    public int fullyWetCount() {
        return (int) universe.stream().filter(Candidate::fullyWet).count();
    }

    private static StatisticalCostSummary compatibilitySummary(
            List<Candidate> universe,
            List<Candidate> dry
    ) {
        int cbUniverse = countComponent(universe, 1);
        int crUniverse = countComponent(universe, 2);
        int cbDry = countComponent(dry, 1);
        int crDry = countComponent(dry, 2);
        return new StatisticalCostSummary(
                "UNMODELED_TEST_FIXTURE", 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                new StatisticalCostSummary.Channel(
                        1, "Cb", cbUniverse, cbDry, 0, 0,
                        Double.NaN, Double.NaN, Double.NaN,
                        Double.NaN, Double.NaN, 1.0, Double.NaN),
                new StatisticalCostSummary.Channel(
                        2, "Cr", crUniverse, crDry, 0, 0,
                        Double.NaN, Double.NaN, Double.NaN,
                        Double.NaN, Double.NaN, 1.0, Double.NaN));
    }

    private static int countComponent(
            List<Candidate> candidates,
            int component
    ) {
        if (candidates == null) return 0;
        int count = 0;
        for (Candidate candidate : candidates) {
            if (candidate != null && candidate.component == component) count++;
        }
        return count;
    }
}
