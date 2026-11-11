package thesis.rap.alg;

/**
 * Bounded aggregate evidence for the cover-derived Cb/Cr transition-cost model.
 * No stego observation enters this object, so embedding and extraction can
 * reconstruct it from the shared PNG and canonical fixed-QF reference.
 */
public record StatisticalCostSummary(
        String modelVersion,
        double regularizerStrength,
        double mseWeight,
        double ssimWeight,
        double klWeight,
        double entropyWeight,
        double mseP95Scale,
        double ssimP95Scale,
        double klP95Scale,
        double entropyP95Scale,
        Channel cb,
        Channel cr
) {
    public StatisticalCostSummary {
        if (modelVersion == null || modelVersion.isBlank()) {
            throw new IllegalArgumentException("modelVersion cannot be blank");
        }
        if (cb == null || cr == null) {
            throw new IllegalArgumentException("Both Cb and Cr summaries are required");
        }
    }

    public Channel channel(int component) {
        return switch (component) {
            case 1 -> cb;
            case 2 -> cr;
            default -> throw new IllegalArgumentException(
                    "Statistical Cb/Cr model has no component " + component);
        };
    }

    public record Channel(
            int component,
            String name,
            int universeCount,
            int dryCount,
            int plusLegalCount,
            int minusLegalCount,
            double meanBaseUniwardCost,
            double meanRgbMseEnergyProxy,
            double meanSsimLossProxy,
            double meanKlRisk,
            double meanEntropyRisk,
            double meanStatisticalMultiplier,
            double medianFinalMinCost
    ) {
        public Channel {
            if ((component != 1 && component != 2)
                    || name == null || name.isBlank()) {
                throw new IllegalArgumentException(
                        "Channel must be component 1/Cb or 2/Cr");
            }
        }
    }
}
