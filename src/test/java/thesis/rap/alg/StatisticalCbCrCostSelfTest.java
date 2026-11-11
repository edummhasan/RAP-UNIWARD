package thesis.rap.alg;

import thesis.rap.app.ChannelAllocationEstimate;
import thesis.rap.app.StcCarrierPlan;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Regression tests for the joint fixed-QF Cb/Cr statistical cost model. */
public final class StatisticalCbCrCostSelfTest {
    private StatisticalCbCrCostSelfTest() {}

    public static void main(String[] args) throws Exception {
        testJointNormalizationAndBounds();
        testLegalCostNeverBecomesWetByScaling();
        testHandComputedMarginalRiskProxies();
        testCostDrivenAllocation();
        testSegmentedGibbsRates();
        testUndefinedRealizedShareWithoutChanges();
        System.out.println(
                "StatisticalCbCrCostSelfTest: ALL TESTS PASSED");
    }

    private static void testLegalCostNeverBecomesWetByScaling() {
        int k = JpegMath.naturalIndexFromZigzagRank(16);
        StatisticalTransitionCostModel.Draft draft =
                new StatisticalTransitionCostModel.Draft(
                        1, 0, 0, 0,
                        JpegMath.uFromNaturalIndex(k),
                        JpegMath.vFromNaturalIndex(k), k, 2,
                        7.5e12, JpegMath.WET_COST,
                        0.0, false, 1.0, 1, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        Candidate candidate = StatisticalTransitionCostModel
                .finalizeCosts(List.of(draft)).candidates().get(0);
        assert candidate.plusLegal()
                : "statistical scaling must not turn a legal direction wet";
        assert candidate.rhoPlus == JpegMath.WET_COST - 1.0
                : "a legal scaled cost must be capped immediately below wet";
        assert !candidate.minusLegal()
                : "a hard-wet direction must remain exactly wet";
        assert JpegMath.capComputedLegalCost(Double.POSITIVE_INFINITY)
                == JpegMath.WET_COST - 1.0;
    }

    private static void testHandComputedMarginalRiskProxies()
            throws Exception {
        Path png = Files.createTempFile("rap-risk-formula-", ".png");
        try {
            BufferedImage image = new BufferedImage(
                    16, 16, BufferedImage.TYPE_INT_RGB);
            if (!ImageIO.write(image, "png", png.toFile())) {
                throw new IllegalStateException("No PNG writer for test");
            }
            CandidateSet set = CandidateGenerator.generateSet(
                    png, syntheticFourTwoZeroImage());
            Candidate cb = set.universe().stream()
                    .filter(candidate -> candidate.component == 1
                            && JpegMath.zigzagRankFromNaturalIndex(
                                    candidate.k) == 16)
                    .findFirst().orElseThrow();

            double expectedCbColourEnergy = 0.344136 * 0.344136
                    + 1.772 * 1.772;
            double expectedMse = expectedCbColourEnergy * 4.0
                    / (64.0 * 3.0);
            double expectedSsim = (1.0 / 64.0)
                    / ((0.03 * 255.0) * (0.03 * 255.0));
            double epsilon = 1.0e-12;
            double denominator = 1.0 + 2.0 * epsilon;
            double high = (1.0 + epsilon) / denominator;
            double low = epsilon / denominator;
            double expectedSymmetricKl =
                    (high - low) * Math.log(high / low);

            assert close(cb.rgbMseEnergyProxy, expectedMse)
                    : "Cb RGB-MSE proxy differs from hand calculation";
            assert close(cb.ssimLossProxy, expectedSsim)
                    : "SSIM-style proxy differs from hand calculation";
            assert close(cb.klRiskPlus, expectedSymmetricKl)
                    : "one-count symmetric KL differs from hand calculation";
            assert close(cb.klRiskMinus, expectedSymmetricKl);
            assert cb.entropyRiskPlus == 0.0
                    : "moving the only count between singleton bins keeps entropy zero";
            assert cb.entropyRiskMinus == 0.0;
        } finally {
            Files.deleteIfExists(png);
        }
    }

    private static void testJointNormalizationAndBounds() {
        List<StatisticalTransitionCostModel.Draft> drafts =
                new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            double risk = i * 10.0;
            int component = i < 3 ? 1 : 2;
            int k = JpegMath.naturalIndexFromZigzagRank(16 + i);
            drafts.add(new StatisticalTransitionCostModel.Draft(
                    component, i, i, 0,
                    JpegMath.uFromNaturalIndex(k),
                    JpegMath.vFromNaturalIndex(k), k, 2,
                    10.0, 10.0, 0.0, false,
                    20.0, 1, 10.0,
                    1.0, 1.0, 1.0, 1.0,
                    risk, risk, risk, risk, risk, risk));
        }

        StatisticalTransitionCostModel.Result first =
                StatisticalTransitionCostModel.finalizeCosts(drafts);
        StatisticalTransitionCostModel.Result second =
                StatisticalTransitionCostModel.finalizeCosts(drafts);
        StatisticalCostSummary summary = first.summary();
        assert summary.modelVersion().equals(
                StatisticalTransitionCostModel.MODEL_VERSION);
        assert close(summary.mseWeight() + summary.ssimWeight()
                + summary.klWeight() + summary.entropyWeight(), 1.0);
        assert summary.cb().dryCount() == 3;
        assert summary.cr().dryCount() == 3;
        assert close(summary.mseP95Scale(), 50.0);
        assert close(first.candidates().get(0).cost, 10.0);
        assert close(first.candidates().get(5).cost, 20.0);
        for (int i = 0; i < first.candidates().size(); i++) {
            Candidate a = first.candidates().get(i);
            Candidate b = second.candidates().get(i);
            assert a.statisticalMultiplierPlus >= 1.0;
            assert a.statisticalMultiplierPlus <= 2.0;
            assert close(a.rhoPlus, b.rhoPlus);
            assert close(a.rhoMinus, b.rhoMinus);
            assert close(a.channelWeight, 1.0)
                    : "fixed Cb/Cr priors must be disabled";
        }
    }

    private static void testCostDrivenAllocation() {
        int blocks = 10;
        int k = JpegMath.naturalIndexFromZigzagRank(20);
        JpegImage image = syntheticImage(blocks, k);
        List<Candidate> active = new ArrayList<>();
        double[] costs = new double[blocks * 2];
        for (int component = 1; component <= 2; component++) {
            for (int block = 0; block < blocks; block++) {
                active.add(new Candidate(
                        component, block, block, 0,
                        JpegMath.uFromNaturalIndex(k),
                        JpegMath.vFromNaturalIndex(k), k, 2,
                        component == 1 ? 1.0 : 10.0,
                        component == 1 ? 1.0 : 10.0));
                costs[(component - 1) * blocks + block] =
                        component == 1 ? 1.0 : 10.0;
            }
        }
        image.component(1).setCoeff(0, k, 3);
        image.component(2).setCoeff(0, k, 3);
        ChannelAllocationEstimate estimate =
                ChannelAllocationEstimate.analyze(
                        active, costs, image, 8);
        assert close(estimate.activeCbShare(), 0.5);
        assert estimate.cbEntropyContributionShare() > 0.5
                : "lower Cb costs must produce a larger idealized entropy contribution";
        assert estimate.expectedCbChangeShare() > 0.5;
        assert estimate.actualModifiedCb() == 1;
        assert estimate.actualModifiedCr() == 1;
        assert close(estimate.actualModifiedCbShare(), 0.5);
    }

    private static void testUndefinedRealizedShareWithoutChanges() {
        int blocks = 2;
        int k = JpegMath.naturalIndexFromZigzagRank(20);
        JpegImage image = syntheticImage(blocks, k);
        List<Candidate> active = List.of(
                new Candidate(1, 0, 0, 0,
                        JpegMath.uFromNaturalIndex(k),
                        JpegMath.vFromNaturalIndex(k), k, 2,
                        1.0, 1.0),
                new Candidate(2, 0, 0, 0,
                        JpegMath.uFromNaturalIndex(k),
                        JpegMath.vFromNaturalIndex(k), k, 2,
                        1.0, 1.0));
        ChannelAllocationEstimate estimate =
                ChannelAllocationEstimate.analyze(
                        active, new double[]{1.0, 1.0}, image, 1);
        assert estimate.actualModifiedCb() == 0;
        assert estimate.actualModifiedCr() == 0;
        assert Double.isNaN(estimate.actualModifiedCbShare());
        assert Double.isNaN(estimate.actualModifiedCrShare());
    }

    private static void testSegmentedGibbsRates() {
        // Exactly eight body carriers for the eight body bits below, so the
        // body segment runs at full rate.  Derived from the protocol constants:


        // 64-carrier, rate-1/8 body whose lambda is correctly positive.
        int headerCarriers = thesis.rap.payload.PayloadPackage.headerBytes() * 8
                + StcCarrierPlan.DEFAULT_HEADER_RESERVE_BITS;
        int carrierCount = headerCarriers + 8;
        if ((carrierCount & 1) != 0) {
            throw new AssertionError("fixture needs an even Cb/Cr carrier count");
        }
        int blocks = carrierCount / 2;
        int k = JpegMath.naturalIndexFromZigzagRank(20);
        JpegImage image = syntheticImage(blocks, k);
        List<Candidate> active = new ArrayList<>(carrierCount);
        double[] costs = new double[carrierCount];
        for (int i = 0; i < carrierCount; i++) {
            int component = (i & 1) == 0 ? 1 : 2;
            int block = i / 2;
            double cost = component == 1 ? 1.0 : 10.0;
            active.add(new Candidate(
                    component, block, block, 0,
                    JpegMath.uFromNaturalIndex(k),
                    JpegMath.vFromNaturalIndex(k), k, 2,
                    cost, cost));
            costs[i] = cost;
        }
        StcCarrierPlan plan = StcCarrierPlan.createForHeight(
                carrierCount, thesis.rap.payload.PayloadPackage.headerBytes() * 8, 8, 10);
        ChannelAllocationEstimate estimate =
                ChannelAllocationEstimate.analyze(
                        active, costs, image, plan);
        assert estimate.model().equals(
                ChannelAllocationEstimate.MODEL);
        assert estimate.headerGibbsLambda() > 0.0
                : "header rate below one bit/carrier needs lambda > 0";
        assert estimate.bodyGibbsLambda() == 0.0
                : "full-rate body segment needs lambda = 0";
        assert close(estimate.solvedEntropyBits(),
                plan.totalMessageBits())
                : "segment-wise entropy constraints must sum to payload";
    }

    private static JpegImage syntheticImage(int blocks, int k) {
        JpegComponent[] components = new JpegComponent[3];
        int[] quant = new int[64];
        Arrays.fill(quant, 1);
        for (int component = 0; component < components.length;
             component++) {
            int[] coefficients = new int[blocks * 64];
            for (int block = 0; block < blocks; block++) {
                coefficients[block * 64 + k] = 2;
            }
            components[component] = new JpegComponent(
                    component, blocks, 1, 1, 1,
                    component == 0 ? 0 : 1,
                    coefficients, Arrays.copyOf(quant, quant.length));
        }
        return new JpegImage(blocks * 8, 8, components);
    }

    private static JpegImage syntheticFourTwoZeroImage() {
        int[] quant = new int[64];
        Arrays.fill(quant, 1);
        return new JpegImage(16, 16, new JpegComponent[]{
                new JpegComponent(0, 2, 2, 2, 2, 0,
                        new int[4 * 64],
                        Arrays.copyOf(quant, quant.length)),
                new JpegComponent(1, 1, 1, 1, 1, 1,
                        new int[64],
                        Arrays.copyOf(quant, quant.length)),
                new JpegComponent(2, 1, 1, 1, 1, 1,
                        new int[64],
                        Arrays.copyOf(quant, quant.length))
        });
    }

    private static boolean close(double a, double b) {
        return Math.abs(a - b) <= 1.0e-12
                * Math.max(1.0, Math.max(Math.abs(a), Math.abs(b)));
    }
}
