package thesis.rap.analysis;

import thesis.rap.alg.JpegMath;

import java.util.ArrayList;
import java.util.List;

/** Complete JPEG-domain, STC, and detector-facing report for one embedding run. */
public class JpegCoefficientReport {
    public int payloadBits;
    public int selectedQF;
    public int candidateCount;
    public int activeCarrierCount;
    /** RAP engineering capacity C_eng = candidate_count - fixed_header_reserve. */
    public int conservativeCapacity;
    public int stcReserveBits;
    public int stcConstraintHeight;

    public int totalCoefficients;
    public int totalNonzeroAC;
    public int targetBandCbCrCoefficients;
    public int targetBandCbCrNonzero;
    public int targetBandCbCrNonzeroStego;
    public int targetBandCbCrZero;
    public int targetBandCbCrZeroStego;
    public int highTailCbCrCoefficients;
    public int highTailCbCrNonzero;
    public int highTailCbCrNonzeroStego;
    public int highTailCbCrZero;
    public int highTailCbCrZeroStego;
    public int highTailModifiedCoefficients;

    public int modifiedCoefficients;
    public int modifiedY;
    public int modifiedCb;
    public int modifiedCr;
    public int plusOneChanges;
    public int minusOneChanges;
    public int otherDeltaChanges;
    public int modifiedNonzeroToNonzero;
    public int modifiedZeroToNonzero;
    public int modifiedNonzeroToZero;

    /** Conventional JPEG payload density alias: payload bits / all nonzero AC coefficients. */
    public double bpnzAC;
    /** Payload bits / eligible RAP candidate count. This is candidate loading, not bpnzAC. */
    public double candidateLoading;
    public int formerFourModeNearHalfCandidateCount;
    public int fullyWetCandidateCount;
    public double overallBpnzAC;
    public double bitsPerTotalCoefficient;
    public double candidateFractionOfNonzeroAC;
    public double candidateFractionOfAllCoefficients;
    public double localGlobalPayloadConcentration;
    /** Payload / RAP engineering capacity C_eng. */
    public double engineeringUtilization;
    /** Payload / actual cover elements passed to the native STC encoder. */
    public double activeCarrierPayloadRate;

    public double modificationRateAllCoefficients;
    public double modificationRateCandidates;
    public double modificationRateActiveCarriers;
    public double modifiedPerPayloadBit;
    public double embeddingEfficiencyBitsPerModification;
    public double cbModificationShare;
    public double crModificationShare;
    public double plusMinusImbalance;
    public double meanAbsoluteCoefficientDelta;

    // Actual native STC call dimensions and measured parity changes.
    public int stcHeaderMessageBits;
    public int stcHeaderCoverElements;
    public int stcHeaderChangedElements;
    public double stcHeaderRelativePayload;
    public double stcHeaderChangeRate;
    public int stcBodyMessageBits;
    public int stcBodyCoverElements;
    public int stcBodyChangedElements;
    public double stcBodyRelativePayload;
    public double stcBodyChangeRate;
    public int stcDistinctColumnWidthLimit;
    public int stcHeaderMaximumSubmatrixWidth;
    public int stcBodyMaximumSubmatrixWidth;
    public boolean stcDuplicateColumnRisk;
    public int stcTotalChangedElements;
    public double stcWeightedDistortion;
    public double stcRelativePayload;
    public double stcChangeRate;
    public int stcCoefficientChangeMismatch;

    // Clipped-support DCT metrics; KL uses additive count smoothing.
    public double dctHistogramL1;
    public double dctKLDivergence;

    // Exact-support global coefficient-distribution metrics (no coefficient clipping).
    public double dctExactHistogramL1;
    public double dctExactTotalVariation;
    public double dctExactKLDivergence;
    public double dctExactSymmetricKL;
    public double dctExactJensenShannon;
    public double dctExactHellinger;

    // Target-band-only distribution metrics for the configured Cb/Cr window.
    public double targetBandHistogramL1;
    public double targetBandTotalVariation;
    public double targetBandKLDivergence;
    public double targetBandSymmetricKL;
    public double targetBandJensenShannon;
    public double targetBandHellinger;

    // High-frequency control-band metrics, never used for embedding.
    public double highTailHistogramL1;
    public double highTailTotalVariation;
    public double highTailKLDivergence;
    public double highTailSymmetricKL;
    public double highTailJensenShannon;
    public double highTailHellinger;

    public boolean embeddingSuccess;
    public boolean extractionSuccess;
    public boolean aesGcmVerified;
    public boolean recoveredMessageMatches;

    /** One entry for each target-band zig-zag rank, aggregated over Cb and Cr. */
    public final List<DctPositionMetrics> positionMetrics = new ArrayList<>();
    /** Two entries: Cb and Cr target-band summaries. */
    public final List<ChannelMetrics> channelMetrics = new ArrayList<>();
    /** Sixty-four entries: 32 target ranks x Cb/Cr. */
    public final List<DctChannelPositionMetrics> channelPositionMetrics = new ArrayList<>();

    public void print() {
        System.out.println("========== JPEG COEFFICIENT / STEGANALYSIS REPORT ==========");
        System.out.printf("Selected QF*: %d%n", selectedQF);
        System.out.printf("Payload bits: %d%n", payloadBits);
        System.out.printf("Eligible candidate count: %d%n", candidateCount);
        System.out.printf("Actual STC cover-element count: %d%n", activeCarrierCount);
        System.out.printf("STC fixed header reserve: %d bits/elements%n", stcReserveBits);
        System.out.printf("RAP engineering capacity C_eng: %d bits%n", conservativeCapacity);
        System.out.printf("STC constraint height: %d%n", stcConstraintHeight);
        System.out.printf("Total coefficients: %d%n", totalCoefficients);
        System.out.printf("Total nonzero AC coefficients: %d%n", totalNonzeroAC);
        System.out.printf("Target-band Cb/Cr coefficients: %d%n", targetBandCbCrCoefficients);
        System.out.printf("Target-band nonzero cover/stego: %d / %d%n", targetBandCbCrNonzero, targetBandCbCrNonzeroStego);
        System.out.printf("Target-band zero cover/stego: %d / %d%n", targetBandCbCrZero, targetBandCbCrZeroStego);
        System.out.printf("High-tail nonzero cover/stego: %d / %d | modified=%d%n",
                highTailCbCrNonzero, highTailCbCrNonzeroStego, highTailModifiedCoefficients);
        System.out.printf("Candidate loading / bits per candidate: %.9f%n", candidateLoading);
        System.out.printf(
                "Former four-mode near-half diagnostic candidates "
                        + "(not wetted): %d%n",
                formerFourModeNearHalfCandidateCount);
        System.out.printf("Fully wet candidates: %d%n", fullyWetCandidateCount);
        System.out.printf("Overall bpnzAC / total nonzero AC: %.12f%n", overallBpnzAC);
        System.out.printf("Bits per total coefficient: %.12f%n", bitsPerTotalCoefficient);
        System.out.printf(
                "Eligible-address count / global nonzero AC: %.12f "
                        + "(not a bounded fraction; eligible addresses include zeros)%n",
                candidateFractionOfNonzeroAC);
        System.out.printf(
                "Candidate/global normalization ratio: %.6f x "
                        + "(different denominator domains)%n",
                localGlobalPayloadConcentration);
        System.out.printf("Engineering utilization / C_eng: %.9f%n", engineeringUtilization);
        System.out.printf("True STC relative payload: %.9f bits/cover-element%n", activeCarrierPayloadRate);
        System.out.printf("STC header: message=%d bits, cover=%d, relative_payload=%.9f, changes=%d, change_rate=%.9f%n",
                stcHeaderMessageBits, stcHeaderCoverElements, stcHeaderRelativePayload,
                stcHeaderChangedElements, stcHeaderChangeRate);
        System.out.printf("STC body: message=%d bits, cover=%d, relative_payload=%.9f, changes=%d, change_rate=%.9f%n",
                stcBodyMessageBits, stcBodyCoverElements, stcBodyRelativePayload,
                stcBodyChangedElements, stcBodyChangeRate);
        System.out.printf("STC matrix-width diagnostics: distinct-column limit=%d, header max width=%d, body max width=%d, duplicate-column-risk=%s%n",
                stcDistinctColumnWidthLimit, stcHeaderMaximumSubmatrixWidth,
                stcBodyMaximumSubmatrixWidth, stcDuplicateColumnRisk);
        System.out.printf("Measured STC parity changes: %d | weighted distortion: %.12g%n",
                stcTotalChangedElements, stcWeightedDistortion);
        System.out.printf("Modified coefficients: %d%n", modifiedCoefficients);
        System.out.printf("STC parity-change / coefficient-change mismatch: %d%n", stcCoefficientChangeMismatch);
        System.out.printf("Modified Y/Cb/Cr: %d / %d / %d%n", modifiedY, modifiedCb, modifiedCr);
        System.out.printf("+1 / -1 / other changes: %d / %d / %d%n", plusOneChanges, minusOneChanges, otherDeltaChanges);
        System.out.printf("Nonzero -> Nonzero: %d%n", modifiedNonzeroToNonzero);
        System.out.printf("Zero -> Nonzero: %d%n", modifiedZeroToNonzero);
        System.out.printf("Nonzero -> Zero: %d%n", modifiedNonzeroToZero);
        System.out.printf("Modification rate / all coefficients: %.12f%n", modificationRateAllCoefficients);
        System.out.printf("Modification rate / eligible candidates: %.9f%n", modificationRateCandidates);
        System.out.printf("Modification rate / actual STC cover elements: %.9f%n", modificationRateActiveCarriers);
        System.out.printf("Modified coefficients / payload bit: %.9f%n", modifiedPerPayloadBit);
        System.out.printf("Embedding efficiency: %.9f payload bits/modification%n", embeddingEfficiencyBitsPerModification);
        System.out.printf("Cb / Cr modification share: %.6f / %.6f%n", cbModificationShare, crModificationShare);
        System.out.printf("+/- change imbalance: %.9f%n", plusMinusImbalance);
        System.out.printf("Mean absolute coefficient delta: %.9f%n", meanAbsoluteCoefficientDelta);

        System.out.println("--- Global DCT distribution ---");
        System.out.printf(
                "Comparison-window diagnostic L1: %.12g | "
                        + "nonnegative stabilized KL: %.12g%n",
                dctHistogramL1, dctKLDivergence);
        System.out.printf("Exact-support L1: %.12g | TV: %.12g | KL: %.12g | symmetric KL: %.12g | JS: %.12g | Hellinger: %.12g%n",
                dctExactHistogramL1, dctExactTotalVariation, dctExactKLDivergence,
                dctExactSymmetricKL, dctExactJensenShannon, dctExactHellinger);
        System.out.println("--- Target-band "
                + JpegMath.targetBandLabel()
                + " Cb/Cr distribution ---");
        System.out.printf("L1: %.12g | TV: %.12g | KL: %.12g | symmetric KL: %.12g | JS: %.12g | Hellinger: %.12g%n",
                targetBandHistogramL1, targetBandTotalVariation, targetBandKLDivergence,
                targetBandSymmetricKL, targetBandJensenShannon, targetBandHellinger);
        System.out.println("--- High-tail control band "
                + JpegMath.highTailLabel()
                + " Cb/Cr distribution ---");
        System.out.printf("L1: %.12g | TV: %.12g | KL: %.12g | symmetric KL: %.12g | JS: %.12g | Hellinger: %.12g%n",
                highTailHistogramL1, highTailTotalVariation, highTailKLDivergence,
                highTailSymmetricKL, highTailJensenShannon, highTailHellinger);

        System.out.printf("Embedding success: %s%n", embeddingSuccess);
        System.out.printf("Extraction success: %s%n", extractionSuccess);
        System.out.printf("AES-GCM verified: %s%n", aesGcmVerified);
        System.out.printf("Recovered message matches original: %s%n", recoveredMessageMatches);
        System.out.println("============================================================");
    }
}
