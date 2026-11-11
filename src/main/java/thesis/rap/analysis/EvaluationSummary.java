package thesis.rap.analysis;

import thesis.rap.alg.JpegMath;

public class EvaluationSummary {
    public ImageComparisonMetrics pngVsStego;
    public ImageComparisonMetrics referenceVsStego;
    public JpegMetadata jpegMetadata;
    public JpegCoefficientReport coefficientReport;

    public void print() {
        System.out.println();
        System.out.println("#########################################################");
        System.out.println("########## COMPLETE RAP-UNIWARD EVALUATION LOG ##########");
        System.out.println("#########################################################");
        if (pngVsStego != null) {
            pngVsStego.print();
            System.out.println("NOTE: PNG-to-stego file-size difference is informational only because the formats differ.");
        }
        if (referenceVsStego != null) referenceVsStego.print();
        if (coefficientReport != null) coefficientReport.print();
        printInterpretation();
        System.out.println("#########################################################");
        System.out.println();
    }

    private void printInterpretation() {
        if (coefficientReport == null) return;
        JpegCoefficientReport c = coefficientReport;

        System.out.println("========== AUTOMATIC DETECTOR-ORIENTED INTERPRETATION ==========");
        System.out.printf("Adaptive QF selected: QF*=%d.%n", c.selectedQF);

        if (referenceVsStego != null) {
            System.out.printf(
                    "Reference-stego screening measurements: "
                            + "PSNR=%.9f dB; SSIM=%.12f. "
                            + "These are fidelity measurements, not security grades.%n",
                    referenceVsStego.psnr, referenceVsStego.ssim);
        }

        System.out.printf("Global payload density: %.9f bpnzAC; candidate loading: %.9f bits/candidate.%n",
                c.overallBpnzAC, c.candidateLoading);
        System.out.printf(
                "Candidate/global normalization ratio: %.3f x "
                        + "(different denominator domains).%n",
                c.localGlobalPayloadConcentration);
        System.out.printf("True native-STC relative payload: %.6f bits per cover element; measured embedding efficiency: %.6f payload bits per coefficient modification.%n",
                c.activeCarrierPayloadRate, c.embeddingEfficiencyBitsPerModification);
        System.out.printf("Global modification rate: %.9f; eligible-candidate modification rate: %.9f.%n",
                c.modificationRateAllCoefficients, c.modificationRateCandidates);
        System.out.printf("Smoothed global DCT L1/KL: %.12g / %.12g.%n",
                c.dctHistogramL1, c.dctKLDivergence);
        System.out.printf("Exact-support global JS/KL: %.12g / %.12g.%n",
                c.dctExactJensenShannon, c.dctExactKLDivergence);
        System.out.printf("Target-band JS divergence: %.12g; target-band KL divergence: %.12g.%n",
                c.targetBandJensenShannon, c.targetBandKLDivergence);
        System.out.printf("High-tail control-band JS divergence: %.12g; modified high-tail coefficients: %d.%n",
                c.highTailJensenShannon, c.highTailModifiedCoefficients);

        if (c.activeCarrierPayloadRate >= 0.95) {
            System.out.println("Carrier-pool warning: the actual native STC cover pool is operating near one payload bit per cover element; coding redundancy is limited in this run.");
        }
        if (c.stcDuplicateColumnRisk) {
            System.out.println("STC matrix warning: at least one segment requests a submatrix width above the distinct-column limit for the configured constraint height.");
        }
        if (c.formerFourModeNearHalfCandidateCount > 0) {
            System.out.printf(
                    "Former four-mode near-half diagnostic: %d "
                            + "candidates were flagged for comparison; "
                            + "RAP does not wet them because of this flag.%n",
                    c.formerFourModeNearHalfCandidateCount);
        }
        if (c.stcCoefficientChangeMismatch != 0) {
            System.out.printf("Integrity warning: native STC parity changes and observed JPEG coefficient modifications differ by %d.%n",
                    c.stcCoefficientChangeMismatch);
        }
        System.out.printf(
                "Three-state zero-flow observation: activations=%d, deactivations=%d, net delta h(0)=%+d.%n",
                c.modifiedZeroToNonzero, c.modifiedNonzeroToZero,
                c.modifiedNonzeroToZero - c.modifiedZeroToNonzero);
        if (c.highTailModifiedCoefficients == 0) {
            System.out.println("Control-band check: no coefficient changes were observed in the non-embedding high-tail "
                    + JpegMath.highTailLabel() + " band.");
        }

        System.out.println("Security statement: PSNR, SSIM, payload density, and divergence metrics are diagnostic evidence only. There is no universal KL/L1/JS threshold that proves undetectability; detector AUC and/or detection error P_E on a source-matched held-out test set are still required.");
        System.out.println("================================================================");
    }
}
