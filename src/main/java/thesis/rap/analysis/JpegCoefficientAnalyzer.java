package thesis.rap.analysis;

import thesis.rap.alg.Candidate;
import thesis.rap.alg.JpegMath;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Computes global, control-band, channel, and per-zig-zag detector-facing statistics. */
public final class JpegCoefficientAnalyzer {
    private JpegCoefficientAnalyzer() {}

    /** Backward-compatible overload for older callers that represented the STC pool only by a count. */
    public static JpegCoefficientReport compare(
            JpegImage reference,
            JpegImage stego,
            int payloadBits,
            int selectedQF,
            List<Candidate> candidates,
            int activeCarrierCount,
            int engineeringCapacity,
            boolean embeddingSuccess,
            boolean extractionSuccess,
            boolean aesGcmVerified,
            boolean recoveredMessageMatches
    ) {
        List<Candidate> active = candidates == null ? List.of()
                : candidates.subList(0, Math.max(0, Math.min(activeCarrierCount, candidates.size())));
        return compare(reference, stego, payloadBits, selectedQF, candidates, active, engineeringCapacity, null,
                embeddingSuccess, extractionSuccess, aesGcmVerified, recoveredMessageMatches);
    }

    public static JpegCoefficientReport compare(
            JpegImage reference,
            JpegImage stego,
            int payloadBits,
            int selectedQF,
            List<Candidate> candidates,
            List<Candidate> activeCarriers,
            int engineeringCapacity,
            StcRunMetrics stcMetrics,
            boolean embeddingSuccess,
            boolean extractionSuccess,
            boolean aesGcmVerified,
            boolean recoveredMessageMatches
    ) {
        requireCompatible(reference, stego);
        JpegCoefficientReport r = new JpegCoefficientReport();
        r.payloadBits = payloadBits;
        r.selectedQF = selectedQF;
        r.candidateCount = candidates == null ? 0 : candidates.size();
        r.activeCarrierCount = activeCarriers == null ? 0 : activeCarriers.size();
        r.conservativeCapacity = engineeringCapacity;
        r.embeddingSuccess = embeddingSuccess;
        r.extractionSuccess = extractionSuccess;
        r.aesGcmVerified = aesGcmVerified;
        r.recoveredMessageMatches = recoveredMessageMatches;
        if (candidates != null) {
            r.formerFourModeNearHalfCandidateCount =
                    (int) candidates.stream()
                    .filter(c -> c.formerFourModeNearHalf)
                    .count();
            r.fullyWetCandidateCount = (int) candidates.stream().filter(Candidate::fullyWet).count();
        }

        if (stcMetrics != null) populateStcMetrics(r, stcMetrics);

        Map<Integer, Long> globalRefHist = new HashMap<>();
        Map<Integer, Long> globalStegoHist = new HashMap<>();
        Map<Integer, Long> targetRefHist = new HashMap<>();
        Map<Integer, Long> targetStegoHist = new HashMap<>();
        Map<Integer, Long> highRefHist = new HashMap<>();
        Map<Integer, Long> highStegoHist = new HashMap<>();

        Map<Integer, DctPositionMetrics> byRank = new LinkedHashMap<>();
        Map<Integer, Map<Integer, Long>> rankRefHist = new HashMap<>();
        Map<Integer, Map<Integer, Long>> rankStegoHist = new HashMap<>();
        Map<Integer, ChannelMetrics> byChannel = new LinkedHashMap<>();
        Map<Integer, Map<Integer, Long>> channelRefHist = new HashMap<>();
        Map<Integer, Map<Integer, Long>> channelStegoHist = new HashMap<>();
        Map<String, DctChannelPositionMetrics> byChannelRank = new LinkedHashMap<>();
        Map<String, Map<Integer, Long>> channelRankRefHist = new HashMap<>();
        Map<String, Map<Integer, Long>> channelRankStegoHist = new HashMap<>();

        initializeMetricMaps(byRank, rankRefHist, rankStegoHist, byChannel, channelRefHist, channelStegoHist,
                byChannelRank, channelRankRefHist, channelRankStegoHist);
        countCandidates(candidates, byRank, byChannel, byChannelRank, false);
        countCandidates(activeCarriers, byRank, byChannel, byChannelRank, true);

        long absoluteDeltaSum = 0L;
        int components = Math.min(reference.components.length, stego.components.length);
        for (int componentIndex = 0; componentIndex < components; componentIndex++) {
            JpegComponent refComp = reference.components[componentIndex];
            JpegComponent stegoComp = stego.components[componentIndex];
            int len = Math.min(refComp.coefficients.length, stegoComp.coefficients.length);

            for (int i = 0; i < len; i++) {
                int natural = i % 64;
                int a = refComp.coefficients[i];
                int b = stegoComp.coefficients[i];
                int delta = b - a;
                int rank = JpegMath.zigzagRankFromNaturalIndex(natural);
                boolean targetCbCr = isCbCr(componentIndex) && JpegMath.isTargetExpandedMidBandRank(rank);
                boolean highTailCbCr = isCbCr(componentIndex) && JpegMath.isHighTailRank(rank);

                r.totalCoefficients++;
                globalRefHist.merge(a, 1L, Long::sum);
                globalStegoHist.merge(b, 1L, Long::sum);
                if (natural != 0 && a != 0) r.totalNonzeroAC++;

                DctPositionMetrics position = targetCbCr ? byRank.get(rank) : null;
                ChannelMetrics channel = targetCbCr ? byChannel.get(componentIndex) : null;
                DctChannelPositionMetrics channelPosition = targetCbCr ? byChannelRank.get(channelRankKey(componentIndex, rank)) : null;

                if (targetCbCr) {
                    r.targetBandCbCrCoefficients++;
                    if (a != 0) r.targetBandCbCrNonzero++;
                    if (b != 0) r.targetBandCbCrNonzeroStego++;
                    targetRefHist.merge(a, 1L, Long::sum);
                    targetStegoHist.merge(b, 1L, Long::sum);
                    rankRefHist.get(rank).merge(a, 1L, Long::sum);
                    rankStegoHist.get(rank).merge(b, 1L, Long::sum);
                    channelRefHist.get(componentIndex).merge(a, 1L, Long::sum);
                    channelStegoHist.get(componentIndex).merge(b, 1L, Long::sum);
                    channelRankRefHist.get(channelRankKey(componentIndex, rank)).merge(a, 1L, Long::sum);
                    channelRankStegoHist.get(channelRankKey(componentIndex, rank)).merge(b, 1L, Long::sum);

                    incrementOccupancy(position, a, b);
                    incrementOccupancy(channel, a, b);
                    incrementOccupancy(channelPosition, a, b);
                }

                if (highTailCbCr) {
                    r.highTailCbCrCoefficients++;
                    if (a != 0) r.highTailCbCrNonzero++;
                    if (b != 0) r.highTailCbCrNonzeroStego++;
                    highRefHist.merge(a, 1L, Long::sum);
                    highStegoHist.merge(b, 1L, Long::sum);
                    if (delta != 0) r.highTailModifiedCoefficients++;
                }

                if (delta != 0) {
                    r.modifiedCoefficients++;
                    absoluteDeltaSum += Math.abs((long) delta);
                    if (delta == 1) r.plusOneChanges++;
                    else if (delta == -1) r.minusOneChanges++;
                    else r.otherDeltaChanges++;

                    if (componentIndex == 0) r.modifiedY++;
                    else if (componentIndex == 1) r.modifiedCb++;
                    else if (componentIndex == 2) r.modifiedCr++;

                    if (a != 0 && b != 0) r.modifiedNonzeroToNonzero++;
                    else if (a == 0 && b != 0) r.modifiedZeroToNonzero++;
                    else if (a != 0) r.modifiedNonzeroToZero++;

                    if (targetCbCr) {
                        incrementModification(position, a, b, delta);
                        incrementModification(channel, a, b, delta);
                        incrementModification(channelPosition, a, b, delta);
                    }
                }
            }
        }

        r.targetBandCbCrZero = r.targetBandCbCrCoefficients - r.targetBandCbCrNonzero;
        r.targetBandCbCrZeroStego = r.targetBandCbCrCoefficients - r.targetBandCbCrNonzeroStego;
        r.highTailCbCrZero = r.highTailCbCrCoefficients - r.highTailCbCrNonzero;
        r.highTailCbCrZeroStego = r.highTailCbCrCoefficients - r.highTailCbCrNonzeroStego;

        r.candidateLoading = ratio(payloadBits, r.candidateCount);
        r.overallBpnzAC = ratio(payloadBits, r.totalNonzeroAC);
        r.bitsPerTotalCoefficient = ratio(payloadBits, r.totalCoefficients);
        r.bpnzAC = r.overallBpnzAC;
        r.candidateFractionOfNonzeroAC = ratio(r.candidateCount, r.totalNonzeroAC);
        r.candidateFractionOfAllCoefficients = ratio(r.candidateCount, r.totalCoefficients);
        r.localGlobalPayloadConcentration = ratio(r.candidateLoading, r.overallBpnzAC);
        r.engineeringUtilization = ratio(payloadBits, engineeringCapacity);
        r.activeCarrierPayloadRate = ratio(payloadBits, r.activeCarrierCount);
        r.modificationRateAllCoefficients = ratio(r.modifiedCoefficients, r.totalCoefficients);
        r.modificationRateCandidates = ratio(r.modifiedCoefficients, r.candidateCount);
        r.modificationRateActiveCarriers = ratio(r.modifiedCoefficients, r.activeCarrierCount);
        r.modifiedPerPayloadBit = ratio(r.modifiedCoefficients, payloadBits);
        r.embeddingEfficiencyBitsPerModification = ratio(payloadBits, r.modifiedCoefficients);
        int chromaChanges = r.modifiedCb + r.modifiedCr;
        r.cbModificationShare = ratio(r.modifiedCb, chromaChanges);
        r.crModificationShare = ratio(r.modifiedCr, chromaChanges);
        r.plusMinusImbalance = ratio(Math.abs(r.plusOneChanges - r.minusOneChanges), r.plusOneChanges + r.minusOneChanges);
        r.meanAbsoluteCoefficientDelta = ratio(absoluteDeltaSum, r.modifiedCoefficients);
        if (stcMetrics != null) r.stcCoefficientChangeMismatch = r.modifiedCoefficients - stcMetrics.totalChangedElements();

        SmoothedDistribution smoothed = smoothedDistribution(globalRefHist, globalStegoHist);
        r.dctHistogramL1 = smoothed.l1;
        r.dctKLDivergence = smoothed.kl;
        applyGlobalDistributionMetrics(r, StatisticalMath.compareHistograms(globalRefHist, globalStegoHist));
        applyTargetDistributionMetrics(r, StatisticalMath.compareHistograms(targetRefHist, targetStegoHist));
        applyHighTailDistributionMetrics(r, StatisticalMath.compareHistograms(highRefHist, highStegoHist));

        finalizePositionMetrics(reference, stego, r, byRank, rankRefHist, rankStegoHist);
        finalizeChannelMetrics(r, byChannel, channelRefHist, channelStegoHist);
        finalizeChannelPositionMetrics(reference, stego, r, byChannelRank, channelRankRefHist, channelRankStegoHist);
        return r;
    }

    private static void populateStcMetrics(JpegCoefficientReport r, StcRunMetrics s) {
        r.stcConstraintHeight = s.constraintHeight();
        r.stcReserveBits = s.plan().reserveBits();
        r.stcHeaderMessageBits = s.header().messageBitCount();
        r.stcHeaderCoverElements = s.header().coverElementCount();
        r.stcHeaderChangedElements = s.header().changedElementCount();
        r.stcHeaderRelativePayload = s.header().relativePayload();
        r.stcHeaderChangeRate = s.header().changeRate();
        r.stcBodyMessageBits = s.body().messageBitCount();
        r.stcBodyCoverElements = s.body().coverElementCount();
        r.stcBodyChangedElements = s.body().changedElementCount();
        r.stcBodyRelativePayload = s.body().relativePayload();
        r.stcBodyChangeRate = s.body().changeRate();
        r.stcDistinctColumnWidthLimit = 1 << Math.max(0, s.constraintHeight() - 2);
        r.stcHeaderMaximumSubmatrixWidth = maximumSubmatrixWidth(
                s.header().coverElementCount(), s.header().messageBitCount());
        r.stcBodyMaximumSubmatrixWidth = maximumSubmatrixWidth(
                s.body().coverElementCount(), s.body().messageBitCount());
        r.stcDuplicateColumnRisk = r.stcHeaderMaximumSubmatrixWidth > r.stcDistinctColumnWidthLimit
                || r.stcBodyMaximumSubmatrixWidth > r.stcDistinctColumnWidthLimit;
        r.stcTotalChangedElements = s.totalChangedElements();
        r.stcWeightedDistortion = s.totalWeightedDistortion();
        r.stcRelativePayload = s.overallRelativePayload();
        r.stcChangeRate = s.overallChangeRate();
    }

    private static void initializeMetricMaps(
            Map<Integer, DctPositionMetrics> byRank,
            Map<Integer, Map<Integer, Long>> rankRefHist,
            Map<Integer, Map<Integer, Long>> rankStegoHist,
            Map<Integer, ChannelMetrics> byChannel,
            Map<Integer, Map<Integer, Long>> channelRefHist,
            Map<Integer, Map<Integer, Long>> channelStegoHist,
            Map<String, DctChannelPositionMetrics> byChannelRank,
            Map<String, Map<Integer, Long>> channelRankRefHist,
            Map<String, Map<Integer, Long>> channelRankStegoHist
    ) {
        for (int rank = JpegMath.TARGET_ZZ_START_RANK; rank <= JpegMath.TARGET_ZZ_END_RANK; rank++) {
            DctPositionMetrics m = new DctPositionMetrics();
            m.zzRank = rank;
            m.naturalIndex = JpegMath.naturalIndexFromZigzagRank(rank);
            m.u = JpegMath.uFromNaturalIndex(m.naturalIndex);
            m.v = JpegMath.vFromNaturalIndex(m.naturalIndex);
            byRank.put(rank, m);
            rankRefHist.put(rank, new HashMap<>());
            rankStegoHist.put(rank, new HashMap<>());
        }

        for (int component = 1; component <= 2; component++) {
            ChannelMetrics c = new ChannelMetrics();
            c.componentIndex = component;
            c.channel = componentName(component);
            byChannel.put(component, c);
            channelRefHist.put(component, new HashMap<>());
            channelStegoHist.put(component, new HashMap<>());

            for (int rank = JpegMath.TARGET_ZZ_START_RANK; rank <= JpegMath.TARGET_ZZ_END_RANK; rank++) {
                DctChannelPositionMetrics p = new DctChannelPositionMetrics();
                p.componentIndex = component;
                p.channel = componentName(component);
                p.zzRank = rank;
                p.thesisZzRankOneBased = rank + 1;
                p.naturalIndex = JpegMath.naturalIndexFromZigzagRank(rank);
                p.u = JpegMath.uFromNaturalIndex(p.naturalIndex);
                p.v = JpegMath.vFromNaturalIndex(p.naturalIndex);
                String key = channelRankKey(component, rank);
                byChannelRank.put(key, p);
                channelRankRefHist.put(key, new HashMap<>());
                channelRankStegoHist.put(key, new HashMap<>());
            }
        }
    }

    private static void countCandidates(List<Candidate> candidates,
                                        Map<Integer, DctPositionMetrics> byRank,
                                        Map<Integer, ChannelMetrics> byChannel,
                                        Map<String, DctChannelPositionMetrics> byChannelRank,
                                        boolean active) {
        if (candidates == null) return;
        for (Candidate c : candidates) {
            int rank = JpegMath.zigzagRankFromNaturalIndex(c.k);
            DctPositionMetrics p = byRank.get(rank);
            ChannelMetrics ch = byChannel.get(c.component);
            DctChannelPositionMetrics cp = byChannelRank.get(channelRankKey(c.component, rank));
            if (p != null) { if (active) p.activeCarriers++; else p.eligibleCandidates++; }
            if (ch != null) { if (active) ch.activeCarriers++; else ch.eligibleCandidates++; }
            if (cp != null) { if (active) cp.activeCarriers++; else cp.eligibleCandidates++; }
        }
    }

    private static void incrementOccupancy(DctPositionMetrics p, int a, int b) {
        p.totalCoefficients++; if (a != 0) p.coverNonzero++; if (b != 0) p.stegoNonzero++;
    }
    private static void incrementOccupancy(ChannelMetrics p, int a, int b) {
        p.totalCoefficients++; if (a != 0) p.coverNonzero++; if (b != 0) p.stegoNonzero++;
    }
    private static void incrementOccupancy(DctChannelPositionMetrics p, int a, int b) {
        p.totalCoefficients++; if (a != 0) p.coverNonzero++; if (b != 0) p.stegoNonzero++;
    }

    private static void incrementModification(DctPositionMetrics p, int a, int b, int delta) {
        p.modifiedCoefficients++; p.meanAbsoluteDelta += Math.abs(delta);
        if (delta == 1) p.plusOneChanges++; else if (delta == -1) p.minusOneChanges++; else p.otherDeltaChanges++;
        if (a == 0 && b != 0) p.zeroToNonzero++; else if (a != 0 && b == 0) p.nonzeroToZero++;
    }
    private static void incrementModification(ChannelMetrics p, int a, int b, int delta) {
        p.modifiedCoefficients++; p.meanAbsoluteDelta += Math.abs(delta);
        if (delta == 1) p.plusOneChanges++; else if (delta == -1) p.minusOneChanges++; else p.otherDeltaChanges++;
        if (a == 0 && b != 0) p.zeroToNonzero++; else if (a != 0 && b == 0) p.nonzeroToZero++;
    }
    private static void incrementModification(DctChannelPositionMetrics p, int a, int b, int delta) {
        p.modifiedCoefficients++; p.meanAbsoluteDelta += Math.abs(delta);
        if (delta == 1) p.plusOneChanges++; else if (delta == -1) p.minusOneChanges++; else p.otherDeltaChanges++;
        if (a == 0 && b != 0) p.zeroToNonzero++; else if (a != 0 && b == 0) p.nonzeroToZero++;
    }

    private static void finalizePositionMetrics(JpegImage reference, JpegImage stego, JpegCoefficientReport r,
                                                Map<Integer, DctPositionMetrics> byRank,
                                                Map<Integer, Map<Integer, Long>> refHist,
                                                Map<Integer, Map<Integer, Long>> stegoHist) {
        for (DctPositionMetrics m : byRank.values()) {
            apply(m, StatisticalMath.compareHistograms(refHist.get(m.zzRank), stegoHist.get(m.zzRank)));
            m.candidateSelectionRate = ratio(m.activeCarriers, m.eligibleCandidates);
            m.modificationRateEligible = ratio(m.modifiedCoefficients, m.eligibleCandidates);
            m.modificationRateActive = ratio(m.modifiedCoefficients, m.activeCarriers);
            m.modificationShareOfAllChanges = ratio(m.modifiedCoefficients, r.modifiedCoefficients);
            m.plusMinusImbalance = ratio(Math.abs(m.plusOneChanges - m.minusOneChanges), m.plusOneChanges + m.minusOneChanges);
            m.meanAbsoluteDelta = ratio(m.meanAbsoluteDelta, m.modifiedCoefficients);
            computeAdjacentCorrelations(reference, stego, m, -1);
            r.positionMetrics.add(m);
        }
    }

    private static void finalizeChannelMetrics(JpegCoefficientReport r,
                                               Map<Integer, ChannelMetrics> byChannel,
                                               Map<Integer, Map<Integer, Long>> refHist,
                                               Map<Integer, Map<Integer, Long>> stegoHist) {
        for (ChannelMetrics m : byChannel.values()) {
            m.coverZero = m.totalCoefficients - m.coverNonzero;
            m.stegoZero = m.totalCoefficients - m.stegoNonzero;
            StatisticalMath.DistributionMetrics d = StatisticalMath.compareHistograms(refHist.get(m.componentIndex), stegoHist.get(m.componentIndex));
            m.histogramL1 = d.l1(); m.totalVariation = d.totalVariation(); m.klCoverToStego = d.klCoverToStego();
            m.symmetricKl = d.symmetricKl(); m.jensenShannon = d.jensenShannon(); m.hellinger = d.hellinger();
            m.candidateSelectionRate = ratio(m.activeCarriers, m.eligibleCandidates);
            m.modificationRateEligible = ratio(m.modifiedCoefficients, m.eligibleCandidates);
            m.modificationRateActive = ratio(m.modifiedCoefficients, m.activeCarriers);
            m.modificationShareAllChanges = ratio(m.modifiedCoefficients, r.modifiedCoefficients);
            m.plusMinusImbalance = ratio(Math.abs(m.plusOneChanges - m.minusOneChanges), m.plusOneChanges + m.minusOneChanges);
            m.meanAbsoluteDelta = ratio(m.meanAbsoluteDelta, m.modifiedCoefficients);
            r.channelMetrics.add(m);
        }
    }

    private static void finalizeChannelPositionMetrics(JpegImage reference, JpegImage stego, JpegCoefficientReport r,
                                                       Map<String, DctChannelPositionMetrics> byChannelRank,
                                                       Map<String, Map<Integer, Long>> refHist,
                                                       Map<String, Map<Integer, Long>> stegoHist) {
        for (Map.Entry<String, DctChannelPositionMetrics> e : byChannelRank.entrySet()) {
            DctChannelPositionMetrics m = e.getValue();
            apply(m, StatisticalMath.compareHistograms(refHist.get(e.getKey()), stegoHist.get(e.getKey())));
            m.candidateSelectionRate = ratio(m.activeCarriers, m.eligibleCandidates);
            m.modificationRateEligible = ratio(m.modifiedCoefficients, m.eligibleCandidates);
            m.modificationRateActive = ratio(m.modifiedCoefficients, m.activeCarriers);
            m.modificationShareAllChanges = ratio(m.modifiedCoefficients, r.modifiedCoefficients);
            m.plusMinusImbalance = ratio(Math.abs(m.plusOneChanges - m.minusOneChanges), m.plusOneChanges + m.minusOneChanges);
            m.meanAbsoluteDelta = ratio(m.meanAbsoluteDelta, m.modifiedCoefficients);
            computeAdjacentCorrelations(reference, stego, m, m.componentIndex);
            r.channelPositionMetrics.add(m);
        }
    }

    private static void apply(DctPositionMetrics m, StatisticalMath.DistributionMetrics d) {
        m.histogramL1 = d.l1(); m.totalVariation = d.totalVariation(); m.klCoverToStego = d.klCoverToStego();
        m.symmetricKl = d.symmetricKl(); m.jensenShannon = d.jensenShannon(); m.hellinger = d.hellinger();
    }
    private static void apply(DctChannelPositionMetrics m, StatisticalMath.DistributionMetrics d) {
        m.histogramL1 = d.l1(); m.totalVariation = d.totalVariation(); m.klCoverToStego = d.klCoverToStego();
        m.symmetricKl = d.symmetricKl(); m.jensenShannon = d.jensenShannon(); m.hellinger = d.hellinger();
    }

    private static void computeAdjacentCorrelations(JpegImage reference, JpegImage stego, Object metric, int onlyComponent) {
        int k;
        if (metric instanceof DctPositionMetrics m) k = m.naturalIndex;
        else if (metric instanceof DctChannelPositionMetrics m) k = m.naturalIndex;
        else throw new IllegalArgumentException("Unsupported metric type");

        List<Double> hCX = new ArrayList<>(), hCY = new ArrayList<>(), hSX = new ArrayList<>(), hSY = new ArrayList<>();
        List<Double> vCX = new ArrayList<>(), vCY = new ArrayList<>(), vSX = new ArrayList<>(), vSY = new ArrayList<>();
        int maxComponent = Math.min(Math.min(reference.components.length, stego.components.length), 3);
        int start = onlyComponent >= 0 ? onlyComponent : 1;
        int end = onlyComponent >= 0 ? onlyComponent + 1 : maxComponent;

        for (int componentIndex = start; componentIndex < end && componentIndex < maxComponent; componentIndex++) {
            JpegComponent rc = reference.components[componentIndex];
            JpegComponent sc = stego.components[componentIndex];
            int width = Math.min(rc.widthBlocks, sc.widthBlocks);
            int height = Math.min(rc.heightBlocks, sc.heightBlocks);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rb = y * rc.widthBlocks + x;
                    int sb = y * sc.widthBlocks + x;
                    if (x + 1 < width) {
                        int rr = y * rc.widthBlocks + x + 1;
                        int sr = y * sc.widthBlocks + x + 1;
                        hCX.add((double) rc.getCoeff(rb, k)); hCY.add((double) rc.getCoeff(rr, k));
                        hSX.add((double) sc.getCoeff(sb, k)); hSY.add((double) sc.getCoeff(sr, k));
                    }
                    if (y + 1 < height) {
                        int rd = (y + 1) * rc.widthBlocks + x;
                        int sd = (y + 1) * sc.widthBlocks + x;
                        vCX.add((double) rc.getCoeff(rb, k)); vCY.add((double) rc.getCoeff(rd, k));
                        vSX.add((double) sc.getCoeff(sb, k)); vSY.add((double) sc.getCoeff(sd, k));
                    }
                }
            }
        }

        double hc = StatisticalMath.pearson(toArray(hCX), toArray(hCY));
        double hs = StatisticalMath.pearson(toArray(hSX), toArray(hSY));
        double vc = StatisticalMath.pearson(toArray(vCX), toArray(vCY));
        double vs = StatisticalMath.pearson(toArray(vSX), toArray(vSY));
        if (metric instanceof DctPositionMetrics m) {
            m.horizontalCorrelationCover = hc; m.horizontalCorrelationStego = hs; m.horizontalCorrelationDelta = finiteDifference(hs, hc);
            m.verticalCorrelationCover = vc; m.verticalCorrelationStego = vs; m.verticalCorrelationDelta = finiteDifference(vs, vc);
        } else {
            DctChannelPositionMetrics m = (DctChannelPositionMetrics) metric;
            m.horizontalCorrelationCover = hc; m.horizontalCorrelationStego = hs; m.horizontalCorrelationDelta = finiteDifference(hs, hc);
            m.verticalCorrelationCover = vc; m.verticalCorrelationStego = vs; m.verticalCorrelationDelta = finiteDifference(vs, vc);
        }
    }

    private static double[] toArray(List<Double> values) {
        double[] out = new double[values.size()];
        for (int i = 0; i < values.size(); i++) out[i] = values.get(i);
        return out;
    }

    private static double finiteDifference(double a, double b) {
        return Double.isFinite(a) && Double.isFinite(b) ? a - b : Double.NaN;
    }

    static SmoothedDistribution smoothedDistribution(Map<Integer, Long> cover, Map<Integer, Long> stego) {
        final int min = -1024, max = 1024, size = max - min + 1;
        final double eps = 1e-12;
        double[] p = new double[size], q = new double[size];
        double pTotal = 0.0, qTotal = 0.0;
        for (Map.Entry<Integer, Long> e : cover.entrySet()) { int clipped = Math.max(min, Math.min(max, e.getKey())); p[clipped - min] += e.getValue(); pTotal += e.getValue(); }
        for (Map.Entry<Integer, Long> e : stego.entrySet()) { int clipped = Math.max(min, Math.min(max, e.getKey())); q[clipped - min] += e.getValue(); qTotal += e.getValue(); }
        double pDenominator = pTotal + eps * size;
        double qDenominator = qTotal + eps * size;
        double l1 = 0.0;
        double kl = 0.0;
        double compensation = 0.0;
        for (int i = 0; i < size; i++) {
            double rawP = pTotal == 0.0 ? 0.0 : p[i] / pTotal;
            double rawQ = qTotal == 0.0 ? 0.0 : q[i] / qTotal;
            l1 += Math.abs(rawP - rawQ);
            double smoothP = (p[i] + eps) / pDenominator;
            double smoothQ = (q[i] + eps) / qDenominator;
            double term = smoothP * StrictMath.log(smoothP / smoothQ);
            double corrected = term - compensation;
            double next = kl + corrected;
            compensation = (next - kl) - corrected;
            kl = next;
        }
        if (kl < -1.0e-12) {
            throw new IllegalStateException(
                    "Normalized smoothed KL became materially negative: "
                            + kl);
        }
        if (kl < 0.0) kl = 0.0;
        return new SmoothedDistribution(l1, kl);
    }

    static record SmoothedDistribution(double l1, double kl) {}

    private static void applyGlobalDistributionMetrics(JpegCoefficientReport r, StatisticalMath.DistributionMetrics d) {
        r.dctExactHistogramL1 = d.l1(); r.dctExactTotalVariation = d.totalVariation(); r.dctExactKLDivergence = d.klCoverToStego();
        r.dctExactSymmetricKL = d.symmetricKl(); r.dctExactJensenShannon = d.jensenShannon(); r.dctExactHellinger = d.hellinger();
    }
    private static void applyTargetDistributionMetrics(JpegCoefficientReport r, StatisticalMath.DistributionMetrics d) {
        r.targetBandHistogramL1 = d.l1(); r.targetBandTotalVariation = d.totalVariation(); r.targetBandKLDivergence = d.klCoverToStego();
        r.targetBandSymmetricKL = d.symmetricKl(); r.targetBandJensenShannon = d.jensenShannon(); r.targetBandHellinger = d.hellinger();
    }
    private static void applyHighTailDistributionMetrics(JpegCoefficientReport r, StatisticalMath.DistributionMetrics d) {
        r.highTailHistogramL1 = d.l1(); r.highTailTotalVariation = d.totalVariation(); r.highTailKLDivergence = d.klCoverToStego();
        r.highTailSymmetricKL = d.symmetricKl(); r.highTailJensenShannon = d.jensenShannon(); r.highTailHellinger = d.hellinger();
    }

    private static int maximumSubmatrixWidth(int coverElements, int messageBits) {
        if (messageBits <= 0) return 0;
        return (int) Math.ceil(coverElements / (double) messageBits);
    }

    private static boolean isCbCr(int componentIndex) { return componentIndex == 1 || componentIndex == 2; }
    private static String componentName(int componentIndex) { return componentIndex == 1 ? "Cb" : componentIndex == 2 ? "Cr" : "C" + componentIndex; }
    private static String channelRankKey(int component, int rank) { return component + ":" + rank; }
    private static double ratio(double numerator, double denominator) { return denominator == 0.0 ? 0.0 : numerator / denominator; }

    private static void requireCompatible(JpegImage a, JpegImage b) {
        if (a.width != b.width || a.height != b.height || a.components.length != b.components.length) {
            throw new IllegalArgumentException("Reference and stego JPEG coefficient structures differ.");
        }
        for (int i = 0; i < a.components.length; i++) {
            if (a.components[i].coefficients.length != b.components[i].coefficients.length) {
                throw new IllegalArgumentException("Reference and stego component coefficient lengths differ at component " + i);
            }
        }
    }
}
