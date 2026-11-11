package thesis.rap.app;

import thesis.rap.alg.Candidate;
import thesis.rap.alg.CandidateSet;
import thesis.rap.alg.JpegMath;
import thesis.rap.jni.JpegComponent;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.jni.JpegImage;
import thesis.rap.payload.PayloadPackage;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/** Implements Algorithm 3 after the canonical reference and encrypted package exist. */
public final class FixedQfEmbeddingKernel {
    private static final int HEADER_BITS = PayloadPackage.headerBytes() * 8;

    private FixedQfEmbeddingKernel() {}

    public record Result(
            CandidateSet candidateSet,
            List<Candidate> keyedDryOrder,
            List<Candidate> activeCandidates,
            StcCarrierPlan carrierPlan,
            ChannelAllocationEstimate channelAllocation,
            ThreeStateEmbeddingEngine.Result balance,
            JpegImage intendedStego,
            byte[] header,
            byte[] body
    ) {
        public Result {
            keyedDryOrder = Collections.unmodifiableList(new ArrayList<>(keyedDryOrder));
            activeCandidates = Collections.unmodifiableList(new ArrayList<>(activeCandidates));
            if (channelAllocation == null) {
                throw new IllegalArgumentException(
                        "channelAllocation cannot be null");
            }
            header = Arrays.copyOf(header, header.length);
            body = Arrays.copyOf(body, body.length);
        }

        @Override public byte[] header() {
            return Arrays.copyOf(header, header.length);
        }
        @Override public byte[] body() {
            return Arrays.copyOf(body, body.length);
        }
    }

    public static Result embed(
            JpegImage canonicalReference,
            CandidateSet candidateSet,
            byte[] packedPayload,
            String encodedKeys,
            int qf,
            SamplingGeometry geometry,
            ThreeStateEmbeddingEngine.Profile profile,
            StcEmbeddingBackend stc
    ) throws Exception {
        if (canonicalReference == null || candidateSet == null || packedPayload == null
                || geometry == null) {
            throw new IllegalArgumentException("Kernel inputs cannot be null");
        }
        geometry.require(canonicalReference);
        if (packedPayload.length < PayloadPackage.headerBytes()
                + PayloadPackage.PREPROCESSING_METADATA_LEN) {
            throw new IllegalArgumentException("Packed payload is shorter than the fixed header/body minimum");
        }

        List<Candidate> keyed = new ArrayList<>(candidateSet.dryBaseCostSorted());
        StcKeyedOrder.apply(keyed, encodedKeys, qf, geometry);
        byte[] header = Arrays.copyOfRange(
                packedPayload, 0, PayloadPackage.headerBytes());
        byte[] body = Arrays.copyOfRange(
                packedPayload, PayloadPackage.headerBytes(), packedPayload.length);
        if (PayloadPackage.ciphertextLengthFromHeader(header)
                != body.length) {
            throw new IllegalArgumentException(
                    "Packed payload header/body does not match the fixed-QF kernel");
        }
        StcCarrierPlan plan = StcCarrierPlan.createForHeight(
                keyed.size(), HEADER_BITS,
                Math.multiplyExact(body.length, 8),
                stc.constraintHeight());

        List<Candidate> headerCandidates = keyed.subList(0, plan.headerCarrierCount());
        int bodyEnd = Math.addExact(plan.headerCarrierCount(), plan.bodyCarrierCount());
        List<Candidate> bodyCandidates = keyed.subList(plan.headerCarrierCount(), bodyEnd);
        List<Candidate> active = keyed.subList(0, bodyEnd);
        int[] headerTieDirections = StcKeyedOrder.directionTieBreaks(
                headerCandidates, encodedKeys, qf, geometry);
        int[] bodyTieDirections = StcKeyedOrder.directionTieBreaks(
                bodyCandidates, encodedKeys, qf, geometry);

        byte[] headerCover = parityBits(canonicalReference, headerCandidates);
        byte[] bodyCover = parityBits(canonicalReference, bodyCandidates);
        int[][] referenceSnapshot = coefficientSnapshot(canonicalReference);
        ThreeStateEmbeddingEngine.Result balanced = ThreeStateEmbeddingEngine.embed(
                stc,
                headerCandidates, headerCover, header,
                bodyCandidates, bodyCover, body,
                qf, profile,
                headerTieDirections, bodyTieDirections);

        applyParityChanges(
                canonicalReference, headerCandidates, headerCover,
                balanced.header().stegoBitsView(), balanced.headerDirectionsView());
        applyParityChanges(
                canonicalReference, bodyCandidates, bodyCover,
                balanced.body().stegoBitsView(), balanced.bodyDirectionsView());
        validatePostconditions(
                candidateSet.universe(), active, canonicalReference);
        validateWholeImagePostconditions(
                referenceSnapshot, active, canonicalReference);

        double[] finalScalarCosts = new double[active.size()];
        double[] headerCosts = balanced.headerAdjustedCostsView();
        double[] bodyCosts = balanced.bodyAdjustedCostsView();
        System.arraycopy(
                headerCosts, 0, finalScalarCosts, 0,
                headerCosts.length);
        System.arraycopy(
                bodyCosts, 0, finalScalarCosts,
                headerCosts.length, bodyCosts.length);
        ChannelAllocationEstimate allocation =
                ChannelAllocationEstimate.analyze(
                        active, finalScalarCosts, canonicalReference,
                        plan);

        return new Result(
                candidateSet, keyed, active, plan, allocation, balanced,
                canonicalReference, header, body);
    }

    public static byte[] parityBits(JpegImage image, List<Candidate> candidates) {
        byte[] bits = new byte[candidates.size()];
        for (int i = 0; i < candidates.size(); i++) {
            Candidate candidate = candidates.get(i);
            int q = image.component(candidate.component)
                    .getCoeff(candidate.blockIndex, candidate.k);
            bits[i] = (byte) Math.floorMod(q, 2);
        }
        return bits;
    }

    private static void applyParityChanges(
            JpegImage image,
            List<Candidate> candidates,
            byte[] coverBits,
            byte[] stegoBits,
            int[] directions
    ) {
        if (coverBits.length != stegoBits.length
                || coverBits.length != candidates.size()
                || coverBits.length != directions.length) {
            throw new IllegalStateException("STC result dimensions do not match the carrier plan");
        }

        for (int i = 0; i < candidates.size(); i++) {
            Candidate candidate = candidates.get(i);
            if (coverBits[i] == stegoBits[i]) continue;
            int delta = directions[i];
            if (delta != -1 && delta != 1) {
                throw new IllegalStateException("Physical direction is not a unit change");
            }
            if (JpegMath.isWetCost(candidate.directionalBaseCost(delta))) {
                throw new IllegalStateException(
                        "STC requested a base-wet transition for " + candidate.describe());
            }
            JpegComponent component = image.component(candidate.component);
            int current = component.getCoeff(candidate.blockIndex, candidate.k);
            if (current != candidate.qRef) {
                throw new IllegalStateException(
                        "Kernel did not restart from the canonical reference for "
                                + candidate.describe());
            }
            component.setCoeff(
                    candidate.blockIndex, candidate.k,
                    Math.addExact(candidate.qRef, delta));
        }
    }

    public static void validatePostconditions(
            List<Candidate> universe,
            List<Candidate> active,
            JpegImage intendedStego
    ) {
        Set<Candidate> activeIdentity = Collections.newSetFromMap(new IdentityHashMap<>());
        activeIdentity.addAll(active);
        for (Candidate candidate : universe) {
            int actual = intendedStego.component(candidate.component)
                    .getCoeff(candidate.blockIndex, candidate.k);
            int delta = actual - candidate.qRef;
            if (delta != 0 && delta != -1 && delta != 1) {
                throw new IllegalStateException(
                        "Candidate changed by non-unit delta=" + delta + ": "
                                + candidate.describe());
            }
            if (delta != 0 && !activeIdentity.contains(candidate)) {
                throw new IllegalStateException(
                        "Inactive carrier changed: " + candidate.describe());
            }
            if (delta != 0 && JpegMath.isWetCost(candidate.directionalBaseCost(delta))) {
                throw new IllegalStateException(
                        "Wet directional change survived postconditions: "
                                + candidate.describe());
            }
        }
    }

    private static int[][] coefficientSnapshot(JpegImage image) {
        int[][] out = new int[image.components.length][];
        for (int i = 0; i < image.components.length; i++) {
            out[i] = Arrays.copyOf(
                    image.component(i).coefficients,
                    image.component(i).coefficients.length);
        }
        return out;
    }

    private static void validateWholeImagePostconditions(
            int[][] reference,
            List<Candidate> active,
            JpegImage intendedStego
    ) {
        if (reference.length != intendedStego.components.length) {
            throw new IllegalStateException(
                    "Component count changed during fixed-QF embedding");
        }
        boolean[][] activeAddress = new boolean[reference.length][];
        for (int component = 0; component < reference.length; component++) {
            activeAddress[component] =
                    new boolean[reference[component].length];
        }
        for (Candidate candidate : active) {
            int linear = Math.addExact(
                    Math.multiplyExact(candidate.blockIndex, 64),
                    candidate.k);
            activeAddress[candidate.component][linear] = true;
        }
        for (int component = 0; component < reference.length; component++) {
            int[] actual =
                    intendedStego.component(component).coefficients;
            if (reference[component].length != actual.length) {
                throw new IllegalStateException(
                        "Coefficient array size changed during embedding");
            }
            for (int index = 0; index < actual.length; index++) {
                int delta = actual[index] - reference[component][index];
                if (delta == 0) continue;
                if (!activeAddress[component][index]) {
                    throw new IllegalStateException(
                            "Coefficient outside the active carrier set changed");
                }
                if (delta != -1 && delta != 1) {
                    throw new IllegalStateException(
                            "Active coefficient changed by a non-unit delta");
                }
            }
        }
    }
}
