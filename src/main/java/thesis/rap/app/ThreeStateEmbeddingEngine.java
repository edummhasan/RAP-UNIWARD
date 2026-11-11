package thesis.rap.app;

import thesis.rap.alg.Candidate;
import thesis.rap.alg.JpegMath;
import thesis.rap.gpu.CudaTrueParallelNative;
import thesis.rap.gpu.CudaBalanceNative;
import thesis.rap.runtime.ExecutionBackend;
import thesis.rap.runtime.GpuStageStats;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Normative finite U/B/S embedding loop for the three-state carrier universe.
 *
 * <p>Candidate membership, base-cost order, keyed windows, and carrier allocation
 * are frozen by the caller.  Every B/S attempt starts from the same cover parity
 * vectors, recomputes both physical directions from their immutable base costs,
 * and freshly runs both STC calls.</p>
 */
public final class ThreeStateEmbeddingEngine {
    static final String VERBOSE_ATTEMPTS_PROPERTY =
            "thesis.rap.verboseBalanceAttempts";
    public static final int BALANCE_MAX_ATTEMPTS = 40;
    public static final int BOUNDED_ZERO_BUDGET = 4;
    public static final double STEP_SCALE = 0.08;
    public static final double MULTIPLIER_BOUND_SCALE = 8.0;

    private ThreeStateEmbeddingEngine() {}

    public enum Profile {
        UNRESTRICTED("U"), BOUNDED("B"), STRICT("S");

        private final String code;

        Profile(String code) {
            this.code = code;
        }

        public String code() {
            return code;
        }

        public static Profile configured() {
            String raw = System.getenv().getOrDefault("RAP_BALANCE_PROFILE", "B")
                    .trim().toUpperCase(Locale.ROOT);
            return switch (raw) {
                case "U", "UNRESTRICTED" -> UNRESTRICTED;
                case "B", "BOUNDED" -> BOUNDED;
                case "S", "STRICT" -> STRICT;
                default -> throw new IllegalArgumentException(
                        "RAP_BALANCE_PROFILE must be U, B, or S: " + raw);
            };
        }

        public int zeroBudget() {
            return switch (this) {
                case UNRESTRICTED -> Integer.MAX_VALUE;
                case BOUNDED -> BOUNDED_ZERO_BUDGET;
                case STRICT -> 0;
            };
        }

        public int maxAttempts() {
            return this == UNRESTRICTED ? 1 : BALANCE_MAX_ATTEMPTS;
        }
    }

    public record TransitionCounts(
            long zeroToPlus,
            long zeroToMinus,
            long plusToZero,
            long minusToZero
    ) implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;

        public long activations() {
            return zeroToPlus + zeroToMinus;
        }

        public long deactivations() {
            return plusToZero + minusToZero;
        }

        public long zeroBinDelta() {
            return deactivations() - activations();
        }
    }

    public record Attempt(
            int qf,
            int iteration,
            double scale,
            double multiplier,
            double step,
            boolean headerStcSuccess,
            boolean bodyStcSuccess,
            TransitionCounts transitions,
            long zeroBinDelta,
            boolean syndromeVerified,
            boolean accepted,
            String status,
            String reason
    ) implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;
    }

    public record Result(
            StcEmbedResult header,
            StcEmbedResult body,
            Profile profile,
            int iterations,
            long zeroBinDelta,
            double finalMultiplier,
            double scale,
            int[] headerDirections,
            int[] bodyDirections,
            double[] headerAdjustedCosts,
            double[] bodyAdjustedCosts,
            List<Attempt> attempts
    ) {
        public Result {
            headerDirections = Arrays.copyOf(headerDirections, headerDirections.length);
            bodyDirections = Arrays.copyOf(bodyDirections, bodyDirections.length);
            headerAdjustedCosts = Arrays.copyOf(headerAdjustedCosts, headerAdjustedCosts.length);
            bodyAdjustedCosts = Arrays.copyOf(bodyAdjustedCosts, bodyAdjustedCosts.length);
            attempts = Collections.unmodifiableList(new ArrayList<>(attempts));
        }

        @Override public int[] headerDirections() {
            return Arrays.copyOf(headerDirections, headerDirections.length);
        }
        @Override public int[] bodyDirections() {
            return Arrays.copyOf(bodyDirections, bodyDirections.length);
        }
        @Override public double[] headerAdjustedCosts() {
            return Arrays.copyOf(headerAdjustedCosts, headerAdjustedCosts.length);
        }
        @Override public double[] bodyAdjustedCosts() {
            return Arrays.copyOf(bodyAdjustedCosts, bodyAdjustedCosts.length);
        }

        int[] headerDirectionsView() { return headerDirections; }
        int[] bodyDirectionsView() { return bodyDirections; }
        double[] headerAdjustedCostsView() { return headerAdjustedCosts; }
        double[] bodyAdjustedCostsView() { return bodyAdjustedCosts; }
    }

    public static final class NoFeasibleBalanceException extends IllegalStateException {
        @Serial
        private static final long serialVersionUID = 1L;
        private final ArrayList<Attempt> attempts;
        private final String reasonCode;

        NoFeasibleBalanceException(
                String message,
                List<Attempt> attempts,
                String reasonCode,
                Throwable cause
        ) {
            super(message, cause);
            this.attempts = new ArrayList<>(attempts);
            if (reasonCode == null
                    || !reasonCode.matches("[A-Z][A-Z0-9_]{1,63}")) {
                throw new IllegalArgumentException(
                        "reasonCode must be a stable uppercase identifier");
            }
            this.reasonCode = reasonCode;
        }

        NoFeasibleBalanceException(String message, List<Attempt> attempts) {
            this(message, attempts, "BALANCE_BUDGET_REJECTED", null);
        }

        NoFeasibleBalanceException(
                String message,
                List<Attempt> attempts,
                String reasonCode
        ) {
            this(message, attempts, reasonCode, null);
        }

        public List<Attempt> attempts() {
            return Collections.unmodifiableList(attempts);
        }

        public String reasonCode() {
            return reasonCode;
        }
    }

    public static Result embed(
            StcEmbeddingBackend stc,
            List<Candidate> headerCandidates, byte[] headerCover, byte[] header,
            List<Candidate> bodyCandidates, byte[] bodyCover, byte[] body
    ) {
        return embed(stc, headerCandidates, headerCover, header,
                bodyCandidates, bodyCover, body, -1, Profile.configured());
    }

    public static Result embed(
            StcEmbeddingBackend stc,
            List<Candidate> headerCandidates, byte[] headerCover, byte[] header,
            List<Candidate> bodyCandidates, byte[] bodyCover, byte[] body,
            int qf, Profile profile
    ) {
        return embed(stc,
                headerCandidates, headerCover, header,
                bodyCandidates, bodyCover, body,
                qf, profile, null, null);
    }

    static Result embed(
            StcEmbeddingBackend stc,
            List<Candidate> headerCandidates, byte[] headerCover, byte[] header,
            List<Candidate> bodyCandidates, byte[] bodyCover, byte[] body,
            int qf, Profile profile,
            int[] headerTieDirections,
            int[] bodyTieDirections
    ) {
        requireDimensions(headerCandidates, headerCover, header, "header");
        requireDimensions(bodyCandidates, bodyCover, body, "body");
        if (stc == null || profile == null) throw new IllegalArgumentException("stc/profile cannot be null");
        requireTieDirections(
                headerCandidates, headerTieDirections, "header");
        requireTieDirections(
                bodyCandidates, bodyTieDirections, "body");

        double scale = finiteDirectionalMedian(headerCandidates, bodyCandidates);
        double multiplier = 0.0;
        double multiplierBound = MULTIPLIER_BOUND_SCALE * scale;
        List<Attempt> attempts = new ArrayList<>();
        GpuBalanceInputs headerGpuInputs = gpuBalanceInputs(
                headerCandidates, headerTieDirections);
        GpuBalanceInputs bodyGpuInputs = gpuBalanceInputs(
                bodyCandidates, bodyTieDirections);

        for (int k = 0; k < profile.maxAttempts(); k++) {
            double step = STEP_SCALE * scale / StrictMath.sqrt(k + 1.0);
            if (verboseAttempts()) {
                System.out.printf(Locale.US,
                        "===== BALANCE ATTEMPT START q=%d profile=%s k=%d =====%n",
                        qf, profile.code(), k);
            }

            DirectionalPlan headerPlan = adjustedPlanGpuAware(
                    headerCandidates, multiplier,
                    headerTieDirections, headerGpuInputs);
            DirectionalPlan bodyPlan = adjustedPlanGpuAware(
                    bodyCandidates, multiplier,
                    bodyTieDirections, bodyGpuInputs);
            StcEmbedResult headerResult;
            StcEmbedResult bodyResult;
            boolean headerSuccess = false;
            boolean bodySuccess = false;

            try {
                headerResult = stc.embedPayloadWithTrace(
                        headerCover, headerPlan.scalarCostsView(), header);
                headerSuccess = true;
                bodyResult = stc.embedPayloadWithTrace(
                        bodyCover, bodyPlan.scalarCostsView(), body);
                bodySuccess = true;
            } catch (thesis.rap.stc.NoFiniteStcSolutionException failure) {
                Attempt attempt = new Attempt(
                        qf, k, scale, multiplier, step,
                        headerSuccess, bodySuccess,
                        new TransitionCounts(0, 0, 0, 0), 0,
                        false, false, "STC_FAILED", safeMessage(failure));
                attempts.add(attempt);
                if (verboseAttempts()) {
                    System.out.println(
                            "===== BALANCE ATTEMPT END status=STC_FAILED deltaZ=0 =====");
                }
                throw new NoFeasibleBalanceException(
                        "Header/body STC failed at QF " + qf + " attempt " + k,
                        attempts,
                        headerSuccess
                                ? "BODY_STC_REJECTED"
                                : "HEADER_STC_REJECTED",
                        failure);
            }

            boolean syndromeVerified = Arrays.equals(
                    header, stc.extractPayload(headerResult.stegoBitsView(), header.length))
                    && Arrays.equals(
                    body, stc.extractPayload(bodyResult.stegoBitsView(), body.length));

            TransitionCounts transitions = combine(
                    transitionCounts(headerCandidates, headerCover,
                            headerResult.stegoBitsView(), headerPlan.directionsView()),
                    transitionCounts(bodyCandidates, bodyCover,
                            bodyResult.stegoBitsView(), bodyPlan.directionsView()));
            long delta = transitions.zeroBinDelta();
            boolean profileAccepted = profile == Profile.UNRESTRICTED
                    || Math.abs(delta) <= profile.zeroBudget();
            boolean accepted = syndromeVerified && profileAccepted;
            double nextMultiplier = accepted
                    ? multiplier
                    : clip(
                            multiplier + step * delta,
                            -multiplierBound,
                            multiplierBound);
            boolean balanceFixedPoint = syndromeVerified
                    && !profileAccepted
                    && Double.doubleToLongBits(nextMultiplier)
                    == Double.doubleToLongBits(multiplier);
            String status = accepted ? "ACCEPTED"
                    : !syndromeVerified ? "SYNDROME_FAILED"
                    : balanceFixedPoint
                    ? "BALANCE_BOUND_REPEATED"
                    : "BALANCE_BUDGET_REJECTED";
            String reason = accepted ? ""
                    : !syndromeVerified
                    ? "STC extraction did not reproduce the embedded header/body"
                    : balanceFixedPoint
                    ? "The clipped balance parameter cannot change; the frozen "
                            + "carrier order, costs, STC inputs, and delta h(0) "
                            + "would repeat identically"
                    : "abs(deltaZ)=" + Math.abs(delta)
                            + " exceeds budget=" + profile.zeroBudget();
            Attempt attempt = new Attempt(
                    qf, k, scale, multiplier, step,
                    true, true, transitions, delta,
                    syndromeVerified, accepted, status, reason);
            attempts.add(attempt);
            if (verboseAttempts()) {
                System.out.printf(Locale.US,
                        "===== BALANCE ATTEMPT END status=%s deltaZ=%+d =====%n",
                        status, delta);
            }

            if (!syndromeVerified) {
                throw new NoFeasibleBalanceException(
                        "STC syndrome verification failed at QF " + qf + " attempt " + k,
                        attempts, "SYNDROME_REJECTED");
            }

            if (accepted) {
                return new Result(
                        headerResult, bodyResult, profile, k + 1, delta,
                        multiplier, scale,
                        headerPlan.directionsView(), bodyPlan.directionsView(),
                        headerPlan.scalarCostsView(), bodyPlan.scalarCostsView(), attempts);
            }

            if (balanceFixedPoint) {
                throw new NoFeasibleBalanceException(
                        "Balance parameter reached a deterministic fixed point "
                                + "outside profile=" + profile.code()
                                + " at QF " + qf + " attempt " + k,
                        attempts, "BALANCE_BOUND_REPEATED");
            }
            multiplier = nextMultiplier;
        }

        throw new NoFeasibleBalanceException(
                "No STC solution satisfied profile=" + profile.code()
                        + " within " + profile.maxAttempts() + " attempts at QF " + qf,
                attempts, "BALANCE_BUDGET_REJECTED");
    }

    private static DirectionalPlan adjustedPlanGpuAware(
            List<Candidate> candidates,
            double multiplier,
            int[] tieDirections,
            GpuBalanceInputs inputs
    ) {
        if (inputs == null) {
            return adjustedPlan(candidates, multiplier, tieDirections);
        }
        try {
            double[] packed = CudaBalanceNative.compute(
                    inputs.rhoPlus(), inputs.rhoMinus(),
                    inputs.minimumPlus(), inputs.minimumMinus(),
                    inputs.zeroFlowPlus(), inputs.zeroFlowMinus(),
                    inputs.tieDirections(), multiplier, JpegMath.WET_COST);
            int n = candidates.size();
            double[] plus = Arrays.copyOfRange(packed, 0, n);
            double[] minus = Arrays.copyOfRange(packed, n, 2 * n);
            double[] scalar = Arrays.copyOfRange(packed, 2 * n, 3 * n);
            int[] directions = new int[n];
            for (int i = 0; i < n; i++) {
                double encoded = packed[3 * n + i];
                if (encoded == -1.0) directions[i] = -1;
                else if (encoded == 1.0) directions[i] = 1;
                else throw new IllegalStateException(
                        "CUDA balance returned a non-unit direction");
            }
            GpuStageStats.gpu(GpuStageStats.Stage.BALANCE_PLAN);
            return new DirectionalPlan(plus, minus, scalar, directions);
        } catch (RuntimeException | LinkageError failure) {
            if (ExecutionBackend.cudaFailureIsFatal()) throw failure;
            GpuStageStats.fallback(GpuStageStats.Stage.BALANCE_PLAN, failure);
            return adjustedPlan(candidates, multiplier, tieDirections);
        }
    }

    private static GpuBalanceInputs gpuBalanceInputs(
            List<Candidate> candidates, int[] tieDirections
    ) {
        if (candidates.size() < 2048 || !gpuBalanceConfigured()) return null;
        int n = candidates.size();
        double[] rhoPlus = new double[n];
        double[] rhoMinus = new double[n];
        double[] minimumPlus = new double[n];
        double[] minimumMinus = new double[n];
        int[] zeroFlowPlus = new int[n];
        int[] zeroFlowMinus = new int[n];
        int[] ties = new int[n];
        for (int i = 0; i < n; i++) {
            Candidate candidate = candidates.get(i);
            rhoPlus[i] = candidate.rhoPlus;
            rhoMinus[i] = candidate.rhoMinus;
            minimumPlus[i] = candidate.minimumAdjustedPlus;
            minimumMinus[i] = candidate.minimumAdjustedMinus;
            zeroFlowPlus[i] = candidate.zeroFlowContribution(1);
            zeroFlowMinus[i] = candidate.zeroFlowContribution(-1);
            ties[i] = tieDirections == null
                    ? candidate.bestParityFlipDelta(1.0, 1.0)
                    : tieDirections[i];
        }
        return new GpuBalanceInputs(
                rhoPlus, rhoMinus, minimumPlus, minimumMinus,
                zeroFlowPlus, zeroFlowMinus, ties);
    }

    private static boolean gpuBalanceConfigured() {
        return ExecutionBackend.cudaPermitted()
                && Boolean.parseBoolean(System.getProperty(
                "rap.gpu.balance.plan",
                System.getenv().getOrDefault(
                        "RAP_GPU_BALANCE_PLAN", "true")))
                && CudaBalanceNative.isAvailable();
    }

    static DirectionalPlan adjustedPlan(List<Candidate> candidates, double multiplier) {
        return adjustedPlan(candidates, multiplier, null);
    }

    static DirectionalPlan adjustedPlan(
            List<Candidate> candidates,
            double multiplier,
            int[] tieDirections
    ) {
        requireTieDirections(candidates, tieDirections, "directional plan");
        double[] plus = new double[candidates.size()];
        double[] minus = new double[candidates.size()];
        double[] scalar = new double[candidates.size()];
        int[] directions = new int[candidates.size()];

        for (int i = 0; i < candidates.size(); i++) {
            Candidate candidate = candidates.get(i);
            plus[i] = adjustedDirection(candidate, 1, multiplier);
            minus[i] = adjustedDirection(candidate, -1, multiplier);
            scalar[i] = Math.min(plus[i], minus[i]);
            directions[i] = tieDirections == null
                    ? candidate.bestParityFlipDelta(plus[i], minus[i])
                    : candidate.bestParityFlipDelta(
                            plus[i], minus[i], tieDirections[i]);
        }
        return new DirectionalPlan(plus, minus, scalar, directions);
    }

    private static void requireTieDirections(
            List<Candidate> candidates,
            int[] tieDirections,
            String label
    ) {
        if (tieDirections == null) return;
        if (tieDirections.length != candidates.size()) {
            throw new IllegalArgumentException(
                    label + " tie-direction count does not match candidates");
        }
        for (int direction : tieDirections) {
            if (direction != -1 && direction != 1) {
                throw new IllegalArgumentException(
                        label + " tie directions must be -1 or +1");
            }
        }
    }

    private static boolean verboseAttempts() {
        return Boolean.parseBoolean(
                System.getProperty(
                        VERBOSE_ATTEMPTS_PROPERTY, "false"));
    }

    private static double adjustedDirection(Candidate candidate, int delta, double multiplier) {
        double base = candidate.directionalBaseCost(delta);
        if (JpegMath.isWetCost(base)) return JpegMath.WET_COST;
        double adjusted = base + multiplier * candidate.zeroFlowContribution(delta);
        if (!Double.isFinite(adjusted)) return JpegMath.WET_COST;
        // rho_min^(0)(delta) = lambda_i(delta) * 1e-4 * w_h * w_c: the floor is
        // direction-specific because the zero-crossing multiplier applies only
        // to the 0 <-> +/-1 directions.
        return Math.min(
                JpegMath.WET_COST - 1.0,
                Math.max(candidate.minimumAdjustedCost(delta), adjusted));
    }

    static double finiteDirectionalMedian(
            List<Candidate> headerCandidates,
            List<Candidate> bodyCandidates
    ) {
        int capacity = Math.multiplyExact(2, Math.addExact(
                headerCandidates.size(), bodyCandidates.size()));
        double[] finite = new double[capacity];
        int count = collectFiniteDirectionalCosts(
                headerCandidates, finite, 0);
        count = collectFiniteDirectionalCosts(
                bodyCandidates, finite, count);
        if (count == 0) {
            throw new IllegalArgumentException(
                    "Active carrier set has no finite directional cost");
        }
        if (count >= 4096 && gpuMedianConfigured()) {
            try {
                double median = CudaTrueParallelNative.median(
                        Arrays.copyOf(finite, count));
                GpuStageStats.gpu(GpuStageStats.Stage.MEDIAN);
                return median;
            } catch (RuntimeException | LinkageError failure) {
                if (ExecutionBackend.cudaFailureIsFatal()) throw failure;
                GpuStageStats.fallback(GpuStageStats.Stage.MEDIAN, failure);
                // Exact CPU sort fallback below.
            }
        }
        Arrays.sort(finite, 0, count);
        return (count & 1) == 1
                ? finite[count / 2]
                : 0.5 * (finite[count / 2 - 1] + finite[count / 2]);
    }

    private static boolean gpuMedianConfigured() {
        return ExecutionBackend.cudaPermitted()
                && Boolean.parseBoolean(System.getProperty(
                "rap.gpu.balance.median",
                System.getenv().getOrDefault(
                        "RAP_GPU_BALANCE_MEDIAN", "true")))
                && CudaTrueParallelNative.isAvailable();
    }

    private static int collectFiniteDirectionalCosts(
            List<Candidate> candidates, double[] out, int offset
    ) {
        int n = offset;
        for (Candidate candidate : candidates) {
            if (!JpegMath.isWetCost(candidate.rhoPlus)) out[n++] = candidate.rhoPlus;
            if (!JpegMath.isWetCost(candidate.rhoMinus)) out[n++] = candidate.rhoMinus;
        }
        return n;
    }

    private static TransitionCounts transitionCounts(
            List<Candidate> candidates,
            byte[] cover,
            byte[] stego,
            int[] directions
    ) {
        long zp = 0;
        long zn = 0;
        long pz = 0;
        long nz = 0;
        for (int i = 0; i < candidates.size(); i++) {
            if (cover[i] == stego[i]) continue;
            Candidate candidate = candidates.get(i);
            int delta = directions[i];
            String transition = candidate.transitionClass(delta);
            switch (transition) {
                case "ZP" -> zp++;
                case "ZN" -> zn++;
                case "PZ" -> pz++;
                case "NZ" -> nz++;
                default -> {
                    // The zero-flow constraint is unaffected by PP/NN changes.
                }
            }
        }
        return new TransitionCounts(zp, zn, pz, nz);
    }

    private static TransitionCounts combine(TransitionCounts a, TransitionCounts b) {
        return new TransitionCounts(
                a.zeroToPlus + b.zeroToPlus,
                a.zeroToMinus + b.zeroToMinus,
                a.plusToZero + b.plusToZero,
                a.minusToZero + b.minusToZero);
    }

    private static void requireDimensions(
            List<Candidate> candidates, byte[] cover, byte[] message, String label
    ) {
        if (candidates == null || cover == null || message == null) {
            throw new IllegalArgumentException(label + " inputs cannot be null");
        }
        if (candidates.size() != cover.length) {
            throw new IllegalArgumentException(label + " candidates/cover size mismatch");
        }
        if (message.length == 0) {
            throw new IllegalArgumentException(label + " message cannot be empty");
        }
    }

    private static double clip(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null ? error.getClass().getSimpleName()
                : message.replaceAll("[\\r\\n]+", " ");
    }

    private record GpuBalanceInputs(
            double[] rhoPlus,
            double[] rhoMinus,
            double[] minimumPlus,
            double[] minimumMinus,
            int[] zeroFlowPlus,
            int[] zeroFlowMinus,
            int[] tieDirections
    ) {}

    record DirectionalPlan(
            double[] plusCosts,
            double[] minusCosts,
            double[] scalarCosts,
            int[] directions
    ) {
        DirectionalPlan {
            plusCosts = Arrays.copyOf(plusCosts, plusCosts.length);
            minusCosts = Arrays.copyOf(minusCosts, minusCosts.length);
            scalarCosts = Arrays.copyOf(scalarCosts, scalarCosts.length);
            directions = Arrays.copyOf(directions, directions.length);
        }

        @Override public double[] plusCosts() {
            return Arrays.copyOf(plusCosts, plusCosts.length);
        }
        @Override public double[] minusCosts() {
            return Arrays.copyOf(minusCosts, minusCosts.length);
        }
        @Override public double[] scalarCosts() {
            return Arrays.copyOf(scalarCosts, scalarCosts.length);
        }
        @Override public int[] directions() {
            return Arrays.copyOf(directions, directions.length);
        }

        double[] scalarCostsView() { return scalarCosts; }
        int[] directionsView() { return directions; }
    }
}
