package thesis.rap.app;

import thesis.rap.alg.Candidate;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.payload.KeyMaterial;
import thesis.rap.gpu.CudaTrueParallelNative;
import thesis.rap.runtime.ExecutionBackend;
import thesis.rap.runtime.GpuStageStats;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Domain-separated, key-dependent carrier synchronization.
 *
 * <p>The exact keyed order is bound to both the adaptive QF and the canonical
 * JPEG sampling geometry.  The geometry signature is derived from the actual
 * Y/Cb/Cr sampling factors, so embedding and extraction reconstruct the same
 * context without embedding an extra public geometry field.  The JPEG sampling
 * factors themselves are observable and therefore the signature is not treated
 * as a secret; its security role is domain/context separation and cross-geometry
 * permutation diversity.</p>
 */
public final class StcKeyedOrder {
    private static final int COST_LOCALITY_WINDOW = 64;

    /** Frozen carrier-order domain retained from the methodology. */
    private static final byte[] ORDER_DOMAIN =
            "ahp-order-v1".getBytes(StandardCharsets.UTF_8);

    /** Independent domain for resolving an exact +1/-1 directional-cost tie. */
    private static final byte[] DIRECTION_DOMAIN =
            "ahp-direction-v1".getBytes(StandardCharsets.UTF_8);

    /** q, geometry-signature, c, b, bx, by, u, v, k, q_i. */
    private static final int SERIALIZED_FIELDS = 10;

    private StcKeyedOrder() {}

    /**
     * Applies the keyed within-window permutation for one geometry/QF context.
     */
    public static void apply(
            List<Candidate> candidates,
            String encodedKeys,
            int qf,
            SamplingGeometry geometry
    ) throws Exception {
        requireFrozenBaseCostOrder(candidates);
        validateContext(qf, geometry);
        byte[] key = derivedOrderingKey(encodedKeys);
        try {
            byte[] gpuDigests = gpuDigestsIfEnabled(
                    candidates, key, ORDER_DOMAIN, qf, geometry);
            if (gpuDigests != null) {
                applyDigestVector(candidates, gpuDigests);
                return;
            }

            Mac mac = initializedMac(key);
            ByteBuffer serialized = ByteBuffer.allocate(
                    SERIALIZED_FIELDS * Integer.BYTES);
            for (int from = 0; from < candidates.size();
                 from += COST_LOCALITY_WINDOW) {
                int to = Math.min(candidates.size(),
                        from + COST_LOCALITY_WINDOW);
                java.util.ArrayList<KeyedCandidate> window =
                        new java.util.ArrayList<>(to - from);
                for (int i = from; i < to; i++) {
                    Candidate candidate = candidates.get(i);
                    window.add(new KeyedCandidate(
                            candidate,
                            digestFor(mac, serialized,
                                    ORDER_DOMAIN, qf, geometry, candidate)));
                }
                window.sort(
                        Comparator.<KeyedCandidate, byte[]>comparing(
                                        KeyedCandidate::digest,
                                        StcKeyedOrder::compareBytes)
                                .thenComparing(KeyedCandidate::candidate));
                for (int i = 0; i < window.size(); i++) {
                    candidates.set(from + i, window.get(i).candidate());
                }
            }
        } finally {
            Arrays.fill(key, (byte) 0);
        }
    }

    private static void applyDigestVector(
            List<Candidate> candidates, byte[] digests
    ) {
        if (digests.length != Math.multiplyExact(candidates.size(), 32)) {
            throw new IllegalStateException(
                    "CUDA HMAC digest vector has the wrong length");
        }
        for (int from = 0; from < candidates.size();
             from += COST_LOCALITY_WINDOW) {
            int to = Math.min(candidates.size(), from + COST_LOCALITY_WINDOW);
            java.util.ArrayList<KeyedCandidate> window =
                    new java.util.ArrayList<>(to - from);
            for (int i = from; i < to; i++) {
                window.add(new KeyedCandidate(
                        candidates.get(i),
                        Arrays.copyOfRange(digests, i * 32, i * 32 + 32)));
            }
            window.sort(
                    Comparator.<KeyedCandidate, byte[]>comparing(
                                    KeyedCandidate::digest,
                                    StcKeyedOrder::compareBytes)
                            .thenComparing(KeyedCandidate::candidate));
            for (int i = 0; i < window.size(); i++) {
                candidates.set(from + i, window.get(i).candidate());
            }
        }
    }

    /**
     * Returns the frozen keyed physical direction for an exact directional-cost
     * tie in the same geometry/QF context used for carrier ordering.
     */
    static int[] directionTieBreaks(
            List<Candidate> candidates,
            String encodedKeys,
            int qf,
            SamplingGeometry geometry
    ) {
        if (candidates == null) {
            throw new IllegalArgumentException("candidates cannot be null");
        }
        validateContext(qf, geometry);
        byte[] key = derivedOrderingKey(encodedKeys);
        try {
            byte[] gpuDigests = gpuDigestsIfEnabled(
                    candidates, key, DIRECTION_DOMAIN, qf, geometry);
            if (gpuDigests != null) {
                int[] directions = new int[candidates.size()];
                for (int i = 0; i < directions.length; i++) {
                    directions[i] = (gpuDigests[i * 32] & 1) == 0 ? -1 : 1;
                }
                return directions;
            }

            Mac mac = initializedMac(key);
            ByteBuffer serialized = ByteBuffer.allocate(
                    SERIALIZED_FIELDS * Integer.BYTES);
            int[] directions = new int[candidates.size()];
            for (int i = 0; i < candidates.size(); i++) {
                byte[] digest = digestFor(
                        mac, serialized, DIRECTION_DOMAIN,
                        qf, geometry, candidates.get(i));
                directions[i] = (digest[0] & 1) == 0 ? -1 : 1;
            }
            return directions;
        } finally {
            Arrays.fill(key, (byte) 0);
        }
    }

    private static byte[] gpuDigestsIfEnabled(
            List<Candidate> candidates,
            byte[] key,
            byte[] domain,
            int qf,
            SamplingGeometry geometry
    ) {
        if (!gpuCryptoConfigured() || candidates.isEmpty()) return null;
        int[] fields = new int[Math.multiplyExact(candidates.size(), 8)];
        int offset = 0;
        for (Candidate c : candidates) {
            fields[offset++] = c.component;
            fields[offset++] = c.blockIndex;
            fields[offset++] = c.blockX;
            fields[offset++] = c.blockY;
            fields[offset++] = c.u;
            fields[offset++] = c.v;
            fields[offset++] = c.k;
            fields[offset++] = c.qRef;
        }
        try {
            byte[] digests = CudaTrueParallelNative.hmacCandidateDigests(
                    key, domain, qf, geometry.signatureCode(), fields);
            GpuStageStats.gpu(GpuStageStats.Stage.HMAC_ORDER);
            return digests;
        } catch (RuntimeException | LinkageError failure) {
            if (strictGpu()) throw failure;
            GpuStageStats.fallback(GpuStageStats.Stage.HMAC_ORDER, failure);
            return null;
        }
    }

    private static boolean gpuCryptoConfigured() {
        String value = System.getProperty(
                "rap.gpu.crypto",
                System.getenv().getOrDefault("RAP_GPU_CRYPTO", "true"));
        return ExecutionBackend.cudaPermitted()
                && Boolean.parseBoolean(value)
                && CudaTrueParallelNative.isAvailable();
    }

    private static boolean strictGpu() {
        return ExecutionBackend.cudaFailureIsFatal();
    }

    static void requireFrozenBaseCostOrder(List<Candidate> candidates) {
        if (candidates == null) throw new IllegalArgumentException("candidates cannot be null");
        for (int i = 1; i < candidates.size(); i++) {
            if (candidates.get(i - 1).compareTo(candidates.get(i)) > 0) {
                throw new IllegalArgumentException(
                        "Keyed ordering requires the frozen base-cost sort before 64-candidate windows");
            }
        }
    }

    public static int localityWindow() {
        return COST_LOCALITY_WINDOW;
    }

    private static byte[] derivedOrderingKey(String encodedKeys) {
        try {
            String independentStcKey =
                    KeyMaterial.decode(encodedKeys).stcKey();
            return MessageDigest.getInstance("SHA-256")
                    .digest(("stc-order:" + independentStcKey)
                            .getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static void validateContext(int qf, SamplingGeometry geometry) {
        if (qf < 0) {
            throw new IllegalArgumentException(
                    "Quality factor must be a nonnegative serializable value: " + qf);
        }
        if (geometry == null) {
            throw new IllegalArgumentException("sampling geometry cannot be null");
        }
    }

    private static Mac initializedMac(byte[] key) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(key, "HmacSHA256"));
            return mac;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static byte[] digestFor(
            Mac mac,
            ByteBuffer serialized,
            byte[] domain,
            int qf,
            SamplingGeometry geometry,
            Candidate c
    ) {
        serialized.clear();
        serialized.putInt(qf)
                .putInt(geometry.signatureCode())
                .putInt(c.component).putInt(c.blockIndex)
                .putInt(c.blockX).putInt(c.blockY)
                .putInt(c.u).putInt(c.v).putInt(c.k).putInt(c.qRef);
        mac.update(domain);
        mac.update(serialized.array());
        return mac.doFinal();
    }

    private record KeyedCandidate(Candidate candidate, byte[] digest) {}

    private static int compareBytes(byte[] a, byte[] b) {
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            int cmp = Integer.compare(a[i] & 0xff, b[i] & 0xff);
            if (cmp != 0) return cmp;
        }
        return Integer.compare(a.length, b.length);
    }
}
