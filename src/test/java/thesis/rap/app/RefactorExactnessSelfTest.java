package thesis.rap.app;

import thesis.rap.alg.Candidate;
import thesis.rap.alg.JpegMath;
import thesis.rap.payload.KeyMaterial;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.payload.PayloadPackage;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Exact regression checks for behavior-preserving performance refactors.
 * These tests intentionally reproduce the old slow implementations rather
 * than merely testing the optimized code against itself.
 */
public final class RefactorExactnessSelfTest {
    private static final int WINDOW = 64;
    private static final byte[] ORDER_DOMAIN =
            "ahp-order-v1".getBytes(StandardCharsets.UTF_8);
    private static final byte[] DIRECTION_DOMAIN =
            "ahp-direction-v1".getBytes(StandardCharsets.UTF_8);

    private RefactorExactnessSelfTest() {}

    public static void main(String[] args) throws Exception {
        dctBatchIsBitExactWithScalarReference();
        keyedOrderingMatchesOldReferenceExactly();
        directionTiesMatchOldReferenceExactly();
        preparedPayloadPreservesLayoutAndRoundTrip();
        System.out.println("RefactorExactnessSelfTest: ALL TESTS PASSED");
    }

    private static void dctBatchIsBitExactWithScalarReference() {
        double[][] plane = new double[24][24];
        for (int y = 0; y < plane.length; y++) {
            for (int x = 0; x < plane[0].length; x++) {
                plane[y][x] = (17 * x + 31 * y + (x * y) % 19) & 0xff;
            }
        }
        int[] all = new int[64];
        for (int i = 0; i < all.length; i++) all[i] = i;
        for (int py : new int[]{0, 8, 16}) {
            for (int px : new int[]{0, 8, 16}) {
                double[] batched = JpegMath.forwardDctCoefficients(
                        plane, px, py, all);
                for (int k = 0; k < 64; k++) {
                    double scalar = JpegMath.forwardDctCoefficient(
                            plane, px, py,
                            JpegMath.uFromNaturalIndex(k),
                            JpegMath.vFromNaturalIndex(k));
                    if (Double.doubleToLongBits(scalar)
                            != Double.doubleToLongBits(batched[k])) {
                        throw new AssertionError(
                                "DCT batch changed arithmetic at px=" + px
                                        + " py=" + py + " k=" + k
                                        + " scalar=" + scalar
                                        + " batched=" + batched[k]);
                    }
                }
            }
        }
    }

    private static void keyedOrderingMatchesOldReferenceExactly()
            throws Exception {
        String keys = new KeyMaterial(
                "enc-refactor-order", "xor-refactor-order",
                "stc-refactor-order").encode();
        List<Candidate> source = candidates(193);
        List<Candidate> optimized = new ArrayList<>(source);
        List<Candidate> reference = new ArrayList<>(source);

        StcKeyedOrder.apply(optimized, keys, 91, SamplingGeometry.JPEG_420);
        oldApply(reference, keys, 91, SamplingGeometry.JPEG_420);

        if (optimized.size() != reference.size()) {
            throw new AssertionError("keyed ordering size changed");
        }
        for (int i = 0; i < optimized.size(); i++) {
            if (optimized.get(i) != reference.get(i)) {
                throw new AssertionError(
                        "optimized keyed order differs from old reference at " + i);
            }
        }
    }

    private static void directionTiesMatchOldReferenceExactly() {
        String keys = new KeyMaterial(
                "enc-refactor-dir", "xor-refactor-dir",
                "stc-refactor-dir").encode();
        List<Candidate> candidates = candidates(257);
        int[] optimized = StcKeyedOrder.directionTieBreaks(
                candidates, keys, 87, SamplingGeometry.JPEG_420);
        int[] reference = oldDirectionTieBreaks(
                candidates, keys, 87, SamplingGeometry.JPEG_420);
        if (!Arrays.equals(optimized, reference)) {
            throw new AssertionError(
                    "optimized direction tie HMAC differs from old reference");
        }
    }

    private static void preparedPayloadPreservesLayoutAndRoundTrip()
            throws Exception {
        byte[] message = new byte[8192];
        for (int i = 0; i < message.length; i++) {
            message[i] = (byte) ((i * 29 + (i >>> 3)) & 0xff);
        }
        String keys = new KeyMaterial(
                "enc-refactor-payload", "xor-refactor-payload",
                "stc-refactor-payload").encode();

        PayloadPackage.PayloadLayout oldLayout =
                PayloadPackage.inspectLayout(message);
        PayloadPackage.PreparedMessage prepared =
                PayloadPackage.prepareMessage(message);
        PayloadPackage.PayloadLayout newLayout =
                PayloadPackage.inspectLayout(prepared);
        if (!oldLayout.equals(newLayout)) {
            throw new AssertionError(
                    "prepared-message cache changed payload layout");
        }
        PayloadPackage.PackResult packed =
                PayloadPackage.packDetailed(prepared, keys, 93);
        if (!Arrays.equals(
                message, PayloadPackage.unpack(packed.packed(), keys, 93))) {
            throw new AssertionError(
                    "prepared-message path changed recovered payload bytes");
        }
        if (packed.layout().totalPayloadBytes()
                != oldLayout.totalPayloadBytes()) {
            throw new AssertionError(
                    "prepared-message path changed packed byte accounting");
        }
    }

    private static List<Candidate> candidates(int count) {
        List<Candidate> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int component = 1 + (i & 1);
            int block = i / 8;
            int k = 16 + (i % 32);
            double base = 1.0 + i * 0.001;
            out.add(new Candidate(
                    component, block, block % 17, block / 17,
                    k % 8, k / 8, k, (i % 9) - 4,
                    base, base + 0.0005));
        }
        out.sort(Comparator.naturalOrder());
        return out;
    }

    private static void oldApply(
            List<Candidate> candidates, String encodedKeys, int qf,
            SamplingGeometry geometry
    ) throws Exception {
        StcKeyedOrder.requireFrozenBaseCostOrder(candidates);
        byte[] key = oldDerivedOrderingKey(encodedKeys);
        try {
            for (int from = 0; from < candidates.size(); from += WINDOW) {
                int to = Math.min(candidates.size(), from + WINDOW);
                candidates.subList(from, to).sort(
                        Comparator.<Candidate, byte[]>comparing(
                                        c -> oldDigestFor(
                                                key, ORDER_DOMAIN, qf, geometry, c),
                                        RefactorExactnessSelfTest::compareBytes)
                                .thenComparing(Comparator.naturalOrder()));
            }
        } finally {
            Arrays.fill(key, (byte) 0);
        }
    }

    private static int[] oldDirectionTieBreaks(
            List<Candidate> candidates, String encodedKeys, int qf,
            SamplingGeometry geometry
    ) {
        byte[] key = oldDerivedOrderingKey(encodedKeys);
        try {
            int[] directions = new int[candidates.size()];
            for (int i = 0; i < candidates.size(); i++) {
                byte[] digest = oldDigestFor(
                        key, DIRECTION_DOMAIN, qf, geometry, candidates.get(i));
                directions[i] = (digest[0] & 1) == 0 ? -1 : 1;
            }
            return directions;
        } finally {
            Arrays.fill(key, (byte) 0);
        }
    }

    private static byte[] oldDerivedOrderingKey(String encodedKeys) {
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

    private static byte[] oldDigestFor(
            byte[] key, byte[] domain, int qf, SamplingGeometry geometry, Candidate c
    ) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(key, "HmacSHA256"));
            if (domain != null && domain.length > 0) {
                mac.update(domain);
            }
            mac.update(ByteBuffer.allocate(10 * Integer.BYTES)
                    .putInt(qf)
                    .putInt(geometry.signatureCode())
                    .putInt(c.component)
                    .putInt(c.blockIndex)
                    .putInt(c.blockX)
                    .putInt(c.blockY)
                    .putInt(c.u)
                    .putInt(c.v)
                    .putInt(c.k)
                    .putInt(c.qRef)
                    .array());
            return mac.doFinal();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static int compareBytes(byte[] a, byte[] b) {
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            int cmp = Integer.compare(a[i] & 0xff, b[i] & 0xff);
            if (cmp != 0) return cmp;
        }
        return Integer.compare(a.length, b.length);
    }
}
