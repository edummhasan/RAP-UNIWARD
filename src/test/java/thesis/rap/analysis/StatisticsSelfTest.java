package thesis.rap.analysis;

import thesis.rap.alg.Candidate;
import thesis.rap.alg.JpegMath;
import thesis.rap.app.StcCarrierPlan;
import thesis.rap.app.StcKeyedOrder;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;
import thesis.rap.image.ImageUtil;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.payload.PayloadPackage;
import thesis.rap.payload.KeyMaterial;
import thesis.rap.payload.HuffmanCodec;

import java.nio.charset.StandardCharsets;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;
import javax.crypto.AEADBadTagException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Dependency-free smoke tests. Run with assertions enabled: java -ea ...StatisticsSelfTest */
public final class StatisticsSelfTest {
    private StatisticsSelfTest() {}

    public static void main(String[] args) throws Exception {
        testDistributionIdentity();
        testSmoothedKlManyBinIdentity();
        testQf75PayloadRoundTrip();
        testQfRangeContract();
        testRoundingErrorPolarity();
        testFourModeDiagnostic();
        testWetDirectionSelection();
        testBalancedDirectionTieBreak();
        testComponentSampling420();
        testComponentSamplingClampsOddEdges();
        testIdentityImageMetrics();
        testMismatchedImageSizesRejected();
        testDeterministicPayloadLayout();
        testAdaptiveHuffmanDecision();
        testIndependentKeys();
        testFreshGcmNonce();
        testStrictThreeKeyContract();
        testAuthenticatedHeader();
        testDeterministicHuffmanTies();
        testKeyedCarrierOrdering();
        testStcCarrierPlan();
        testJpegMetadata();
        testCoefficientAnalyzer();
        System.out.println("StatisticsSelfTest: ALL TESTS PASSED");
    }

    private static void testDistributionIdentity() {
        StatisticalMath.DistributionMetrics d = StatisticalMath.compareHistograms(
                Map.of(-1, 10L, 1, 20L), Map.of(-1, 10L, 1, 20L));
        assert close(d.l1(), 0.0);
        assert close(d.klCoverToStego(), 0.0);
        assert close(d.jensenShannon(), 0.0);
        assert close(d.hellinger(), 0.0);
    }

    
    private static void testSmoothedKlManyBinIdentity() {
        Map<Integer, Long> histogram = new LinkedHashMap<>();
        for (int value = -1024; value <= 1024; value++) {
            histogram.put(value, 1L);
        }
        JpegCoefficientAnalyzer.SmoothedDistribution metrics =
                JpegCoefficientAnalyzer.smoothedDistribution(
                        histogram, histogram);
        assert metrics.l1() == 0.0
                : "identical smoothed histograms must have L1=0";
        assert metrics.kl() == 0.0
                : "identical many-bin smoothed histograms must have KL=0";
    }

    private static void testQf75PayloadRoundTrip() throws Exception {
        byte[] message = "QF 75 must be valid".getBytes(StandardCharsets.UTF_8);
        String keys = new KeyMaterial("enc-qf75", "xor-qf75", "stc-qf75").encode();
        byte[] packed = PayloadPackage.pack(message, keys, 75);
        byte[] recovered = PayloadPackage.unpack(packed, keys, 75);
        assert Arrays.equals(message, recovered) : "QF=75 payload round trip failed";
    }

    private static void testQfRangeContract() throws Exception {
        byte[] message = "QF contract".getBytes(StandardCharsets.UTF_8);
        String keys = new KeyMaterial("enc-qf-range", "xor-qf-range", "stc-qf-range").encode();
        boolean rejected = false;
        try { PayloadPackage.pack(message, keys, 74); }
        catch (IllegalArgumentException expected) { rejected = true; }
        assert rejected : "payload format must reject QF values below the algorithmic range";
    }

    private static void testRoundingErrorPolarity() {
        assert close(JpegMath.normalizedRoundingError(23.0, 10, 2), 0.3);
        assert close(JpegMath.normalizedRoundingError(17.0, 10, 2), -0.3);
        assert close(JpegMath.normalizedRoundingError(20.0, 10, 2), 0.0);
        assert close(JpegMath.normalizedRoundingError(40.0, 10, 2), 0.0)
                : "encoder-bin mismatch must disable side information";
        double[] positive = JpegMath.applySideInformedModulation(10.0, 10.0, 0.3);
        assert close(positive[0], 4.0 + JpegMath.SI_COST_FLOOR)
                && close(positive[1], 10.0 + JpegMath.SI_COST_FLOOR);
        double[] negative = JpegMath.applySideInformedModulation(10.0, 10.0, -0.3);
        assert close(negative[0], 10.0 + JpegMath.SI_COST_FLOOR)
                && close(negative[1], 4.0 + JpegMath.SI_COST_FLOOR);
        double[] zero = JpegMath.applySideInformedModulation(10.0, 10.0, 0.0);
        assert close(zero[0], 10.0 + JpegMath.SI_COST_FLOOR)
                && close(zero[1], 10.0 + JpegMath.SI_COST_FLOOR);
    }


    private static void testFourModeDiagnostic() {
        assert JpegMath.isFormerFourModeNearHalf(0, 0, 0.5);
        assert JpegMath.isFormerFourModeNearHalf(0, 4, -0.5);
        assert JpegMath.isFormerFourModeNearHalf(4, 0, 0.5);
        assert JpegMath.isFormerFourModeNearHalf(4, 4, -0.5);
        assert JpegMath.isFormerFourModeNearHalf(
                4, 4, 0.49995);
        assert !JpegMath.isFormerFourModeNearHalf(4, 4, 0.49);
        assert !JpegMath.isFormerFourModeNearHalf(3, 3, 0.5);
        assert Math.abs(
                JpegMath.exactReconstructionErrorIncrement(
                        0.25, 1) - 0.5) < 1.0e-12;
        assert Math.abs(
                JpegMath.exactReconstructionErrorIncrement(
                        0.25, -1) - 1.5) < 1.0e-12;
        assert JpegMath.zigzagIndex(4, 4) == 39 : "retained mode (4,4) must be ZZ0 rank 39";

        Candidate diagnosticCandidate = new Candidate(
                1, 0, 0, 0, 4, 4, JpegMath.naturalIndex(4, 4), 3,
                7.0, 9.0, 0.5, true);
        assert diagnosticCandidate.formerFourModeNearHalf;
        assert !diagnosticCandidate.fullyWet()
                : "the literature four-mode diagnostic must not wet an RAP carrier";
        assert diagnosticCandidate.bestParityFlipValue() == 4
                : "finite RAP directional costs remain actionable";
    }

    private static void testWetDirectionSelection() {
        Candidate plusOnly = new Candidate(1, 0, 0, 0, 1, 1, 9, 1,
                7.0, JpegMath.WET_COST, 0.2, false);
        assert plusOnly.bestParityFlipValue() == 2 : "+1 candidate must not shrink to zero";

        Candidate minusOnly = new Candidate(1, 0, 0, 0, 1, 1, 9, -1,
                JpegMath.WET_COST, 7.0, -0.2, false);
        assert minusOnly.bestParityFlipValue() == -2 : "-1 candidate must not grow to zero";

        Candidate zeroCostLegal = new Candidate(1, 0, 0, 0, 3, 3, 27, 4,
                0.0, 9.0, 0.5, false);
        assert zeroCostLegal.plusLegal() : "legal zero distortion cost must remain legal";
        assert zeroCostLegal.bestParityFlipValue() == 5;
    }

    private static void testBalancedDirectionTieBreak() {
        Candidate a = new Candidate(1, 0, 0, 0, 1, 1, 9, 4, 10.0, 10.0);
        Candidate b = new Candidate(1, 1, 1, 0, 1, 1, 9, 4, 10.0, 10.0);
        assert a.bestParityFlipValue() != b.bestParityFlipValue()
                : "equal-cost ties must not always use the same polarity";
    }

    private static void testComponentSampling420() {
        BufferedImage image = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) {
            int gray = (x + y) & 255;
            image.setRGB(x, y, (gray << 16) | (gray << 8) | gray);
        }
        int[] q = new int[64];
        Arrays.fill(q, 1);
        JpegImage jpeg = new JpegImage(16, 16, new JpegComponent[]{
                new JpegComponent(0, 2, 2, 2, 2, 0, new int[256], q),
                new JpegComponent(1, 1, 1, 1, 1, 1, new int[64], q),
                new JpegComponent(2, 1, 1, 1, 1, 1, new int[64], q)
        });
        double[][][] planes = ImageUtil.yCbCr(image);
        double[][] cb = ImageUtil.jpegComponentPlane(planes, jpeg, 1);
        assert cb.length == 8 && cb[0].length == 8 : "4:2:0 chroma plane must be half resolution";
    }

    private static void testComponentSamplingClampsOddEdges() {
        double[][][] planes = new double[3][1][6];
        for (int component = 0; component < 3; component++) {
            for (int x = 0; x < 6; x++) {
                planes[component][0][x] = x;
            }
        }
        int[] q = new int[64];
        Arrays.fill(q, 1);
        JpegImage jpeg = new JpegImage(6, 1, new JpegComponent[]{
                new JpegComponent(
                        0, 1, 1, 4, 1, 0,
                        new int[64], q),
                new JpegComponent(
                        1, 1, 1, 1, 1, 1,
                        new int[64], q),
                new JpegComponent(
                        2, 1, 1, 1, 1, 1,
                        new int[64], q)
        });
        double[][] cb = ImageUtil.jpegComponentPlane(
                planes, jpeg, 1);
        assert cb.length == 1 && cb[0].length == 2;
        assert close(cb[0][0], 1.5);
        assert close(cb[0][1], 4.75)
                : "partial sampling footprints must repeat the edge sample";
    }

    private static void testIdentityImageMetrics() throws Exception {
        // At least one full 11x11 SSIM window must fit; a smaller fixture used to
        // return a hard-coded 1.0 without measuring anything.
        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                image.setRGB(x, y, ((x * 7) << 16) | ((y * 5) << 8) | ((x + y) & 0xff));
            }
        }
        Path file = Files.createTempFile("rap-identity-", ".png");
        try {
            ImageIO.write(image, "png", file.toFile());
            ImageComparisonMetrics metrics = ImageQualityAnalyzer.compare("identity", file.toString(), file.toString());
            assert close(metrics.mse, 0.0);
            assert Double.isInfinite(metrics.psnr) && metrics.psnr > 0.0
                    : "identical images have zero error and therefore infinite PSNR, not a finite stand-in";
            assert close(metrics.ssim, 1.0);
            assert close(metrics.histogramL1, 0.0);
            assert close(metrics.klDivergence, 0.0) : "identity KL must be exactly zero";
        } finally {
            Files.deleteIfExists(file);
        }
    }

    /** A comparison between differently sized images is not a fidelity measurement. */
    private static void testMismatchedImageSizesRejected() throws Exception {
        Path small = Files.createTempFile("rap-small-", ".png");
        Path large = Files.createTempFile("rap-large-", ".png");
        try {
            ImageIO.write(new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB), "png", small.toFile());
            ImageIO.write(new BufferedImage(48, 32, BufferedImage.TYPE_INT_RGB), "png", large.toFile());
            boolean rejected = false;
            try {
                ImageQualityAnalyzer.compare("mismatch", small.toString(), large.toString());
            } catch (IllegalArgumentException expected) {
                rejected = true;
            }
            assert rejected : "differently sized images must not be silently rescaled and compared";
        } finally {
            Files.deleteIfExists(small);
            Files.deleteIfExists(large);
        }
    }


    private static void testDeterministicPayloadLayout() throws Exception {
        byte[] message = "deterministic payload accounting".getBytes(StandardCharsets.UTF_8);
        PayloadPackage.PayloadLayout estimated = PayloadPackage.inspectLayout(message);
        String keys = new KeyMaterial("enc-layout", "xor-layout", "stc-layout").encode();
        PayloadPackage.PackResult packed = PayloadPackage.packDetailed(message, keys, 87);
        assert estimated.totalPayloadBytes() == packed.packed().length : "payload byte estimate mismatch";
        assert estimated.totalPayloadBits() == packed.packed().length * 8 : "payload bit estimate mismatch";
        assert estimated.totalPayloadBits() == packed.layout().totalPayloadBits() : "layout mismatch";
    }

    private static void testAdaptiveHuffmanDecision() {
        PayloadPackage.PayloadLayout shortLayout = PayloadPackage.inspectLayout(
                "short message".getBytes(StandardCharsets.UTF_8));
        assert !shortLayout.huffmanUsed() : "Huffman must not be used when its serialized form is larger";

        byte[] repetitive = new byte[4096];
        Arrays.fill(repetitive, (byte) 'A');
        PayloadPackage.PayloadLayout repetitiveLayout = PayloadPackage.inspectLayout(repetitive);
        assert repetitiveLayout.huffmanUsed() : "Huffman should be used when its complete serialized form is smaller";
    }

    private static void testIndependentKeys() throws Exception {
        KeyMaterial keys = new KeyMaterial("enc-independent", "xor-independent", "stc-independent");
        byte[] message = "independent keys".getBytes(StandardCharsets.UTF_8);
        byte[] packed = PayloadPackage.packDetailed(message, keys.encryptionKey(), keys.xorKey(), 80).packed();
        assert Arrays.equals(message, PayloadPackage.unpack(packed, keys.encryptionKey(), keys.xorKey(), 80));
        boolean rejected = false;
        try { PayloadPackage.unpack(packed, "wrong-encryption-key", keys.xorKey(), 80); }
        catch (Exception expected) { rejected = true; }
        assert rejected : "wrong independent encryption key must fail authentication";
        rejected = false;
        try {
            PayloadPackage.unpack(
                    packed, keys.encryptionKey(), "wrong-xor-key", 80);
        } catch (AEADBadTagException expected) {
            rejected = true;
        }
        assert rejected
                : "wrong independent XOR key must fail GCM authentication";
    }

    private static void testFreshGcmNonce() throws Exception {
        byte[] message = "fresh nonce probe".getBytes(StandardCharsets.UTF_8);
        byte[] first = PayloadPackage.packDetailed(
                message, "enc-nonce", "xor-nonce", 88).packed();
        byte[] second = PayloadPackage.packDetailed(
                message, "enc-nonce", "xor-nonce", 88).packed();
        int nonceStart = PayloadPackage.QF_LEN
                + PayloadPackage.CIPHERTEXT_LENGTH_FIELD_LEN;
        byte[] firstNonce = Arrays.copyOfRange(
                first, nonceStart, nonceStart + PayloadPackage.NONCE_LEN);
        byte[] secondNonce = Arrays.copyOfRange(
                second, nonceStart, nonceStart + PayloadPackage.NONCE_LEN);
        assert !Arrays.equals(firstNonce, secondNonce)
                : "two fresh random GCM nonces unexpectedly matched";
        assert !Arrays.equals(first, second)
                : "randomized authenticated payloads must not be described "
                        + "as byte-deterministic";
    }

    private static void testStrictThreeKeyContract() {
        boolean rejected = false;
        try { KeyMaterial.decode("single-shared-password"); }
        catch (IllegalArgumentException expected) { rejected = true; }
        assert rejected : "single-password compatibility must be disabled";

        rejected = false;
        try { new KeyMaterial("same", "same", "different"); }
        catch (IllegalArgumentException expected) { rejected = true; }
        assert rejected : "key roles must not reuse identical key material";
    }

    private static void testAuthenticatedHeader() throws Exception {
        byte[] message = "AAD must authenticate trial QF and compact length"
                .getBytes(StandardCharsets.UTF_8);
        byte[] packed = PayloadPackage.packDetailed(
                message, "enc-aad", "xor-aad", 80).packed();

        boolean rejected = false;
        try {
            PayloadPackage.unpack(packed, "enc-aad", "xor-aad", 81);
        } catch (Exception expected) {
            rejected = true;
        }
        assert rejected
                : "wrong trial QF must fail AES-GCM AAD authentication";

        byte[] tamperedLength = Arrays.copyOf(packed, packed.length);
        tamperedLength[PayloadPackage.CIPHERTEXT_LENGTH_FIELD_LEN - 1] ^= 1;
        rejected = false;
        try {
            PayloadPackage.unpack(tamperedLength, "enc-aad", "xor-aad", 80);
        } catch (Exception expected) {
            rejected = true;
        }
        assert rejected
                : "compact ciphertext-length tampering must be rejected";

        byte[] tamperedCiphertext = Arrays.copyOf(packed, packed.length);
        tamperedCiphertext[tamperedCiphertext.length - 1] ^= 1;
        rejected = false;
        try {
            PayloadPackage.unpack(
                    tamperedCiphertext, "enc-aad", "xor-aad", 80);
        } catch (Exception expected) {
            rejected = true;
        }
        assert rejected : "ciphertext tampering must fail AES-GCM authentication";
    }

    private static void testDeterministicHuffmanTies() {
        byte[] equalFrequency = new byte[1024];
        for (int i = 0; i < equalFrequency.length; i++) equalFrequency[i] = (byte) (i & 0xff);
        byte[] first = HuffmanCodec.encode(equalFrequency);
        byte[] second = HuffmanCodec.encode(equalFrequency);
        assert Arrays.equals(first, second) : "equal-frequency Huffman trees must be deterministic";
        assert Arrays.equals(equalFrequency, HuffmanCodec.decode(first));
        assert Arrays.equals(new byte[0], HuffmanCodec.decode(HuffmanCodec.encode(new byte[0])));
    }

    private static void testKeyedCarrierOrdering() throws Exception {
        List<Candidate> first = syntheticCandidates(128);
        List<Candidate> second = syntheticCandidates(128);
        List<Candidate> repeated = syntheticCandidates(128);
        String keysA = new KeyMaterial("enc-order", "xor-order", "stc-order-a").encode();
        String keysB = new KeyMaterial("enc-order", "xor-order", "stc-order-b").encode();
        StcKeyedOrder.apply(first, keysA, 85, SamplingGeometry.JPEG_420);
        StcKeyedOrder.apply(second, keysB, 85, SamplingGeometry.JPEG_420);
        StcKeyedOrder.apply(repeated, keysA, 85, SamplingGeometry.JPEG_420);
        boolean differs = false;
        for (int i = 0; i < first.size(); i++) {
            if (first.get(i).blockIndex != second.get(i).blockIndex) { differs = true; break; }
        }
        assert differs : "different K_stc values must produce different carrier orderings";
        for (int i = 0; i < first.size(); i++) {
            assert first.get(i).blockIndex == repeated.get(i).blockIndex
                    : "the same K_stc must reproduce the same order";
            assert first.get(i).blockIndex / 64 == i / 64
                    : "keyed ordering must preserve 64-carrier cost-locality windows";
        }

        // The methodology serializes the quality factor into the order HMAC, so
        // the same image and key must not reuse one carrier order across the
        // whole 75-100 scan.
        List<Candidate> atQf75 = syntheticCandidates(128);
        List<Candidate> atQf95 = syntheticCandidates(128);
        StcKeyedOrder.apply(atQf75, keysA, 75, SamplingGeometry.JPEG_420);
        StcKeyedOrder.apply(atQf95, keysA, 95, SamplingGeometry.JPEG_420);
        boolean qfChangesOrder = false;
        for (int i = 0; i < atQf75.size(); i++) {
            if (atQf75.get(i).blockIndex != atQf95.get(i).blockIndex) {
                qfChangesOrder = true;
                break;
            }
        }
        assert qfChangesOrder
                : "the quality factor must be part of the keyed carrier order";

        List<Candidate> at420 = syntheticCandidates(128);
        List<Candidate> at444 = syntheticCandidates(128);
        StcKeyedOrder.apply(at420, keysA, 85, SamplingGeometry.JPEG_420);
        StcKeyedOrder.apply(at444, keysA, 85, SamplingGeometry.JPEG_444);
        boolean geometryChangesOrder = false;
        for (int i = 0; i < at420.size(); i++) {
            if (at420.get(i).blockIndex != at444.get(i).blockIndex) {
                geometryChangesOrder = true;
                break;
            }
        }
        assert geometryChangesOrder
                : "the geometry signature must be part of the keyed carrier order";
    }

    private static List<Candidate> syntheticCandidates(int count) {
        List<Candidate> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.add(new Candidate(1 + (i & 1), i, i, 0, 1, 1, 9, 4,
                    10.0 + i, 10.0 + i));
        }
        out.sort(null);
        return out;
    }

    private static void testStcCarrierPlan() {
        int headerBits = PayloadPackage.headerBytes() * 8;
        StcCarrierPlan p = StcCarrierPlan.create(1516, headerBits, 1272 - headerBits);
        assert p.reserveBits() == 32;
        assert p.engineeringMessageCapacityBits() == 1484;
        assert p.totalMessageBits() == 1272;
        assert p.totalCarrierCount() == 1516;
        assert p.headerCarrierCount() == headerBits + 32;
        assert p.bodyCarrierCount() == 1516 - (headerBits + 32);
        assert close(p.overallRelativePayload(), 1272.0 / 1516.0);
    }

    private static void testJpegMetadata() {
        JpegMetadata m = JpegMetadataAnalyzer.analyze(syntheticImage());
        assert m.width() == 16 && m.height() == 16;
        assert m.componentCount() == 3;
        assert "4:4:4".equals(m.subsampling());
        assert m.mcuCount() == 4;
        assert m.yBlockCount() == 4 && m.cbBlockCount() == 4 && m.crBlockCount() == 4;
        assert m.quantizationTableSha256().length() == 64;
    }

    private static void testCoefficientAnalyzer() {
        JpegImage cover = syntheticImage();
        JpegImage stego = cloneImage(cover);
        List<Candidate> candidates = new ArrayList<>();
        int rank = 20;
        int k = JpegMath.naturalIndexFromZigzagRank(rank);

        // Four Cb + four Cr eligible candidates.
        for (int comp = 1; comp <= 2; comp++) {
            for (int block = 0; block < 4; block++) {
                candidates.add(new Candidate(comp, block, block % 2, block / 2,
                        JpegMath.uFromNaturalIndex(k), JpegMath.vFromNaturalIndex(k), k,
                        cover.component(comp).getCoeff(block, k), 1.0, 1.0));
            }
        }

        stego.component(1).setCoeff(0, k, cover.component(1).getCoeff(0, k) + 1);
        stego.component(2).setCoeff(1, k, cover.component(2).getCoeff(1, k) - 1);

        JpegCoefficientReport r = JpegCoefficientAnalyzer.compare(
                cover, stego, 8, 50, candidates, 8, 8,
                true, true, true, true);

        assert r.modifiedCoefficients == 2;
        assert r.plusOneChanges == 1;
        assert r.minusOneChanges == 1;
        assert r.modifiedZeroToNonzero == 0;
        assert r.modifiedNonzeroToZero == 0;
        assert r.positionMetrics.size() == 32;
        DctPositionMetrics p20 = r.positionMetrics.get(
                20 - JpegMath.TARGET_ZZ_START_RANK);
        assert p20.zzRank == 20;
        assert p20.eligibleCandidates == 8;
        assert p20.activeCarriers == 8;
        assert p20.modifiedCoefficients == 2;
        assert close(p20.modificationRateEligible, 0.25);
        assert r.targetBandJensenShannon >= 0.0;
    }

    private static JpegImage syntheticImage() {
        JpegComponent[] components = new JpegComponent[3];
        for (int comp = 0; comp < 3; comp++) {
            int[] coeffs = new int[4 * 64];
            int[] q = new int[64];
            Arrays.fill(q, 1);
            for (int block = 0; block < 4; block++) {
                for (int rank = JpegMath.TARGET_ZZ_START_RANK; rank <= JpegMath.TARGET_ZZ_END_RANK; rank++) {
                    int k = JpegMath.naturalIndexFromZigzagRank(rank);
                    coeffs[block * 64 + k] = (comp + 1) * 2 + (rank % 3);
                }
            }
            components[comp] = new JpegComponent(
                    comp, 2, 2, 1, 1, comp, coeffs, q);
        }
        return new JpegImage(16, 16, components);
    }

    private static JpegImage cloneImage(JpegImage in) {
        JpegComponent[] components = new JpegComponent[in.components.length];
        for (int i = 0; i < components.length; i++) {
            JpegComponent c = in.components[i];
            components[i] = new JpegComponent(
                    c.componentIndex, c.widthBlocks, c.heightBlocks,
                    c.hSamplingFactor, c.vSamplingFactor,
                    c.quantTableIndex,
                    Arrays.copyOf(
                            c.coefficients, c.coefficients.length),
                    Arrays.copyOf(
                            c.quantTable, c.quantTable.length));
        }
        return new JpegImage(in.width, in.height, components);
    }

    private static boolean close(double a, double b) {
        return Math.abs(a - b) < 1e-10;
    }
}
