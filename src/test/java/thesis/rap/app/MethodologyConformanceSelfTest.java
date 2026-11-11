package thesis.rap.app;

import thesis.rap.alg.Candidate;
import thesis.rap.alg.CandidateSet;
import thesis.rap.alg.JpegMath;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;
import thesis.rap.payload.HuffmanCodec;
import thesis.rap.payload.KeyMaterial;
import thesis.rap.payload.PayloadPackage;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.List;

/** Dependency-free regression tests for the normative six-algorithm invariants. */
public final class MethodologyConformanceSelfTest {
    private MethodologyConformanceSelfTest() {}

    public static void main(String[] args) throws Exception {
        testFrozenCostSortBeforeKeyedWindows();
        testKeyedDirectionalTieBreak();
        testCandidateUniverseAndDrySetSeparation();
        testFullAdaptiveQfRangeOnly();
        testStrictCliOptions();
        testAuditProfilesAndDiskParsing();
        testResultOnlyAuditSchema();
        testMarkerlessAuthenticatedPayloadContract();
        testPayloadByteEqualityAndOverheadAcrossMessages();
        testUnambiguousBatchOutputNames();
        testBodyCarrierCap();
        testAdaptiveQfContinuesAfterKernelFailure();
        testAdditiveMultiplierReconsidersBothDirections();
        testBalanceRetriesRestartBothStcCalls();
        testBalanceFixedPointStopsRetries();
        testBalanceFailureSerialization();
        testWetDirectionRemainsWet();
        testStrictHuffmanBoundaries();
        testEmptyHuffmanUsesNormativeLayout();
        testJpegModelRejectsMalformedArrays();
        testFailureAuditBundle();
        System.out.println("MethodologyConformanceSelfTest: ALL TESTS PASSED");
    }

    private static void testFrozenCostSortBeforeKeyedWindows() throws Exception {
        Candidate low = candidate(1, 0, 2, 1.0, 2.0);
        Candidate high = candidate(1, 1, 2, 5.0, 6.0);
        List<Candidate> unsorted = new ArrayList<>(List.of(high, low));
        boolean rejected = false;
        try {
            StcKeyedOrder.apply(
                    unsorted,
                    new KeyMaterial("enc-sort", "xor-sort", "stc-sort").encode(),
                    85, SamplingGeometry.JPEG_420);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected : "keyed windows must reject a list not frozen in base-cost order";

        List<Candidate> sorted = new ArrayList<>(List.of(low, high));
        StcKeyedOrder.apply(
                sorted,
                new KeyMaterial("enc-sort", "xor-sort", "stc-sort").encode(),
                85, SamplingGeometry.JPEG_420);
        assert sorted.size() == 2;
    }

    private static void testKeyedDirectionalTieBreak() {
        List<Candidate> candidates = new ArrayList<>();
        for (int i = 0; i < 128; i++) {
            candidates.add(candidate(1 + (i & 1), i, 2, 10.0, 10.0));
        }
        String keysA = new KeyMaterial(
                "enc-direction", "xor-direction", "stc-direction-a").encode();
        String keysB = new KeyMaterial(
                "enc-direction", "xor-direction", "stc-direction-b").encode();
        int[] first = StcKeyedOrder.directionTieBreaks(
                candidates, keysA, 85, SamplingGeometry.JPEG_420);
        int[] repeated = StcKeyedOrder.directionTieBreaks(
                candidates, keysA, 85, SamplingGeometry.JPEG_420);
        int[] otherKey = StcKeyedOrder.directionTieBreaks(
                candidates, keysB, 85, SamplingGeometry.JPEG_420);
        int[] otherQf = StcKeyedOrder.directionTieBreaks(
                candidates, keysA, 95, SamplingGeometry.JPEG_420);
        int[] otherGeometry = StcKeyedOrder.directionTieBreaks(
                candidates, keysA, 85, SamplingGeometry.JPEG_444);
        assert Arrays.equals(first, repeated)
                : "same key/QF must reproduce physical-direction ties";
        assert !Arrays.equals(first, otherKey)
                : "K_stc must affect physical-direction ties";
        assert !Arrays.equals(first, otherQf)
                : "QF must affect physical-direction ties";
        assert !Arrays.equals(first, otherGeometry)
                : "geometry signature must affect physical-direction ties";
        for (int i = 0; i < first.length; i++) {
            assert first[i] == -1 || first[i] == 1;
            assert candidates.get(i).bestParityFlipDelta(
                    10.0, 10.0, first[i]) == first[i];
        }
    }

    private static void testCandidateUniverseAndDrySetSeparation() {
        Candidate low = candidate(1, 0, 0, 1.0, 2.0);
        Candidate high = candidate(2, 1, 1, 3.0, 4.0);
        Candidate wet = candidate(
                2, 2, 0, JpegMath.WET_COST, JpegMath.WET_COST);
        CandidateSet set = new CandidateSet(
                List.of(low, high, wet), List.of(low, high));
        assert set.universeCount() == 3;
        assert set.dryCount() == 2;
        assert set.fullyWetCount() == 1;
        assert set.universe().contains(wet);
        assert !set.dryBaseCostSorted().contains(wet);

        boolean rejected = false;
        try {
            new CandidateSet(
                    List.of(low, candidate(1, 0, 4, 5.0, 6.0)),
                    List.of(low));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected
                : "logical duplicate carrier addresses must be rejected";
    }

    private static void testBodyCarrierCap() {
        int headerBits = PayloadPackage.headerBytes() * 8;
        int bodyBits = 10;
        StcCarrierPlan plan =
                StcCarrierPlan.create(100_000, headerBits, bodyBits);
        assert plan.headerCarrierCount() == 240;
        assert plan.bodyCarrierCount()
                == bodyBits * StcCarrierPlan.BODY_WIDTH_CAP;
        assert plan.totalCarrierCount()
                == 240 + bodyBits * StcCarrierPlan.BODY_WIDTH_CAP;
        assert plan.bodyCarrierCount() < 100_000 - 240
                : "the body must not consume every remaining candidate";
    }

    private static void testMarkerlessAuthenticatedPayloadContract()
            throws Exception {
        assert PayloadPackage.PUBLIC_MARKER_LEN == 0;
        assert PayloadPackage.headerBytes() == 26
                : "compact length/nonce/tag header must be 26 bytes";
        assert PayloadPackage.AAD_LEN == 60
                : "non-geometry AAD must remain 60 authenticated bytes";
        assert PayloadPackage.GEOMETRY_AAD_LEN == 72
                : "production geometry-bound AAD must remain 72 authenticated bytes";
        assert PayloadPackage.QF_LEN == 0
                : "RAP-UNIWARD must not embed QF in the compact wire header";
        assert PayloadPackage.CIPHERTEXT_LENGTH_FIELD_LEN == 2;
        assert PayloadPackage.NONCE_LEN == 12;
        assert PayloadPackage.TAG_LEN == 12;
        byte[] message = new byte[]{0, 1, 2, (byte) 0xff, 4};
        String keys = new KeyMaterial(
                "enc-markerless", "xor-markerless",
                "stc-markerless").encode();
        PayloadPackage.PackResult result =
                PayloadPackage.packDetailed(
                        message, keys, 83, SamplingGeometry.JPEG_420);
        byte[] packed = result.packed();
        assert result.layout().protocolMarkerBytes() == 0;
        assert result.layout().fixedHeaderBytes() == 26;
        assert result.layout().totalPayloadBytes()
                == result.layout().preparedPlaintextBytes() + 26;
        assert !(packed.length >= 4
                && packed[0] == 'A'
                && packed[1] == 'H'
                && packed[2] == 'P')
                : "wire payload must not contain the ASCII protocol prefix";
        assert Arrays.equals(
                message, PayloadPackage.unpack(
                        packed, keys, 83, SamplingGeometry.JPEG_420));
        boolean wrongGeometryRejected = false;
        try {
            PayloadPackage.unpack(
                    packed, keys, 83, SamplingGeometry.JPEG_444);
        } catch (java.security.GeneralSecurityException expected) {
            wrongGeometryRejected = true;
        }
        assert wrongGeometryRejected
                : "AES-GCM AAD must reject a wrong geometry signature";
    }

    private static void
    testPayloadByteEqualityAndOverheadAcrossMessages()
            throws Exception {
        byte[] repetitive = new byte[4096];
        Arrays.fill(repetitive, (byte) 'A');
        List<byte[]> messages = List.of(
                new byte[0],
                new byte[]{
                        0, 1, 2, 3, 0x7f, (byte) 0x80,
                        (byte) 0xfe, (byte) 0xff
                },
                repetitive);
        String keys = new KeyMaterial(
                "enc-byte-equality", "xor-byte-equality",
                "stc-byte-equality").encode();
        for (byte[] message : messages) {
            PayloadPackage.PayloadLayout estimated =
                    PayloadPackage.inspectLayout(message);
            PayloadPackage.PackResult result =
                    PayloadPackage.packDetailed(message, keys, 91);
            PayloadPackage.PayloadLayout observed = result.layout();
            assert Arrays.equals(
                    message,
                    PayloadPackage.unpack(result.packed(), keys, 91))
                    : "payload bytes must round-trip exactly";
            assert result.packed().length
                    == estimated.totalPayloadBytes();
            assert observed.totalPayloadBytes()
                    == observed.contentBytes() + 35
                    : "payload must equal content plus 26-byte header "
                            + "and 9-byte preprocessing metadata";
            assert observed.fixedFramingOverheadBytes() == 35;
            assert observed.netPayloadExpansionBytes()
                    == observed.totalPayloadBytes()
                    - observed.rawMessageBytes();
            assert observed.protocolMarkerBytes() == 0;
            assert observed.fixedHeaderBytes() == 26;
        }
    }

    private static void testFullAdaptiveQfRangeOnly() {
        RapUniward.QfRange full = RapUniward.qfRange(Map.of());
        assert full.minQf() == 75 && full.maxQf() == 100;
        boolean rejected = false;
        try {
            RapUniward.qfRange(Map.of(
                    "--minQf", "76", "--maxQf", "100"));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected
                : "adaptive execution must not permit a narrowed QF range";
    }

    private static void testStrictCliOptions() throws Exception {
        assertRejected(
                () -> RapUniward.main(new String[]{
                        "summarize", "--runz", "misspelled.csv"
                }),
                "unknown CLI options must be rejected");
        assertRejected(
                () -> RapUniward.main(new String[]{
                        "summarize",
                        "--runs", "first.csv",
                        "--runs", "second.csv"
                }),
                "duplicate CLI options must be rejected");
        assertRejected(
                () -> RapUniward.main(new String[]{"not-a-mode"}),
                "unknown CLI modes must be rejected");

        Path root = Files.createTempDirectory("rap-cli-strict-");
        Path pngDir = Files.createDirectory(root.resolve("png"));
        Path msgDir = Files.createDirectory(root.resolve("msg"));
        Path outDir = root.resolve("out");
        try {
            assertRejected(
                    () -> RapUniward.main(new String[]{
                            "batch-embed",
                            "--pngDir", pngDir.toString(),
                            "--msgDir", msgDir.toString(),
                            "--outDir", outDir.toString(),
                            "--recursive", "yes"
                    }),
                    "boolean options must be exactly true or false");
            assertRejected(
                    () -> RapUniward.main(new String[]{
                            "batch-embed",
                            "--pngDir", pngDir.toString(),
                            "--msgDir", msgDir.toString(),
                            "--outDir", msgDir.resolve("output").toString()
                    }),
                    "batch output must not overlap an input tree");
        } finally {
            Files.deleteIfExists(outDir);
            Files.deleteIfExists(pngDir);
            Files.deleteIfExists(msgDir);
            Files.deleteIfExists(root);
        }
    }

    private static void testUnambiguousBatchOutputNames() {
        Path pngRoot = Path.of("/tmp/png-root");
        Path messageRoot = Path.of("/tmp/message-root");
        String first = RapUniward.batchOutputName(
                pngRoot, pngRoot.resolve("a_b.png"),
                messageRoot, messageRoot.resolve("c.msg"));
        String second = RapUniward.batchOutputName(
                pngRoot, pngRoot.resolve("a.png"),
                messageRoot, messageRoot.resolve("b_c.msg"));
        assert !first.equals(second)
                : "length-prefixed batch names must separate image/message keys";
    }

    private static void testAuditProfilesAndDiskParsing()
            throws Exception {
        assert AuditMode.parse("regular") == AuditMode.RESULT_ONLY;
        assert AuditMode.parse("result-only") == AuditMode.RESULT_ONLY;
        assert AuditMode.parse("qf-forensic") == AuditMode.QF_FORENSIC;
        assert !AuditMode.RESULT_ONLY.writesQfForensics();
        assert AuditMode.QF_FORENSIC.writesQfForensics();
        assertRejected(
                () -> AuditMode.parse("mixed"),
                "unknown audit modes must fail closed");
        assert AuditProfile.parse("thesis") == AuditProfile.THESIS;
        assert AuditProfile.parse("changes") == AuditProfile.CHANGES;
        assert AuditProfile.parse("full-audit")
                == AuditProfile.FULL_AUDIT;
        assert !AuditProfile.THESIS.writesChanges();
        assert AuditProfile.CHANGES.writesChanges();
        assert AuditProfile.FULL_AUDIT
                .writesFullCandidateUniverse();
        assertRejected(
                () -> AuditProfile.parse("debug"),
                "unknown audit profiles must fail closed");

        String df = """
                Filesystem Inodes IUsed IFree IUse% Mounted on
                /dev/test 1000000 250000 750000 25% /data
                """;
        assert DiskSpaceGuard.parseAvailableInodes(df) == 750_000L;
        assertRejected(
                () -> new DiskSpaceGuard.Limits(0L, 0L, 0),
                "disk-check interval must be positive");
    }

    private static void testResultOnlyAuditSchema() throws Exception {
        Path root = Files.createTempDirectory("rap-result-audit-");
        Path ledger = root.resolve("result.csv");
        try {
            FinalQfAuditState terminal = new FinalQfAuditState();
            terminal.begin(
                    100, 8_192,
                    ThreeStateEmbeddingEngine.Profile.BOUNDED);
            terminal.finish(
                    "CAPACITY_REJECTED",
                    "dry_count_minus_32_below_required_bits", false);
            ResultAuditWriter.Entry entry =
                    ResultAuditWriter.Entry.builder()
                            .identity(
                                    "2026-01-01T00:00:00Z",
                                    "2026-01-01T00:00:01Z",
                                    "run-0001", "test-version",
                                    "EMBED_AND_AUTHENTICATED_EXTRACT")
                            .paths(
                                    Path.of("000001.png"),
                                    Path.of("001.msg"),
                                    Path.of("result.jpg"))
                            .qfSearch(75, 100, "capacity")
                            .outcome(
                                    "UNABLE_TO_EMBED_OR_EXTRACT",
                                    false, false, false, false,
                                    false, false)
                            .hashes(
                                    "image-hash", "message-hash", "")
                            .failure(
                                    "NO_VERIFIED_QF",
                                    "No verified output")
                            .finalQf(terminal)
                            .evaluationUnavailable(
                                    "NO_VERIFIED_QF_HAS_NO_STEGO_EVALUATION",
                                    1_000L)
                            .build();
            ResultAuditWriter.append(ledger, entry);
            List<Map<String, String>> rows =
                    thesis.rap.analysis.CsvUtil.read(ledger);
            assert rows.size() == 1;
            Map<String, String> row = rows.get(0);
            assert "RESULT_ONLY".equals(row.get("audit_mode"));
            assert "ONE_IMAGE_MESSAGE_GEOMETRY_ONE_TERMINAL_QF".equals(
                    row.get("record_model"));
            assert "000001.png".equals(row.get("image_name"));
            assert "000001".equals(row.get("image_id_text"));
            assert "001.msg".equals(row.get("message_name"));
            assert "001".equals(row.get("message_id_text"));
            assert "75".equals(row.get("qf_min"));
            assert "100".equals(row.get("qf_max"));
            assert "100".equals(row.get("terminal_qf"));
            assert "LAST_TESTED".equals(row.get("terminal_qf_role"));
            assert "CAPACITY_REJECTED".equals(
                    row.get("terminal_qf_status"));
            assert "false".equals(row.get("qf_history_retained"));
            assert "false".equals(row.get("balance_history_retained"));
            assert "false".equals(
                    row.get("final_qf_evaluation_available"));
            assert row.containsKey("reference_stego_psnr");
            assert row.get("reference_stego_psnr").isEmpty();
            assert row.containsKey("cb_modified_coefficients");
            assert row.containsKey("cr_modified_coefficients");
        } finally {
            Files.deleteIfExists(
                    ledger.resolveSibling(ledger.getFileName() + ".lock"));
            Files.deleteIfExists(ledger);
            Files.deleteIfExists(root);
        }
    }

    private static void testAdaptiveQfContinuesAfterKernelFailure()
            throws Exception {
        AdaptiveQfSearch.Result<String> result =
                AdaptiveQfSearch.firstVerified(75, 78, qf -> switch (qf) {
                    case 75 -> AdaptiveQfSearch.Trial.capacityRejected(
                            qf, "capacity");
                    case 76 -> AdaptiveQfSearch.Trial.trialRejected(
                            qf, "balance");
                    case 77 -> AdaptiveQfSearch.Trial.verified(qf, "ok");
                    default -> throw new AssertionError("scan did not stop");
                });
        assert result.verified();
        assert result.selectedQf() == 77;
        assert result.trials().size() == 3;
        assert result.trials().get(1).status()
                == AdaptiveQfSearch.Status.TRIAL_REJECTED;
    }

    private static void testAdditiveMultiplierReconsidersBothDirections() {
        Candidate carrier = candidate(1, 0, 1, 3.0, 1.0);
        ThreeStateEmbeddingEngine.DirectionalPlan base =
                ThreeStateEmbeddingEngine.adjustedPlan(
                        List.of(carrier), 0.0);
        ThreeStateEmbeddingEngine.DirectionalPlan penalized =
                ThreeStateEmbeddingEngine.adjustedPlan(
                        List.of(carrier), 5.0);
        assert base.directions()[0] == -1
                : "base cost should choose +1->0 shrinkage";
        assert penalized.directions()[0] == 1
                : "a retry must be able to switch to +1->+2";
        assert close(penalized.minusCosts()[0], 6.0);
        assert close(penalized.plusCosts()[0], 3.0);
    }

    private static void testBalanceRetriesRestartBothStcCalls() {
        Candidate header = candidate(1, 0, 1, 1.10, 1.00);
        Candidate body = candidate(2, 1, 2, 1.00, 1.00);
        CountingFlipBackend backend = new CountingFlipBackend();
        ThreeStateEmbeddingEngine.Result result =
                ThreeStateEmbeddingEngine.embed(
                        backend,
                        List.of(header), new byte[]{0}, new byte[]{0},
                        List.of(body), new byte[]{0}, new byte[]{0},
                        75, ThreeStateEmbeddingEngine.Profile.STRICT);
        assert result.iterations() >= 2
                : "strict balance should require a multiplier retry";
        assert backend.embedCalls == 2 * result.iterations()
                : "every retry must rerun both header and body STC";
        assert result.zeroBinDelta() == 0;
        assert result.headerDirections()[0] == 1
                : "retry must switch +1 shrinkage to +1 growth";
    }

    private static void testWetDirectionRemainsWet() {
        Candidate carrier = candidate(
                1, 0, 2, JpegMath.WET_COST, 2.0);
        ThreeStateEmbeddingEngine.DirectionalPlan plan =
                ThreeStateEmbeddingEngine.adjustedPlan(
                        List.of(carrier), -1.0e12);
        assert JpegMath.isWetCost(plan.plusCosts()[0]);
        assert plan.directions()[0] == -1;
    }

    private static void testBalanceFixedPointStopsRetries() {
        List<Candidate> header = new ArrayList<>();
        List<Candidate> body = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            header.add(candidate(1, i, 0, 1.0, 1.0));
            body.add(candidate(2, i + 50, 0, 1.0, 1.0));
        }
        CountingFlipBackend backend = new CountingFlipBackend();
        boolean rejected = false;
        try {
            ThreeStateEmbeddingEngine.embed(
                    backend,
                    header, new byte[header.size()], new byte[]{0},
                    body, new byte[body.size()], new byte[]{0},
                    75, ThreeStateEmbeddingEngine.Profile.BOUNDED);
        } catch (ThreeStateEmbeddingEngine.NoFeasibleBalanceException
                 expected) {
            rejected = true;
            assert "BALANCE_BOUND_REPEATED".equals(
                    expected.reasonCode());
            assert expected.attempts().size() == 2
                    : "a deterministic bounded state must not run "
                            + "all 40 attempts";
            assert "BALANCE_BOUND_REPEATED".equals(
                    expected.attempts().get(1).status());
        }
        assert rejected
                : "out-of-budget fixed point must fail closed";
    }

    private static void testBalanceFailureSerialization() throws Exception {
        ThreeStateEmbeddingEngine.Attempt attempt =
                new ThreeStateEmbeddingEngine.Attempt(
                        75, 0, 1.0, 0.0, 0.08,
                        false, false,
                        new ThreeStateEmbeddingEngine.TransitionCounts(
                                0, 0, 0, 0),
                        0, false, false,
                        "STC_FAILED", "synthetic");
        ThreeStateEmbeddingEngine.NoFeasibleBalanceException failure =
                new ThreeStateEmbeddingEngine.NoFeasibleBalanceException(
                        "synthetic", List.of(attempt));

        byte[] encoded;
        try (ByteArrayOutputStream bytes = new ByteArrayOutputStream();
             ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(failure);
            encoded = bytes.toByteArray();
        }

        Object decoded;
        try (ObjectInputStream in = new ObjectInputStream(
                new ByteArrayInputStream(encoded))) {
            decoded = in.readObject();
        }
        assert decoded instanceof
                ThreeStateEmbeddingEngine.NoFeasibleBalanceException;
        ThreeStateEmbeddingEngine.NoFeasibleBalanceException restored =
                (ThreeStateEmbeddingEngine.NoFeasibleBalanceException) decoded;
        assert restored.attempts().size() == 1;
        assert restored.attempts().get(0).equals(attempt);
        assert "BALANCE_BUDGET_REJECTED".equals(
                restored.reasonCode());
    }

    private static void testStrictHuffmanBoundaries() {
        byte[] encoded = HuffmanCodec.encode(new byte[]{7, 7, 7});
        byte[] trailing = Arrays.copyOf(encoded, encoded.length + 1);
        boolean rejected = false;
        try {
            HuffmanCodec.decode(trailing);
        } catch (IllegalStateException expected) {
            rejected = true;
        }
        assert rejected : "trailing Huffman bytes must be rejected";

        int totalBits = ByteBuffer.wrap(encoded)
                .order(ByteOrder.BIG_ENDIAN)
                .getInt(4 + 256 * Integer.BYTES);
        assert totalBits % 8 != 0;
        byte[] nonzeroPadding = Arrays.copyOf(encoded, encoded.length);
        nonzeroPadding[nonzeroPadding.length - 1] |= 1;
        rejected = false;
        try {
            HuffmanCodec.decode(nonzeroPadding);
        } catch (IllegalStateException expected) {
            rejected = true;
        }
        assert rejected : "nonzero Huffman padding must be rejected";

        byte[] wrongSingleSymbolPath =
                HuffmanCodec.encode(new byte[]{7, 7, 7});
        wrongSingleSymbolPath[wrongSingleSymbolPath.length - 1]
                = (byte) 0b1110_0000;
        rejected = false;
        try {
            HuffmanCodec.decode(wrongSingleSymbolPath);
        } catch (IllegalStateException expected) {
            rejected = true;
        }
        assert rejected
                : "decoded symbols must match the serialized frequency table";

        byte[] zeroSymbolPath =
                HuffmanCodec.encode(new byte[]{0, 0, 0});
        zeroSymbolPath[zeroSymbolPath.length - 1]
                = (byte) 0b1110_0000;
        rejected = false;
        try {
            HuffmanCodec.decode(zeroSymbolPath);
        } catch (IllegalStateException expected) {
            rejected = true;
        }
        assert rejected
                : "the synthetic single-symbol branch must never decode";
    }

    private static void testEmptyHuffmanUsesNormativeLayout() {
        byte[] encoded = HuffmanCodec.encode(new byte[0]);
        assert encoded.length == 4 + 256 * Integer.BYTES + 4
                : "H(empty) must retain the complete deterministic header";
        assert HuffmanCodec.decode(encoded).length == 0;

        byte[] oldShortcut = new byte[4];
        boolean rejected = false;
        try {
            HuffmanCodec.decode(oldShortcut);
        } catch (IllegalStateException expected) {
            rejected = true;
        }
        assert rejected
                : "the obsolete four-byte empty shortcut must be rejected";
    }

    private static void testJpegModelRejectsMalformedArrays() {
        boolean rejected = false;
        try {
            new thesis.rap.jni.JpegComponent(
                    1, 2, 2, 1, 1, 1,
                    new int[63], unitQuantTable());
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected : "malformed coefficient arrays must be rejected";

        rejected = false;
        try {
            int[] invalidQuant = unitQuantTable();
            invalidQuant[0] = 0;
            new thesis.rap.jni.JpegComponent(
                    1, 1, 1, 1, 1, 1,
                    new int[64], invalidQuant);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected : "invalid baseline quantization values must be rejected";
    }

    private static void testFailureAuditBundle() throws Exception {
        Path inputs = Files.createTempDirectory(
                "rap-audit-inputs-");
        Path png = inputs.resolve("cover.png");
        Path message = inputs.resolve("message.msg");
        Files.write(png, new byte[]{1, 2, 3});
        Files.write(
                message, "audit".getBytes(StandardCharsets.UTF_8));
        String runId = "selftest-" + System.nanoTime();
        Path audit = RunAuditWriter.writeFailure(
                runId, "1.0.0.1",
                "2026-07-24T00:00:00Z",
                png, message,
                ThreeStateEmbeddingEngine.Profile.BOUNDED,
                PayloadPackage.inspectLayout(
                        Files.readAllBytes(message)),
                List.of(), List.of(),
                10,
                "NO_VERIFIED_QF", "self-test",
                AuditProfile.THESIS,
                SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_422);
        List<String> names = List.of(
                "run.json", "qf_trials.csv",
                "balance_attempts.csv", "candidates.csv",
                "jpeg_statistics.csv", "verification.json");
        try {
            for (String name : names) {
                assert Files.isRegularFile(audit.resolve(name))
                        : "mandatory audit artifact missing: " + name;
            }
            String run = Files.readString(audit.resolve("run.json"));
            assert run.contains("\"schema_version\": 1");
            assert run.contains(
                    "\"geometry_mode\": \"ALL_GEOMETRY\"");
            assert run.contains(
                    "\"sampling_geometry\": \"4:2:2\"");
            assert run.contains(
                    "\"sampling_factors_expected\": "
                            + "\"Y=2x1;Cb=1x1;Cr=1x1\"");
            assert run.contains("\"audit_profile\": \"THESIS\"");
            assert run.contains(
                    "\"payload_protocol_marker_bytes\": 0");
            assert run.contains("\"payload_header_bytes\": 26");
            assert run.contains("\"header_bits\": 208");
            assert run.contains("\"header_carriers\": 240");
            assert run.contains("\"gcm_aad_bytes\": "
                    + PayloadPackage.GEOMETRY_AAD_LEN);
            assert run.contains(
                    "\"payload_protocol_version_transport\": "
                            + "\"out-of-band-application-version\"");
            assert run.contains(
                    "\"si_uniward_stego_generated_by_this_run\": "
                            + "false");
            assert run.contains("\"target_zz0_start\": 16");
            assert run.contains("\"target_zz0_end\": 47");
            assert run.contains("\"stc_height\": 10");
            assert run.contains(
                    "\"stc_implementation_id\": "
                            + "\"rap-portable-scalar-binary64-stc-height10\"");
            assert run.contains(
                    "\"stc_numeric_profile\": "
                            + "\"PORTABLE_SCALAR_BINARY64_STC_V1\"");
            assert run.contains(
                    "\"stc_metric_array_bytes_per_call\": 16384");
            assert run.contains(
                    "\"statistical_cost_model\": "
                            + "\"CBCR-MARGINAL-RISK-V1\"");
            assert run.contains("\"keyed_order_window\": 64");
            assert run.contains(
                    "\"jpeg_profile_metadata_available\": false");
            assert run.contains(
                    "\"reference_luminance_reconstruction\": "
                            + "\"canonical-quantized-y|"
                            + "ideal-orthonormal-idct|level-shift-128|"
                            + "clip-0-255|crop-to-image\"");
            assert occurrences(run, "\"bounded_zero_budget\"") == 1
                    : "run.json cannot contain duplicate field names";
            String qfTrials = Files.readString(
                    audit.resolve("qf_trials.csv"));
            assert qfTrials.contains("header_gibbs_lambda");
            assert qfTrials.contains("body_gibbs_lambda");
            String candidates = Files.readString(
                    audit.resolve("candidates.csv"));
            assert candidates.contains("reference_state");
            assert candidates.contains("modified_count");
            assert candidates.contains("base_min_cost_p50");
            assert candidates.contains("candidate_stream_sha256");
            assert !candidates.contains("raw_dct")
                    : "THESIS candidate evidence must be compact";
            String verification = Files.readString(
                    audit.resolve("verification.json"));
            assert verification.contains(
                    "\"overall_verified\": false");
            assert verification.contains(
                    "\"failure_reason_code\": \"NO_VERIFIED_QF\"");
            assert verification.contains(
                    "\"message_byte_equality_reason_code\": "
                            + "\"NOT_VERIFIED\"");
            assert verification.contains(
                    "\"header_decode_verified\": false");
            assert verification.contains(
                    "\"body_decode_verified\": false");
        } finally {
            for (String name : names) {
                Files.deleteIfExists(audit.resolve(name));
            }
            Files.deleteIfExists(audit);
            Files.deleteIfExists(png);
            Files.deleteIfExists(message);
            Files.deleteIfExists(inputs);
        }
    }

    private static Candidate candidate(
            int component,
            int block,
            int qRef,
            double rhoPlus,
            double rhoMinus
    ) {
        return new Candidate(
                component, block, block, 0,
                0, 5, 40, qRef,
                rhoPlus, rhoMinus,
                0.0, false,
                0.0, 1, 1.0,
                1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0e-4);
    }

    private static int[] unitQuantTable() {
        int[] quant = new int[64];
        Arrays.fill(quant, 1);
        return quant;
    }

    private static int occurrences(String text, String token) {
        int count = 0;
        int from = 0;
        while ((from = text.indexOf(token, from)) >= 0) {
            count++;
            from += token.length();
        }
        return count;
    }

    private static boolean close(double a, double b) {
        return Math.abs(a - b) < 1.0e-12;
    }

    private static void assertRejected(
            ThrowingAction action, String message
    ) throws Exception {
        boolean rejected = false;
        try {
            action.run();
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected : message;
    }

    @FunctionalInterface
    private interface ThrowingAction {
        void run() throws Exception;
    }

    private static final class CountingFlipBackend
            implements StcEmbeddingBackend {
        private int embedCalls;

        @Override
        public StcEmbedResult embedPayloadWithTrace(
                byte[] coverBits,
                double[] costs,
                byte[] payloadBytes
        ) {
            embedCalls++;
            byte[] stego = Arrays.copyOf(
                    coverBits, coverBits.length);
            for (int i = 0; i < stego.length; i++) {
                stego[i] ^= 1;
            }
            return new StcEmbedResult(
                    stego,
                    new StcEmbeddingTrace(
                            coverBits.length,
                            payloadBytes.length * 8,
                            coverBits.length, costs[0]));
        }

        @Override
        public byte[] extractPayload(
                byte[] stegoBits, int payloadByteLength
        ) {
            return new byte[payloadByteLength];
        }
    }
}
