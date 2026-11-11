package thesis.rap.app;

import thesis.rap.alg.CandidateSet;
import thesis.rap.image.StandardJpegQuantization;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;
import thesis.rap.payload.PayloadPackage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public final class RuntimeContractSelfTest {
    private RuntimeContractSelfTest() {}

    public static void main(String[] args) throws Exception {
        protocolPreflightUsesTheRealPreparedLength();
        batchStopsBeforeAnyPopulationOnOversizedMessage();
        quantTablePrecheckIsolatesExactlyOneQf();
        preparedStateTemplateIsImmutableInMemoryBytes();
        imageShardingKeepsEachImageOnOneShard();
        System.out.println("RuntimeContractSelfTest: ALL TESTS PASSED");
    }

    /** 15 messages x 7 geometries, the thesis layout, over 1..8 shards. */
    private static void imageShardingKeepsEachImageOnOneShard() {
        int images = 40;
        long pairsPerImage = 15L * 7L;
        for (int shards = 1; shards <= 8; shards++) {
            int[] imageOwner = new int[images];
            Arrays.fill(imageOwner, -1);
            long[] perShard = new long[shards];
            for (long pair = 1; pair <= images * pairsPerImage; pair++) {
                int owners = 0;
                int owner = -1;
                for (int r = 0; r < shards; r++) {
                    if (RapUniward.imageBelongsToShard(pair, pairsPerImage, shards, r)) {
                        owners++;
                        owner = r;
                    }
                }
                require(owners == 1, "pair " + pair + " has " + owners
                        + " owners at " + shards + " shards");
                int image = (int) ((pair - 1) / pairsPerImage);
                require(imageOwner[image] == -1 || imageOwner[image] == owner,
                        "image " + image + " split across shards");
                imageOwner[image] = owner;
                perShard[owner]++;
            }
            long min = Arrays.stream(perShard).min().orElseThrow();
            long max = Arrays.stream(perShard).max().orElseThrow();
            require(max - min <= pairsPerImage,
                    "image sharding unbalanced by more than one image: "
                            + Arrays.toString(perShard));
        }
        // PAIR sharding (unchanged default) scatters one (image, geometry)
        // over every shard when 7 and the shard count are coprime.
        java.util.Set<Integer> shardsSeen = new java.util.HashSet<>();
        for (int message = 0; message < 15; message++) {
            long pair = 1 + message * 7L;           // image 0, geometry 4:4:4
            for (int r = 0; r < 4; r++) {
                if (RapUniward.pairBelongsToShard(pair, 4, r)) shardsSeen.add(r);
            }
        }
        require(shardsSeen.size() == 4,
                "expected PAIR sharding to scatter one (image, geometry): " + shardsSeen);
    }

    /** The bound is on the prepared payload, not the raw file length. */
    private static void protocolPreflightUsesTheRealPreparedLength() throws Exception {
        Path dir = Files.createTempDirectory("rap-standalone-preflight-");
        int maxRaw = PayloadPackage.MAX_CIPHERTEXT_LEN
                - PayloadPackage.PREPROCESSING_METADATA_LEN;
        Path atLimit = write(dir.resolve("at_limit.msg"), incompressible(maxRaw, 1));
        Path overLimit = write(dir.resolve("over_limit.msg"), incompressible(maxRaw + 1, 2));
        byte[] repetitive = new byte[200_000];
        Arrays.fill(repetitive, (byte) 'A');
        Path compressible = write(dir.resolve("compressible.msg"), repetitive);
        Path empty = write(dir.resolve("empty.msg"), new byte[0]);

        RapUniward.requireMessagesWithinProtocolLimit(
                List.of(atLimit, compressible, empty));
        try {
            RapUniward.requireMessagesWithinProtocolLimit(
                    List.of(atLimit, overLimit, compressible));
            throw new AssertionError("over-limit message passed the preflight");
        } catch (IllegalArgumentException expected) {
            String message = expected.getMessage();
            require(message.contains("over_limit.msg=" + (maxRaw + 1) + "B"),
                    "preflight must name the rejected file and size: " + message);
            require(!message.contains("at_limit.msg")
                            && !message.contains("compressible.msg"),
                    "preflight named a legal message: " + message);
        }
    }

    
    private static void batchStopsBeforeAnyPopulationOnOversizedMessage() throws Exception {
        Path root = Files.createTempDirectory("rap-standalone-batch-");
        Path input = Files.createDirectory(root.resolve("input"));
        Path messages = Files.createDirectory(root.resolve("msg"));
        Path output = root.resolve("output");
        Path manifest = root.resolve("manifest.csv");
        BufferedImage png = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB);
        ImageIO.write(png, "png", input.resolve("image.png").toFile());
        write(messages.resolve("big.msg"), incompressible(70_000, 3));
        try {
            RapUniward.main(new String[]{
                    "batch-embed",
                    "--pngDir", input.toString(),
                    "--msgDir", messages.toString(),
                    "--outDir", output.toString(),
                    "--manifest", manifest.toString(),
                    "--encKey", "regression-enc-key",
                    "--xorKey", "regression-xor-key",
                    "--stcKey", "regression-stc-key",
                    "--geometryMode", "ALL_GEOMETRY"});
            throw new AssertionError("batch accepted an over-limit message");
        } catch (IllegalArgumentException expected) {
            require(expected.getMessage().contains("cannot carry 1 message file"),
                    "wrong batch rejection: " + expected.getMessage());
        }
        require(!Files.exists(manifest), "a population was recorded before rejection");
    }

    /** Only the embedding QF's tables pass; all 26 table pairs are distinct. */
    private static void quantTablePrecheckIsolatesExactlyOneQf() {
        for (int embeddedQf = 75; embeddedQf <= 100; embeddedQf++) {
            JpegImage stego = canonicalTables(embeddedQf);
            for (int trialQf = 75; trialQf <= 100; trialQf++) {
                boolean match = RapUniward.quantTablesMatchCanonical(stego, trialQf);
                require(match == (trialQf == embeddedQf),
                        "embedded QF " + embeddedQf + " vs trial " + trialQf
                                + " matched=" + match);
            }
        }
        JpegImage tampered = canonicalTables(80);
        tampered.component(2).quantTable[63]++;
        require(!RapUniward.quantTablesMatchCanonical(tampered, 80),
                "a changed Cr table must fail the canonical pre-check");
    }

    private static void preparedStateTemplateIsImmutableInMemoryBytes() throws Exception {
        byte[] original = {(byte) 0xFF, (byte) 0xD8, 1, 2, 3, (byte) 0xFF, (byte) 0xD9};
        byte[] supplied = original.clone();
        JpegImage suppliedReference = canonicalTables(90);
        TrueParallelStateCache.State state = new TrueParallelStateCache.State(
                supplied, suppliedReference,
                new CandidateSet(List.of(), List.of()));
        supplied[2] = 99;
        state.canonicalTemplateBytes()[3] = 99;
        suppliedReference.component(1).coefficients[5] = 99;
        state.canonicalReference().component(1).coefficients[5] = 88;
        state.freshReferenceImage().component(1).quantTable[5] = 77;
        require(state.freshReferenceImage().component(1).coefficients[5] == 0,
                "prepared coefficient state escaped through a mutable reference");
        require(state.freshReferenceImage().component(1).quantTable[5]
                        == StandardJpegQuantization.chrominance(90)[5],
                "prepared quantization state escaped through a mutable reference");
        Path target = Files.createTempFile("rap-standalone-state-", ".jpg");
        try {
            state.materializeReference(target);
            require(Arrays.equals(original, Files.readAllBytes(target)),
                    "materialized reference is not the byte-identical template");
        } finally {
            Files.deleteIfExists(target);
        }
    }

    private static JpegImage canonicalTables(int qf) {
        int[] luma = StandardJpegQuantization.luminance(qf);
        int[] chroma = StandardJpegQuantization.chrominance(qf);
        JpegComponent[] components = new JpegComponent[3];
        for (int i = 0; i < 3; i++) {
            components[i] = new JpegComponent(
                    i, 1, 1, 1, 1, i == 0 ? 0 : 1,
                    new int[64], (i == 0 ? luma : chroma).clone());
        }
        return new JpegImage(8, 8, components);
    }

    private static byte[] incompressible(int length, long seed) {
        byte[] bytes = new byte[length];
        new Random(seed).nextBytes(bytes);
        return bytes;
    }

    private static Path write(Path path, byte[] bytes) throws Exception {
        Files.write(path, bytes);
        return path;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
