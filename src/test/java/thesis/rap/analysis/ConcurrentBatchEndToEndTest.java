package thesis.rap.analysis;

import thesis.rap.app.RapUniward;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/** Native/JPEG smoke test for concurrent pair scheduling and audit-off mode. */
public final class ConcurrentBatchEndToEndTest {
    private ConcurrentBatchEndToEndTest() {}

    public static void main(String[] args) throws Exception {
        int pairs = Integer.getInteger("rap.test.concurrentPairs", 8);
        if (pairs < 2 || pairs > 64) {
            throw new IllegalArgumentException(
                    "rap.test.concurrentPairs must be from 2 through 64");
        }
        Path root = Files.createTempDirectory("rap-concurrent-batch-");
        Path input = Files.createDirectory(root.resolve("input"));
        Path messages = Files.createDirectory(root.resolve("msg"));
        Path output = Files.createDirectory(root.resolve("output"));
        Path manifest = root.resolve("batch.csv");
        Path message = messages.resolve("001.msg");
        byte[] expected = new byte[16];
        new Random(20260817L).nextBytes(expected);
        Files.write(message, expected);

        BufferedImage cover = new BufferedImage(
                128, 128, BufferedImage.TYPE_INT_RGB);
        Random pixels = new Random(2026081701L);
        for (int y = 0; y < cover.getHeight(); y++) {
            for (int x = 0; x < cover.getWidth(); x++) {
                int rgb = (pixels.nextInt(256) << 16)
                        | (pixels.nextInt(256) << 8)
                        | pixels.nextInt(256);
                cover.setRGB(x, y, rgb);
            }
        }
        List<String> imageNames = new ArrayList<>();
        imageNames.add("000001.png");
        imageNames.add("0001.png");
        for (int i = 2; i < pairs; i++) {
            imageNames.add(String.format("image_%03d.png", i));
        }
        for (String name : imageNames) {
            ImageIO.write(cover, "png", input.resolve(name).toFile());
        }

        Set<Path> auditsBefore = auditDirectories();
        boolean virtualMode = Runtime.version().feature() >= 22;
        String threadMode = virtualMode ? "virtual" : "platform";
        int workers = Integer.getInteger(
                "rap.test.workers", Math.min(8, pairs));
        int carriers = Integer.getInteger(
                "rap.test.carriers",
                Math.min(4, Math.max(
                        1, Runtime.getRuntime().availableProcessors())));
        int cpuSlots = Integer.getInteger(
                "rap.test.cpuSlots",
                virtualMode ? carriers : Math.min(workers, carriers));
        if (!virtualMode && (workers < 1 || workers > pairs)) {
            throw new IllegalArgumentException(
                    "rap.test.workers must be from 1 through the pair count");
        }
        if (carriers < 1) {
            throw new IllegalArgumentException(
                    "rap.test.carriers must be positive");
        }
        if (cpuSlots < 1
                || (!virtualMode && cpuSlots > workers)) {
            throw new IllegalArgumentException(
                    "rap.test.cpuSlots does not match the selected thread mode");
        }
        String workerOption = virtualMode
                ? "auto" : Integer.toString(workers);
        String[] keys = {
                "--encKey", "concurrent-enc-key-2026",
                "--xorKey", "concurrent-xor-key-2026",
                "--stcKey", "concurrent-stc-key-2026"
        };
        RapUniward.main(concat(new String[]{
                "batch-embed",
                "--pngDir", input.toString(),
                "--msgDir", messages.toString(),
                "--outDir", output.toString(),
                "--recursive", "false",
                "--resume", "false",
                "--continueOnError", "true",
                "--startIndex", "1",
                "--manifest", manifest.toString(),
                "--messageExtension", ".msg",
                "--onlyMessageName", "001.msg",
                "--minQf", "75",
                "--maxQf", "100",
                "--selectionPolicy", "capacity",
                "--auditEnabled", "false",
                "--auditProfile", "THESIS",
                "--deleteJpegsAfterEvaluation", "false",
                "--workers", workerOption,
                "--carrierParallelism", Integer.toString(carriers),
                "--cpuSlots", Integer.toString(cpuSlots),
                "--threadMode", threadMode
        }, keys));

        List<Map<String, String>> rows = CsvUtil.read(manifest);
        int geometryCount = SamplingGeometryMode.ALL_GEOMATRY
                .geometries().size();
        int expectedTrials = Math.multiplyExact(pairs, geometryCount);
        if (rows.size() != expectedTrials) {
            throw new AssertionError(
                    "Expected " + expectedTrials
                            + " image-message-geometry manifest rows, got "
                            + rows.size());
        }
        for (int i = 0; i < rows.size(); i++) {
            Map<String, String> row = rows.get(i);
            int observedIndex = Integer.parseInt(row.get("pair_index"));
            if (observedIndex != i + 1) {
                throw new AssertionError(
                        "Manifest order is not deterministic at row " + i);
            }
            if (!"COMPLETED".equals(row.get("status"))) {
                throw new AssertionError(
                        "Concurrent pair did not complete: " + row);
            }
            SamplingGeometry expectedGeometry =
                    SamplingGeometryMode.ALL_GEOMATRY.geometries()
                            .get(i % geometryCount);
            String expectedImage = imageNames.get(i / geometryCount);
            if (!"ALL_GEOMETRY".equals(row.get("geometry_mode"))
                    || !expectedGeometry.label().equals(
                    row.get("sampling_geometry"))
                    || !Path.of(row.get("image")).getFileName()
                    .toString().equals(expectedImage)
                    || !String.valueOf(expectedTrials).equals(
                    row.get("total_pairs"))) {
                throw new AssertionError(
                        "Manifest geometry expansion/order mismatch: " + row);
            }
            Path stego = Path.of(row.get("output"));
            if (!Files.isRegularFile(stego) || Files.size(stego) == 0) {
                throw new AssertionError(
                        "Manifest output is missing: " + stego);
            }
        }
        if (!auditsBefore.equals(auditDirectories())) {
            throw new AssertionError(
                    "--auditEnabled=false created an audit directory");
        }

        for (int index : new int[]{0, rows.size() - 1}) {
            Map<String, String> row = rows.get(index);
            Path recovered = root.resolve("recovered_" + index + ".msg");
            RapUniward.main(concat(new String[]{
                    "extract",
                    "--png", row.get("image"),
                    "--stego", row.get("output"),
                    "--outFile", recovered.toString(),
                    "--minQf", "75",
                    "--maxQf", "100"
            }, keys));
            if (!Arrays.equals(expected, Files.readAllBytes(recovered))) {
                throw new AssertionError(
                        "Concurrent batch extraction mismatch at index "
                                + index);
            }
        }
        if (imageNames.get(0).equals(imageNames.get(1))) {
            throw new AssertionError("leading-zero test IDs collapsed");
        }
        System.out.printf(
                "ConcurrentBatchEndToEndTest: PASSED images=%d "
                        + "geometryTrials=%d taskPolicy=%s "
                        + "cpuSlots=%d threadMode=%s auditEnabled=false "
                        + "orderedManifest=true%n",
                pairs, expectedTrials,
                virtualMode ? "VIRTUAL_AUTO_IN_FLIGHT_WINDOW_"
                        + (2 * Math.max(cpuSlots, carriers))
                        : "PLATFORM_BOUNDED_" + workers,
                cpuSlots, threadMode);
    }

    private static Set<Path> auditDirectories() throws Exception {
        Path auditRoot = Path.of("results", "audit");
        Set<Path> directories = new HashSet<>();
        if (!Files.isDirectory(auditRoot)) return directories;
        try (var stream = Files.list(auditRoot)) {
            stream.filter(Files::isDirectory)
                    .map(Path::toAbsolutePath)
                    .forEach(directories::add);
        }
        return directories;
    }

    private static String[] concat(String[] first, String[] second) {
        String[] out = Arrays.copyOf(
                first, first.length + second.length);
        System.arraycopy(
                second, 0, out, first.length, second.length);
        return out;
    }
}
