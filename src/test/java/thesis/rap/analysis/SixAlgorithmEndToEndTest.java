package thesis.rap.analysis;

import thesis.rap.app.RapUniward;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;
import thesis.rap.image.StandardJpegQuantization;
import thesis.rap.jni.JpegCore;
import thesis.rap.jni.JpegImage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/** Real native-JPEG/STC round-trip test for the complete six-algorithm pipeline. */
public final class SixAlgorithmEndToEndTest {
    private SixAlgorithmEndToEndTest() {}

    public static void main(String[] args) throws Exception {
        String auditProfile = System.getProperty(
                "rap.test.auditProfile", "THESIS");
        String auditMode = System.getProperty(
                "rap.test.auditMode", "QF_FORENSIC");
        Path dir = Files.createTempDirectory("rap-six-algorithms-");
        Path png = dir.resolve("cover.png");
        Path msg = dir.resolve("message.msg");
        Path requestedStego = dir.resolve("stego.jpg");
        BufferedImage image = new BufferedImage(128, 128, BufferedImage.TYPE_INT_RGB);
        Random random = new Random(20260714L);
        for (int y = 0; y < image.getHeight(); y++) for (int x = 0; x < image.getWidth(); x++) {
            int r = random.nextInt(256), g = random.nextInt(256), b = random.nextInt(256);
            image.setRGB(x, y, (r << 16) | (g << 8) | b);
        }
        ImageIO.write(image, "png", png.toFile());
        byte[] expectedMessage = new byte[64];
        random.nextBytes(expectedMessage);
        expectedMessage[0] = 0;
        expectedMessage[1] = (byte) 0xff;
        Files.write(msg, expectedMessage);
        Set<Path> auditsBefore = auditDirectories();
        int resultRowsBefore = resultAuditRows().size();
        String[] keys = {"--encKey", "enc-test-key-2026", "--xorKey", "xor-test-key-2026", "--stcKey", "stc-test-key-2026"};
        RapUniward.main(concat(new String[]{"embed", "--png", png.toString(), "--msgFile", msg.toString(),
                "--out", requestedStego.toString(), "--minQf", "75", "--maxQf", "100",
                "--geometryMode", "ALL_GEOMETRY",
                "--auditMode", auditMode,
                "--auditProfile", auditProfile}, keys));
        Map<SamplingGeometry, GeometryRun> runs =
                new EnumMap<>(SamplingGeometry.class);
        for (SamplingGeometry geometry
                : SamplingGeometryMode.ALL_GEOMATRY.geometries()) {
            Path stego = geometryOutput(requestedStego, geometry);
            if (!Files.isRegularFile(stego) || Files.size(stego) == 0) {
                throw new AssertionError(
                        "No verified stego output for " + geometry.label());
            }
            JpegImage written = JpegCore.readCoefficients(
                    stego.toString());
            SamplingGeometry detected = SamplingGeometry.detect(written);
            if (detected != geometry) {
                throw new AssertionError(
                        "Written geometry mismatch: expected "
                                + geometry.label() + ", got "
                                + detected.label());
            }
            int selectedQf = quantizationQf(written, 75, 100);
            if (selectedQf < 0) {
                throw new AssertionError(
                        "Written quantization tables for "
                                + geometry.label()
                                + " do not match any tested standard QF");
            }
            Path recovered = dir.resolve(
                    "recovered_" + geometry.outputTag() + ".msg");
            RapUniward.main(concat(new String[]{
                    "extract", "--png", png.toString(),
                    "--stego", stego.toString(),
                    "--outFile", recovered.toString(),
                    "--minQf", "75", "--maxQf", "100",
                    "--geometryMode", geometry.modeToken()
            }, keys));
            if (!Arrays.equals(
                    expectedMessage, Files.readAllBytes(recovered))) {
                throw new AssertionError(
                        "Binary extraction did not reproduce the exact "
                                + "message bytes for " + geometry.label());
            }
            runs.put(
                    geometry,
                    new GeometryRun(geometry, stego, selectedQf));
        }

        Path authenticationProbe = runs.get(
                SamplingGeometry.JPEG_420).stego();

        assertWrongKeysRejected(
                png, authenticationProbe, dir.resolve("wrong-enc.msg"),
                new String[]{
                        "--encKey", "wrong-enc-test-key-2026",
                        "--xorKey", "xor-test-key-2026",
                        "--stcKey", "stc-test-key-2026"
                });
        assertWrongKeysRejected(
                png, authenticationProbe, dir.resolve("wrong-xor.msg"),
                new String[]{
                        "--encKey", "enc-test-key-2026",
                        "--xorKey", "wrong-xor-test-key-2026",
                        "--stcKey", "stc-test-key-2026"
                });
        assertWrongKeysRejected(
                png, authenticationProbe, dir.resolve("wrong-stc.msg"),
                new String[]{
                        "--encKey", "enc-test-key-2026",
                        "--xorKey", "xor-test-key-2026",
                        "--stcKey", "wrong-stc-test-key-2026"
                });

        if ("RESULT_ONLY".equalsIgnoreCase(auditMode)) {
            verifyResultOnlyAudit(
                    auditsBefore, resultRowsBefore, png, msg, runs);
        } else if ("QF_FORENSIC".equalsIgnoreCase(auditMode)) {
            verifyAuditBundles(auditsBefore, auditProfile);
        } else {
            throw new AssertionError(
                    "Unsupported test audit mode: " + auditMode);
        }
        int libjpegVersion = JpegCore.libjpegVersion();
        if (libjpegVersion <= 0) {
            throw new AssertionError("Invalid native libjpeg version");
        }
        System.out.println(
                "Native environment: java="
                        + System.getProperty("java.version")
                        + ", os=" + System.getProperty("os.name")
                        + ", arch=" + System.getProperty("os.arch")
                        + ", JPEG_LIB_VERSION=" + libjpegVersion);
        System.out.println(
                "SixAlgorithmEndToEndTest: PASSED, selectedQFs="
                        + runs.entrySet().stream()
                        .map(entry -> entry.getKey().label() + "="
                                + entry.getValue().selectedQf())
                        .toList()
                        + ", exactBinary=true, wrongEnc=true,"
                        + " wrongXor=true, wrongStc=true,"
                        + " auditMode=" + auditMode);
    }

    private static Set<Path> auditDirectories() throws Exception {
        Path root = Path.of("results", "audit");
        Set<Path> directories = new HashSet<>();
        if (!Files.isDirectory(root)) return directories;
        try (var stream = Files.list(root)) {
            stream.filter(Files::isDirectory)
                    .map(Path::toAbsolutePath)
                    .forEach(directories::add);
        }
        return directories;
    }

    private static java.util.List<java.util.Map<String, String>>
    resultAuditRows() throws Exception {
        Path ledger = Path.of("results", "audit_final_qf.csv");
        return Files.isRegularFile(ledger)
                ? CsvUtil.read(ledger) : java.util.List.of();
    }

    private static void verifyResultOnlyAudit(
            Set<Path> auditsBefore,
            int rowsBefore,
            Path png,
            Path message,
            Map<SamplingGeometry, GeometryRun> runs
    ) throws Exception {
        if (!auditsBefore.equals(auditDirectories())) {
            throw new AssertionError(
                    "RESULT_ONLY created a per-run forensic audit directory");
        }
        var rows = resultAuditRows();
        if (rows.size() != rowsBefore + runs.size()) {
            throw new AssertionError(
                    "RESULT_ONLY did not append exactly one final row "
                            + "per geometry");
        }
        var appended = rows.subList(rowsBefore, rows.size());
        for (GeometryRun run : runs.values()) {
            Map<String, String> row = appended.stream()
                    .filter(candidate -> run.stego().getFileName()
                            .toString().equals(
                                    candidate.get("output_name")))
                    .findFirst()
                    .orElseThrow(() -> new AssertionError(
                            "Missing RESULT_ONLY row for "
                                    + run.geometry().label()));
            if (!"RESULT_ONLY".equals(row.get("audit_mode"))
                    || !"ALL_GEOMETRY".equals(
                    row.get("geometry_mode"))
                    || !run.geometry().label().equals(
                    row.get("sampling_geometry"))
                    || !"VERIFIED_SUCCESS".equals(row.get("status"))
                    || !"true".equals(row.get("embedding_success"))
                    || !"true".equals(row.get("extraction_success"))
                    || !"true".equals(row.get("aes_gcm_verified"))
                    || !"true".equals(
                    row.get("exact_message_verified"))
                    || !png.getFileName().toString().equals(
                    row.get("image_name"))
                    || !message.getFileName().toString().equals(
                    row.get("message_name"))
                    || !String.valueOf(run.selectedQf()).equals(
                    row.get("terminal_qf"))
                    || !"SELECTED_VERIFIED".equals(
                    row.get("terminal_qf_role"))
                    || !"VERIFIED".equals(
                    row.get("terminal_qf_status"))
                    || !"true".equals(
                    row.get("final_qf_evaluation_available"))
                    || row.get("reference_stego_psnr").isBlank()
                    || row.get("reference_stego_ssim").isBlank()
                    || row.get("eval_modified_coefficients").isBlank()
                    || row.get("cb_modified_coefficients").isBlank()
                    || row.get("cr_modified_coefficients").isBlank()) {
                throw new AssertionError(
                        "RESULT_ONLY final row is incomplete: " + row);
            }
            if (!"false".equals(row.get("qf_history_retained"))
                    || !"false".equals(
                    row.get("balance_history_retained"))
                    || !"false".equals(
                    row.get("per_zz_rows_retained"))) {
                throw new AssertionError(
                        "RESULT_ONLY retained a forbidden intermediate "
                                + "history");
            }
        }
    }

    private static void assertWrongKeysRejected(
            Path png,
            Path stego,
            Path output,
            String[] wrongKeys
    ) throws Exception {
        boolean rejected = false;
        try {
            RapUniward.main(concat(new String[]{
                    "extract", "--png", png.toString(),
                    "--stego", stego.toString(),
                    "--outFile", output.toString(),
                    "--minQf", "75", "--maxQf", "100"
            }, wrongKeys));
        } catch (IllegalStateException expected) {
            rejected = true;
        }
        if (!rejected || Files.exists(output)) {
            throw new AssertionError(
                    "Wrong-key extraction must fail without releasing plaintext: "
                            + output.getFileName());
        }
    }

    private static void verifyAuditBundles(
            Set<Path> before,
            String auditProfile
    )
            throws Exception {
        Set<Path> after = auditDirectories();
        after.removeAll(before);
        int expectedCount = SamplingGeometryMode.ALL_GEOMATRY
                .geometries().size();
        if (after.size() != expectedCount) {
            throw new AssertionError(
                    "Expected one new atomic audit bundle per geometry, got "
                            + after.size());
        }
        Set<String> observedGeometries = new HashSet<>();
        for (Path audit : after) {
            String[] required = {
                    "run.json",
                    "qf_trials.csv",
                    "balance_attempts.csv",
                    "candidates.csv",
                    "jpeg_statistics.csv",
                    "verification.json"
            };
            for (String name : required) {
                Path file = audit.resolve(name);
                if (!Files.isRegularFile(file)
                        || Files.size(file) == 0) {
                    throw new AssertionError(
                            "Missing or empty mandatory audit file: "
                                    + name + " in " + audit);
                }
            }
            String runJson = Files.readString(
                    audit.resolve("run.json"),
                    StandardCharsets.UTF_8);
            for (SamplingGeometry geometry
                    : SamplingGeometryMode.ALL_GEOMATRY.geometries()) {
                if (runJson.contains(
                        "\"sampling_geometry\": \""
                                + geometry.label() + "\"")) {
                    observedGeometries.add(geometry.label());
                }
            }
            String verification = Files.readString(
                    audit.resolve("verification.json"),
                    StandardCharsets.UTF_8);
            if (!verification.contains("\"overall_verified\": true")) {
                throw new AssertionError(
                        "verification.json does not certify run " + audit);
            }

            Path candidates = audit.resolve("candidates.csv");
            var lines = Files.readAllLines(
                    candidates, StandardCharsets.UTF_8);
            if (lines.size() < 2) {
                throw new AssertionError(
                        "Candidate summary must contain a TOTAL row");
            }
            String[] header = lines.get(0).split(",", -1);
            int component = indexOf(header, "component");
            int modifiedCount = indexOf(header, "modified_count");
            int universeCount = indexOf(header, "universe_count");
            int candidateHash = indexOf(
                    header, "candidate_stream_sha256");
            String[] total = lines.get(1).split(",", -1);
            if (!"ALL".equals(total[component])) {
                throw new AssertionError(
                        "First candidate-summary row must be TOTAL");
            }
            if (Long.parseLong(total[universeCount]) <= 0L
                    || Long.parseLong(total[modifiedCount]) <= 0L) {
                throw new AssertionError(
                        "Candidate summary must record the universe "
                                + "and changes");
            }
            if (!total[candidateHash].matches("[0-9a-f]{64}")) {
                throw new AssertionError(
                        "Candidate stream hash is missing or malformed");
            }
            Path changes = audit.resolve("candidate_changes.csv");
            Path full = audit.resolve("candidates_full.csv.gz");
            if ("THESIS".equalsIgnoreCase(auditProfile)
                    && (Files.exists(changes) || Files.exists(full))) {
                throw new AssertionError(
                        "Default THESIS profile emitted expanded "
                                + "candidate logs");
            }
            if (("CHANGES".equalsIgnoreCase(auditProfile)
                    || "FULL_AUDIT".equalsIgnoreCase(auditProfile))
                    && (!Files.isRegularFile(changes)
                    || Files.size(changes) == 0L)) {
                throw new AssertionError(
                        "Changed-row profile did not emit "
                                + "candidate_changes.csv");
            }
            if ("FULL_AUDIT".equalsIgnoreCase(auditProfile)
                    && (!Files.isRegularFile(full)
                    || Files.size(full) == 0L)) {
                throw new AssertionError(
                        "FULL_AUDIT did not emit candidates_full.csv.gz");
            }
        }
        Set<String> expectedGeometries = new HashSet<>();
        for (SamplingGeometry geometry
                : SamplingGeometryMode.ALL_GEOMATRY.geometries()) {
            expectedGeometries.add(geometry.label());
        }
        if (!observedGeometries.equals(expectedGeometries)) {
            throw new AssertionError(
                    "Audit bundles do not cover all geometries: "
                            + observedGeometries);
        }
    }

    private static int indexOf(String[] header, String name) {
        for (int i = 0; i < header.length; i++) {
            if (header[i].equals(name)) return i;
        }
        throw new AssertionError(
                "Missing candidate-audit field: " + name);
    }

    private static int quantizationQf(JpegImage image, int minQf, int maxQf) {
        for (int qf = minQf; qf <= maxQf; qf++) {
            if (java.util.Arrays.equals(
                    image.component(0).quantTable,
                    StandardJpegQuantization.luminance(qf))
                    && java.util.Arrays.equals(
                    image.component(1).quantTable,
                    StandardJpegQuantization.chrominance(qf))
                    && java.util.Arrays.equals(
                    image.component(2).quantTable,
                    StandardJpegQuantization.chrominance(qf))) {
                return qf;
            }
        }
        return -1;
    }

    private static Path geometryOutput(
            Path requested,
            SamplingGeometry geometry
    ) {
        String name = requested.getFileName().toString();
        String lower = name.toLowerCase(java.util.Locale.ROOT);
        int extensionStart = lower.endsWith(".jpeg")
                ? name.length() - 5
                : lower.endsWith(".jpg")
                ? name.length() - 4 : name.length();
        return requested.resolveSibling(
                name.substring(0, extensionStart)
                        + "_" + geometry.outputTag()
                        + name.substring(extensionStart));
    }

    private static String[] concat(String[] a, String[] b) {
        String[] out = new String[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private record GeometryRun(
            SamplingGeometry geometry,
            Path stego,
            int selectedQf
    ) {}
}
