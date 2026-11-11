package thesis.rap.app;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;

/** Verifies production rejects a mislabeled non-PNG before JNI work. */
public final class ProductionInputGuardSelfTest {
    private ProductionInputGuardSelfTest() {}

    public static void main(String[] args) throws Exception {
        Path root = Files.createTempDirectory("rap-production-guard-");
        Path message = root.resolve("actual.msg");
        Files.writeString(
                message, "actual-test-data", StandardCharsets.UTF_8);

        Path falsePng = root.resolve("not-really.png");
        Files.writeString(
                falsePng, "not a PNG", StandardCharsets.UTF_8);
        Path falsePngOutput = root.resolve("false-png.jpg");
        expectInputRejection(() -> embed(
                falsePng, message, falsePngOutput));
        requireNoOutput(falsePngOutput);

        System.out.println(
                "ProductionInputGuardSelfTest: ALL TESTS PASSED");
    }

    private static void embed(
            Path png, Path message, Path output
    ) throws Exception {
        RapUniward.main(new String[]{
                "embed",
                "--png", png.toString(),
                "--msgFile", message.toString(),
                "--out", output.toString(),
                "--encKey", "production-guard-enc-key",
                "--xorKey", "production-guard-xor-key",
                "--stcKey", "production-guard-stc-key",
                "--auditEnabled", "false",
                "--auditProfile", "THESIS"
        });
    }

    private static void expectInputRejection(
            ThrowingRunnable action
    ) throws Exception {
        boolean rejected = false;
        try {
            action.run();
        } catch (RuntimeException failure) {
            Throwable cursor = failure;
            while (cursor != null) {
                if (cursor instanceof IllegalArgumentException) {
                    rejected = true;
                    break;
                }
                cursor = cursor.getCause();
            }
            if (!rejected) {
                throw new AssertionError(
                        "Failure was not an input rejection", failure);
            }
        }
        if (!rejected) {
            throw new AssertionError(
                    "Invalid production input was not rejected");
        }
    }

    private static void requireNoOutput(Path output)
            throws Exception {
        requirePathAbsent(output);
        for (SamplingGeometry geometry
                : SamplingGeometryMode.ALL_GEOMATRY.geometries()) {
            requirePathAbsent(RapUniward.outputForGeometry(
                    output, SamplingGeometryMode.ALL_GEOMATRY,
                    geometry));
        }
    }

    private static void requirePathAbsent(Path output) {
        if (Files.exists(output)
                || Files.exists(VerifiedOutputManifest.sidecarFor(output))) {
            throw new AssertionError(
                    "Rejected input created output evidence: " + output);
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
