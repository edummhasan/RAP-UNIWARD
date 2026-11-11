package thesis.rap.app;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;

/** Regression tests for non-destructive cleanup and stale-artifact resume. */
public final class BatchArtifactSafetySelfTest {
    private BatchArtifactSafetySelfTest() {}

    public static void main(String[] args) throws Exception {
        Path directory = Files.createTempDirectory("rap-batch-artifacts-");
        try {
            failedStatusPreservesExistingArtifacts(directory);
            acceptedStatusDeletesOnlyItsAcceptedArtifacts(directory);
            orphanAndZeroLengthArtifactsAreQuarantined(directory);
            nonRegularArtifactsAreRefused(directory);
            System.out.println(
                    "BatchArtifactSafetySelfTest: ALL TESTS PASSED");
        } finally {
            try (var paths = Files.walk(directory)) {
                paths.sorted(Comparator.reverseOrder()).forEach(path -> {
                    try {
                        Files.deleteIfExists(path);
                    } catch (IOException failure) {
                        throw new RuntimeException(failure);
                    }
                });
            }
        }
    }

    private static void failedStatusPreservesExistingArtifacts(Path root)
            throws Exception {
        Path stego = root.resolve("preexisting.jpg");
        Path sidecar = VerifiedOutputManifest.sidecarFor(stego);
        Files.write(stego, new byte[]{1, 2, 3});
        Files.write(sidecar, new byte[]{4, 5, 6});
        boolean deleted = RapUniward.deletePairJpegsIfAccepted(
                stego, "FAILED_EXCEPTION");
        assert !deleted;
        assert Files.readAllBytes(stego).length == 3;
        assert Files.readAllBytes(sidecar).length == 3;
    }

    private static void acceptedStatusDeletesOnlyItsAcceptedArtifacts(
            Path root
    ) throws Exception {
        Path stego = root.resolve("accepted.jpg");
        Path sidecar = VerifiedOutputManifest.sidecarFor(stego);
        Files.write(stego, new byte[]{7});
        Files.write(sidecar, new byte[]{8});
        assert RapUniward.deletePairJpegsIfAccepted(stego, "COMPLETED");
        assert !Files.exists(stego);
        assert !Files.exists(sidecar);
    }

    private static void orphanAndZeroLengthArtifactsAreQuarantined(
            Path root
    ) throws Exception {
        Path orphan = root.resolve("orphan.jpg");
        Path orphanSidecar = VerifiedOutputManifest.sidecarFor(orphan);
        Files.write(orphanSidecar, new byte[]{9});
        assert RapUniward.outputArtifactsExist(orphan);
        RapUniward.quarantineExistingOutput(
                orphan, root, "test orphan sidecar");
        assert !Files.exists(orphanSidecar);

        Path zero = root.resolve("zero.jpg");
        Path zeroSidecar = VerifiedOutputManifest.sidecarFor(zero);
        Files.write(zero, new byte[0]);
        Files.write(zeroSidecar, new byte[]{10});
        RapUniward.quarantineExistingOutput(
                zero, root, "test zero-length output");
        assert !Files.exists(zero);
        assert !Files.exists(zeroSidecar);

        Path quarantine = root.resolve("quarantine");
        try (var entries = Files.list(quarantine)) {
            assert entries.count() == 3
                    : "one orphan and two zero-output artifacts must move";
        }
    }

    private static void nonRegularArtifactsAreRefused(Path root)
            throws Exception {
        Path directoryArtifact = root.resolve("directory.jpg");
        Files.createDirectory(directoryArtifact);
        boolean rejected = false;
        try {
            RapUniward.quarantineExistingOutput(
                    directoryArtifact, root, "test directory artifact");
        } catch (IOException expected) {
            rejected = true;
        }
        assert rejected;
        assert Files.isDirectory(directoryArtifact)
                : "a non-regular artifact must remain untouched";
    }
}
