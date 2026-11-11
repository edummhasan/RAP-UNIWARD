package thesis.rap.app;

import thesis.rap.analysis.CsvUtil;
import thesis.rap.image.SamplingGeometryMode;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.TimeUnit;

/** Real no-capacity outcomes, fatal inputs, null cleanup, and forced shutdown failures. */
public final class ForensicHardeningSelfTest {
    public static void main(String[] args) throws Exception {
        assert SamplingGeometryMode.parse("ALL_GEOMATRY").token().equals("ALL_GEOMETRY");
        assert SamplingGeometryMode.parse("444GEOMATRY").token().equals("444GEOMETRY");
        RapUniward.deleteOptionalTemporary(null);
        Path temporary = Files.createTempFile("rap-delete-test-", ".tmp");
        RapUniward.deleteOptionalTemporary(temporary);
        assert !Files.exists(temporary);
        try {
            VerifiedOutputManifest.sidecarFor(null);
            throw new AssertionError("Required output path accepted null");
        } catch (NullPointerException expected) {
            assert expected.getMessage().equals("stegoOutputPath");
        }
        Error original = new OutOfMemoryError("test-only synthetic primary error");
        BatchExecutorSupport.closePreservingFailure(new NeverTerminates(), false, original);
        assert original.getSuppressed().length == 1;
        assert original.getSuppressed()[0].getMessage().contains("did not terminate");
        FailureDiagnostics.Info info = FailureDiagnostics.info(original);
        assert info.stackHash().length() == 64;
        assert info.className().equals("java.lang.OutOfMemoryError");
        assert FailureDiagnostics.info(null).className().isEmpty();
        for (boolean infeasible : new boolean[]{false, true}) {
            RuntimeException originalStcFailure = infeasible
                    ? new thesis.rap.stc.NoFiniteStcSolutionException("synthetic no-solution")
                    : new IllegalStateException("synthetic native invariant failure");
            StcEmbeddingBackend backend = new StcEmbeddingBackend() {
                public StcEmbedResult embedPayloadWithTrace(byte[] bits, double[] costs, byte[] payload) {
                    throw originalStcFailure;
                }
                public byte[] extractPayload(byte[] bits, int length) {
                    throw new AssertionError("Must not extract after failed embedding");
                }
            };
            thesis.rap.alg.Candidate candidate = new thesis.rap.alg.Candidate(
                    1, 0, 0, 0, 0, 5, 40, 2, 1.0, 2.0, 0.0, false,
                    0.0, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0e-4);
            try {
                ThreeStateEmbeddingEngine.embed(backend, List.of(candidate), new byte[1], new byte[1],
                        List.of(candidate), new byte[1], new byte[1], 75,
                        ThreeStateEmbeddingEngine.Profile.UNRESTRICTED);
                throw new AssertionError("Failed STC was accepted");
            } catch (ThreeStateEmbeddingEngine.NoFeasibleBalanceException expected) {
                assert infeasible && expected.getCause() == originalStcFailure;
            } catch (IllegalStateException expected) {
                assert !infeasible && expected == originalStcFailure;
            }
        }
        noOutputIsScientificOutcome();
        System.out.println("ForensicHardeningSelfTest: ALL TESTS PASSED");
    }

    private static void noOutputIsScientificOutcome() throws Exception {
        Path dir=Files.createTempDirectory("rap-no-output-test-");
        Path input=Files.createDirectory(dir.resolve("input"));
        Path messages=Files.createDirectory(dir.resolve("msg"));
        Path output=Files.createDirectory(dir.resolve("output"));
        Path manifest=dir.resolve("manifest.csv");
        ImageIO.write(new BufferedImage(8,8,BufferedImage.TYPE_INT_RGB),"png",input.resolve("000001.png").toFile());
        Files.write(messages.resolve("001.msg"), new byte[]{1,2,3});
        String[] options={"batch-embed", "--pngDir", input.toString(), "--msgDir", messages.toString(),
                "--outDir",output.toString(), "--manifest",manifest.toString(),
                "--geometryMode","ALL_GEOMETRY", "--continueOnError","true",
                "--auditEnabled","true", "--auditMode","RESULT_ONLY",
                "--deleteJpegsAfterEvaluation","false", "--workers","1", "--cpuSlots","1",
                "--encKey","test-enc", "--xorKey","test-xor", "--stcKey","test-stc"};
        RapUniward.main(options); // must return normally with zero verified outputs
        List<Map<String,String>> rows=CsvUtil.read(manifest);
        assert rows.size()==7;
        for(Map<String,String> row:rows) assert row.get("status").equals("NO_OUTPUT_CHECK_RESULT_LEDGER");

        // NO_OUTPUT is a scientific terminal result, not an infrastructure error.
        // Fail-fast mode must therefore not retry/abort merely because capacity or
        // the verified kernel rejects all QFs.
        Path failFastManifest=dir.resolve("manifest_fail_fast.csv");
        String[] failFastOptions=options.clone();
        for (int i=0;i<failFastOptions.length-1;i++) {
            if (failFastOptions[i].equals("--manifest")) failFastOptions[i+1]=failFastManifest.toString();
            if (failFastOptions[i].equals("--continueOnError")) failFastOptions[i+1]="false";
        }
        RapUniward.main(failFastOptions);
        List<Map<String,String>> failFastRows=CsvUtil.read(failFastManifest);
        assert failFastRows.size()==7;
        for(Map<String,String> row:failFastRows) assert row.get("status").equals("NO_OUTPUT_CHECK_RESULT_LEDGER");

        // Invalid image is an infrastructure/input failure and must escape, even with continuation enabled.
        Files.write(input.resolve("000001.png"), new byte[]{1,2,3});
        Files.delete(manifest);
        try {
            RapUniward.main(options);
            throw new AssertionError("Invalid PNG was treated as a scientific outcome");
        } catch (Exception expected) {
            assert expected.getMessage() != null;
        }
    }

    private static final class NeverTerminates extends AbstractExecutorService {
        public void shutdown() {}
        public List<Runnable> shutdownNow() { return List.of(); }
        public boolean isShutdown() { return true; }
        public boolean isTerminated() { return false; }
        public boolean awaitTermination(long timeout, TimeUnit unit) { return false; }
        public void execute(Runnable command) { throw new UnsupportedOperationException(); }
    }
}
