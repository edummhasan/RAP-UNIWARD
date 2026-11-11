package thesis.rap.runtime;

import java.nio.file.Path;

/** Result redirection must affect all paths without moving compiled artifacts. */
public final class RuntimePathsSelfTest {
    private RuntimePathsSelfTest() {}
    public static void main(String[] args) {
        String previous = System.getProperty("rap.report.dir");
        try {
            System.setProperty("rap.report.dir", "custom-results/experiment");
            assert RuntimePaths.results("audit", "trial.json").equals(
                    Path.of("custom-results", "experiment", "audit", "trial.json"));
            System.clearProperty("rap.report.dir");
            assert RuntimePaths.results("evaluation_runs.csv").equals(
                    Path.of(System.getenv().getOrDefault("RAP_REPORT_DIR", "results"), "evaluation_runs.csv"));
        } finally {
            if (previous == null) System.clearProperty("rap.report.dir");
            else System.setProperty("rap.report.dir", previous);
        }
        System.out.println("RuntimePathsSelfTest: PASS");
    }
}
