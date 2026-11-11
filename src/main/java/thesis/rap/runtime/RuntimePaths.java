package thesis.rap.runtime;

import java.nio.file.Path;

/** Location of evaluation ledgers and audit artifacts, independent of build files. */
public final class RuntimePaths {
    private RuntimePaths() {}

    public static Path results(String... parts) {
        String configured = System.getProperty("rap.report.dir");
        if (configured == null || configured.isBlank()) {
            configured = System.getenv().getOrDefault("RAP_REPORT_DIR", "results");
        }
        if (configured.isBlank()) throw new IllegalArgumentException("RAP_REPORT_DIR must not be blank");
        Path path = Path.of(configured);
        for (String part : parts) path = path.resolve(part);
        return path;
    }
}
