package thesis.rap.analysis;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Verifies that aggregate ledgers cannot interleave or lose concurrent rows. */
public final class CsvConcurrentAppendSelfTest {
    private CsvConcurrentAppendSelfTest() {}

    public static void main(String[] args) throws Exception {
        Path directory = Files.createTempDirectory(
                "rap-csv-concurrency-");
        Path csv = directory.resolve("ledger.csv");
        Path lock = directory.resolve("ledger.csv.lock");
        int rows = 64;
        ExecutorService executor = Executors.newFixedThreadPool(8);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < rows; i++) {
                int value = i;
                futures.add(executor.submit(() -> {
                    CsvUtil.append(
                            csv, List.of("id", "value"),
                            List.of(Integer.toString(value),
                                    "row-" + value));
                    return null;
                }));
            }
            for (Future<?> future : futures) future.get();
        } finally {
            executor.shutdownNow();
        }

        List<Map<String, String>> observed = CsvUtil.read(csv);
        require(observed.size() == rows,
                "expected " + rows + " rows, observed "
                        + observed.size());
        Set<String> ids = new HashSet<>();
        for (Map<String, String> row : observed) {
            require(("row-" + row.get("id")).equals(row.get("value")),
                    "CSV row was torn or interleaved: " + row);
            ids.add(row.get("id"));
        }
        require(ids.size() == rows,
                "concurrent append lost or duplicated an ID");

        Files.deleteIfExists(csv);
        Files.deleteIfExists(lock);
        Files.deleteIfExists(directory);
        System.out.println(
                "CsvConcurrentAppendSelfTest: ALL TESTS PASSED");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
