package thesis.rap.analysis;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** Regression tests for logical-record CSV parsing and strict schema checks. */
public final class CsvStrictParsingSelfTest {
    private CsvStrictParsingSelfTest() {}

    public static void main(String[] args) throws Exception {
        Path directory = Files.createTempDirectory("rap-csv-strict-");
        try {
            roundTripsQuotedLineBreaks(directory);
            acceptsCrLfAndEscapedQuotes(directory);
            rejectsMalformedGrammar(directory);
            rejectsSchemaDefects(directory);
            rejectsAppendAfterTruncatedRecord(directory);
            System.out.println("CsvStrictParsingSelfTest: ALL TESTS PASSED");
        } finally {
            try (var paths = Files.walk(directory)) {
                paths.sorted(java.util.Comparator.reverseOrder())
                        .forEach(path -> {
                            try {
                                Files.deleteIfExists(path);
                            } catch (IOException failure) {
                                throw new RuntimeException(failure);
                            }
                        });
            }
        }
    }

    private static void roundTripsQuotedLineBreaks(Path directory)
            throws Exception {
        Path csv = directory.resolve("round-trip.csv");
        List<String> header = List.of("id", "detail", "note");
        List<String> first = List.of(
                "1", "first line\nsecond line", "comma,quote \"x\"");
        List<String> second = List.of(
                "2", "carriage\rreturn", "plain");
        CsvUtil.appendRows(csv, header, List.of(first, second));
        List<Map<String, String>> rows = CsvUtil.read(csv);
        assert rows.size() == 2;
        assert rows.get(0).get("detail").equals(first.get(1));
        assert rows.get(0).get("note").equals(first.get(2));
        assert rows.get(1).get("detail").equals(second.get(1));
        assert CsvUtil.countDataRows(csv) == 2
                : "logical rows, not physical lines, must be counted";
    }

    private static void acceptsCrLfAndEscapedQuotes(Path directory)
            throws Exception {
        Path csv = directory.resolve("crlf.csv");
        Files.writeString(
                csv,
                "id,detail\r\n1,\"alpha\r\nbeta\"\r\n"
                        + "2,\"said \"\"yes\"\"\"\r\n",
                StandardCharsets.UTF_8);
        List<Map<String, String>> rows = CsvUtil.read(csv);
        assert rows.size() == 2;
        assert rows.get(0).get("detail").equals("alpha\r\nbeta");
        assert rows.get(1).get("detail").equals("said \"yes\"");
    }

    private static void rejectsMalformedGrammar(Path directory)
            throws Exception {
        assertMalformed(directory, "unterminated.csv", "a,b\n1,\"open\n");
        assertMalformed(directory, "unquoted-quote.csv", "a,b\n1,bad\"quote\n");
        assertMalformed(directory, "post-quote.csv", "a,b\n1,\"ok\"junk\n");
    }

    private static void rejectsSchemaDefects(Path directory)
            throws Exception {
        assertMalformed(directory, "short-row.csv", "a,b\n1\n");
        assertMalformed(directory, "long-row.csv", "a,b\n1,2,3\n");
        assertMalformed(directory, "duplicate.csv", "a,a\n1,2\n");
        assertMalformed(directory, "blank-header.csv", "a,\n1,2\n");
    }

    private static void rejectsAppendAfterTruncatedRecord(Path directory)
            throws Exception {
        Path csv = directory.resolve("truncated-append.csv");
        Files.writeString(csv, "a,b\n1,partial", StandardCharsets.UTF_8);
        boolean rejected = false;
        try {
            CsvUtil.append(csv, List.of("a", "b"), List.of("2", "safe"));
        } catch (IOException expected) {
            rejected = true;
        }
        assert rejected : "append accepted a non-terminated existing record";
        assert Files.readString(csv, StandardCharsets.UTF_8)
                .equals("a,b\n1,partial")
                : "failed append must not modify the existing ledger";
    }

    private static void assertMalformed(
            Path directory, String name, String content
    ) throws Exception {
        Path csv = directory.resolve(name);
        Files.writeString(csv, content, StandardCharsets.UTF_8);
        boolean rejected = false;
        try {
            CsvUtil.read(csv);
        } catch (IOException expected) {
            rejected = true;
        }
        assert rejected : "malformed CSV was accepted: " + name;
    }
}
