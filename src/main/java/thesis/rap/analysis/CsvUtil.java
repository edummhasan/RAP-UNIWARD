package thesis.rap.analysis;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PushbackReader;
import java.io.Reader;
import java.io.StringReader;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Strict UTF-8 CSV helper for the RFC-4180 field grammar.
 *
 * <p>The reader accepts CRLF, LF, or CR record separators and preserves line
 * breaks inside quoted fields. It rejects unmatched quotes, quotes inside an
 * unquoted field, characters following a closing quote, duplicate or blank
 * header names, and data rows whose field count differs from the header. The
 * writer quotes commas, double quotes, and line breaks and doubles embedded
 * double quotes.</p>
 */
public final class CsvUtil {
    private CsvUtil() {}

    public static synchronized void append(
            Path path, List<String> header, List<String> row
    ) throws IOException {
        appendRows(path, header, List.of(row));
    }

    /**
     * Appends multiple rows with one schema check and one file open.
     * A sibling lock file provides inter-process protection for parallel batch
     * workers.
     */
    public static synchronized void appendRows(
            Path path, List<String> header, List<List<String>> rows
    ) throws IOException {
        validateHeader(header, "requested append schema");
        if (rows.isEmpty()) return;
        for (List<String> row : rows) {
            if (header.size() != row.size()) {
                throw new IllegalArgumentException(
                        "CSV header/row size mismatch: "
                                + header.size() + " vs " + row.size());
            }
        }
        Path absolute = path.toAbsolutePath();
        Path parent = absolute.getParent();
        if (parent != null) Files.createDirectories(parent);
        Path lockPath = absolute.resolveSibling(
                absolute.getFileName().toString() + ".lock");

        try (FileChannel lockChannel = FileChannel.open(
                    lockPath, StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE);
             FileLock fileLock = lockChannel.lock()) {
            if (!fileLock.isValid()) {
                throw new IOException(
                        "Could not acquire CSV lock: " + lockPath);
            }
            boolean exists = Files.exists(absolute)
                    && Files.size(absolute) > 0;
            if (exists) {
                requireTerminatedExistingFile(absolute);
                try (RecordReader reader = openRecords(absolute)) {
                    List<String> observed = reader.readRecord();
                    if (observed == null || !observed.equals(header)) {
                        throw new IOException(
                                "Refusing to append to CSV with incompatible "
                                        + "schema: " + absolute);
                    }
                }
            }

            try (BufferedWriter writer = Files.newBufferedWriter(
                    absolute,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND)) {
                if (!exists) {
                    writer.write(encodeRow(header));
                    writer.newLine();
                }
                for (List<String> row : rows) {
                    writer.write(encodeRow(row));
                    writer.newLine();
                }
            }
        }
    }

    /**
     * Every file produced by this writer ends each logical record with CR or
     * LF. Refusing a non-terminated existing file prevents a crash-truncated
     * final record from being joined to the next append. A complete third-party
     * CSV without a final record separator must be normalized explicitly before
     * this append-only audit writer will accept it.
     */
    private static void requireTerminatedExistingFile(Path path)
            throws IOException {
        long size = Files.size(path);
        if (size <= 0) return;
        ByteBuffer last = ByteBuffer.allocate(1);
        try (FileChannel channel = FileChannel.open(
                path, StandardOpenOption.READ)) {
            int read = channel.read(last, size - 1);
            if (read != 1) {
                throw new IOException(
                        "Could not inspect final byte of CSV: " + path);
            }
        }
        byte value = last.array()[0];
        if (value != '\n' && value != '\r') {
            throw new IOException(
                    "Refusing to append to a CSV whose final logical record "
                            + "is not terminated: " + path);
        }
    }

    /** Reads a complete table with strict schema-width validation. */
    public static List<Map<String, String>> read(Path path)
            throws IOException {
        List<Map<String, String>> rows = new ArrayList<>();
        try (TableReader table = openTable(path)) {
            Map<String, String> row;
            while ((row = table.readRow()) != null) rows.add(row);
        }
        return rows;
    }

    /** Opens a streaming, schema-aware CSV table reader. */
    public static TableReader openTable(Path path) throws IOException {
        return new TableReader(path);
    }

    /** Opens a streaming record reader when positional fields are required. */
    public static RecordReader openRecords(Path path) throws IOException {
        return new RecordReader(
                Files.newBufferedReader(path, StandardCharsets.UTF_8),
                path.toString());
    }

    /** Counts logical data records, not physical lines. */
    public static long countDataRows(Path path) throws IOException {
        long count = 0;
        try (TableReader table = openTable(path)) {
            while (table.readRow() != null) count++;
        }
        return count;
    }

    public static String encodeRow(List<String> values) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) text.append(',');
            text.append(escape(values.get(i)));
        }
        return text.toString();
    }

    /**
     * Parses exactly one logical CSV record. Retained for focused tests and
     * callers that already hold a complete record; file readers should use
     * {@link #openTable(Path)} or {@link #openRecords(Path)}.
     */
    public static List<String> parseLine(String line) {
        if (line == null) {
            throw new IllegalArgumentException("CSV record cannot be null");
        }
        try (RecordReader reader = new RecordReader(
                new StringReader(line), "in-memory CSV record")) {
            List<String> record = reader.readRecord();
            List<String> extra = reader.readRecord();
            if (record == null || extra != null) {
                throw new IllegalArgumentException(
                        "Input must contain exactly one CSV record");
            }
            return record;
        } catch (IOException invalid) {
            throw new IllegalArgumentException(
                    "Malformed CSV record: " + invalid.getMessage(), invalid);
        }
    }

    private static String escape(String value) {
        if (value == null) return "";
        boolean quote = value.indexOf(',') >= 0
                || value.indexOf('"') >= 0
                || value.indexOf('\n') >= 0
                || value.indexOf('\r') >= 0;
        if (!quote) return value;
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    private static void validateHeader(
            List<String> header, String source
    ) throws IOException {
        if (header == null || header.isEmpty()) {
            throw new IOException("CSV header is empty in " + source);
        }
        Set<String> unique = new HashSet<>();
        for (String name : header) {
            if (name == null || name.isBlank()) {
                throw new IOException(
                        "CSV header contains a blank name in " + source);
            }
            if (!unique.add(name)) {
                throw new IOException(
                        "Duplicate CSV header in " + source + ": " + name);
            }
        }
    }

    private static boolean blankRecord(List<String> record) {
        return record.size() == 1 && record.get(0).isEmpty();
    }

    /** Streaming logical-record reader with RFC-4180 quote handling. */
    public static final class RecordReader implements AutoCloseable {
        private final PushbackReader reader;
        private final String source;
        private long completedRecords;

        private RecordReader(Reader reader, String source) {
            this.reader = new PushbackReader(reader, 1);
            this.source = source;
        }

        public List<String> readRecord() throws IOException {
            List<String> fields = new ArrayList<>();
            StringBuilder field = new StringBuilder();
            boolean inQuotes = false;
            boolean afterClosingQuote = false;
            boolean atFieldStart = true;
            boolean consumedAnything = false;

            while (true) {
                int raw = reader.read();
                if (raw < 0) {
                    if (!consumedAnything && fields.isEmpty()
                            && field.isEmpty()) {
                        return null;
                    }
                    if (inQuotes) {
                        throw malformed("unterminated quoted field");
                    }
                    fields.add(field.toString());
                    completedRecords++;
                    return fields;
                }

                consumedAnything = true;
                char ch = (char) raw;
                if (inQuotes) {
                    if (ch == '"') {
                        int next = reader.read();
                        if (next == '"') {
                            field.append('"');
                        } else {
                            inQuotes = false;
                            afterClosingQuote = true;
                            if (next >= 0) reader.unread(next);
                        }
                    } else {
                        field.append(ch);
                    }
                    continue;
                }

                if (afterClosingQuote) {
                    if (ch == ',') {
                        fields.add(field.toString());
                        field.setLength(0);
                        afterClosingQuote = false;
                        atFieldStart = true;
                    } else if (ch == '\r' || ch == '\n') {
                        consumeOptionalLf(ch);
                        fields.add(field.toString());
                        completedRecords++;
                        return fields;
                    } else {
                        throw malformed(
                                "unexpected character after a closing quote");
                    }
                    continue;
                }

                if (ch == ',') {
                    fields.add(field.toString());
                    field.setLength(0);
                    atFieldStart = true;
                } else if (ch == '\r' || ch == '\n') {
                    consumeOptionalLf(ch);
                    fields.add(field.toString());
                    completedRecords++;
                    return fields;
                } else if (ch == '"') {
                    if (!atFieldStart) {
                        throw malformed(
                                "quote inside an unquoted field");
                    }
                    inQuotes = true;
                    atFieldStart = false;
                } else {
                    field.append(ch);
                    atFieldStart = false;
                }
            }
        }

        private void consumeOptionalLf(char separator) throws IOException {
            if (separator != '\r') return;
            int next = reader.read();
            if (next >= 0 && next != '\n') reader.unread(next);
        }

        private IOException malformed(String detail) {
            return new IOException(
                    "Malformed CSV in " + source + " near logical record "
                            + (completedRecords + 1) + ": " + detail);
        }

        @Override
        public void close() throws IOException {
            reader.close();
        }
    }

    /** Streaming CSV table reader with strict header and row-width checks. */
    public static final class TableReader implements AutoCloseable {
        private final Path source;
        private final RecordReader records;
        private final List<String> header;
        private long dataRecordNumber;

        private TableReader(Path source) throws IOException {
            this.source = source;
            this.records = openRecords(source);
            List<String> observed = records.readRecord();
            if (observed == null) {
                records.close();
                throw new IOException("Empty CSV file: " + source);
            }
            if (!observed.isEmpty()
                    && observed.get(0).startsWith("\ufeff")) {
                observed.set(0, observed.get(0).substring(1));
            }
            validateHeader(observed, source.toString());
            this.header = List.copyOf(observed);
        }

        public List<String> header() {
            return header;
        }

        public Map<String, String> readRow() throws IOException {
            List<String> values;
            do {
                values = records.readRecord();
            } while (values != null && blankRecord(values));
            if (values == null) return null;
            dataRecordNumber++;
            if (values.size() != header.size()) {
                throw new IOException(
                        "CSV row width mismatch in " + source
                                + " at data record " + dataRecordNumber
                                + ": expected " + header.size()
                                + ", observed " + values.size());
            }
            Map<String, String> row = new LinkedHashMap<>();
            for (int i = 0; i < header.size(); i++) {
                row.put(header.get(i), values.get(i));
            }
            return row;
        }

        @Override
        public void close() throws IOException {
            records.close();
        }
    }
}
