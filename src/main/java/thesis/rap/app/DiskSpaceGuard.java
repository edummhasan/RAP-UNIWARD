package thesis.rap.app;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileStore;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/** Fail-closed free-byte and free-inode checks for long corpus runs. */
public final class DiskSpaceGuard {
    private static final long DF_TIMEOUT_SECONDS = 10L;

    private DiskSpaceGuard() {}

    public record Limits(
            long minimumFreeBytes,
            long minimumFreeInodes,
            int checkEveryPairs
    ) {
        public Limits {
            if (minimumFreeBytes < 0L) {
                throw new IllegalArgumentException(
                        "minimumFreeBytes must be nonnegative");
            }
            if (minimumFreeInodes < 0L) {
                throw new IllegalArgumentException(
                        "minimumFreeInodes must be nonnegative");
            }
            if (checkEveryPairs < 1) {
                throw new IllegalArgumentException(
                        "checkEveryPairs must be at least one");
            }
        }
    }

    public static final class InsufficientStorageException
            extends IOException {
        private static final long serialVersionUID = 1L;

        InsufficientStorageException(String message) {
            super(message);
        }
    }

    public static void check(Limits limits, Path... roots)
            throws IOException {
        if (limits == null || roots == null || roots.length == 0) {
            throw new IllegalArgumentException(
                    "limits and at least one storage root are required");
        }
        List<String> checkedStores = new ArrayList<>();
        for (Path root : roots) {
            if (root == null) continue;
            Path absolute = root.toAbsolutePath().normalize();
            Files.createDirectories(absolute);
            FileStore store = Files.getFileStore(absolute);
            String identity = store.name() + "|" + store.type();
            if (checkedStores.contains(identity)) continue;
            checkedStores.add(identity);

            long usableBytes = store.getUsableSpace();
            if (usableBytes < limits.minimumFreeBytes()) {
                throw new InsufficientStorageException(
                        "Free-byte reserve violated for " + absolute
                                + ": usable=" + usableBytes
                                + ", required=" + limits.minimumFreeBytes());
            }

            if (limits.minimumFreeInodes() > 0L) {
                long availableInodes = availableInodes(absolute);
                if (availableInodes < limits.minimumFreeInodes()) {
                    throw new InsufficientStorageException(
                            "Free-inode reserve violated for " + absolute
                                    + ": available=" + availableInodes
                                    + ", required="
                                    + limits.minimumFreeInodes());
                }
            }
        }
    }

    static long availableInodes(Path root) throws IOException {
        Process process;
        try {
            process = new ProcessBuilder(
                    "df", "-Pi", "--", root.toAbsolutePath().toString())
                    .redirectErrorStream(true)
                    .start();
        } catch (IOException unavailable) {
            throw new IOException(
                    "Cannot execute df -Pi for inode protection", unavailable);
        }
        byte[] output;
        try {
            boolean finished = process.waitFor(
                    DF_TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new IOException(
                        "df -Pi timed out while checking " + root);
            }
            output = process.getInputStream().readAllBytes();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            process.destroyForcibly();
            throw new IOException(
                    "Interrupted while checking free inodes", interrupted);
        }
        String text = new String(output, StandardCharsets.UTF_8);
        if (process.exitValue() != 0) {
            throw new IOException(
                    "df -Pi failed for " + root + ": "
                            + text.replaceAll("[\\r\\n]+", " ").trim());
        }
        return parseAvailableInodes(text);
    }

    static long parseAvailableInodes(String output) throws IOException {
        if (output == null) {
            throw new IOException("df -Pi returned null output");
        }
        String[] lines = output.strip().split("\\R");
        for (int index = lines.length - 1; index >= 0; index--) {
            String line = lines[index].trim();
            if (line.isEmpty() || line.startsWith("Filesystem")) continue;
            String[] fields = line.split("\\s+");
            if (fields.length < 6) {
                throw new IOException(
                        "Unexpected df -Pi data row: " + line);
            }
            try {
                return Long.parseLong(fields[3]);
            } catch (NumberFormatException invalid) {
                throw new IOException(
                        "Invalid available-inode value in df -Pi row: "
                                + line, invalid);
            }
        }
        throw new IOException("df -Pi returned no filesystem data row");
    }
}
