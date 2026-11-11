package thesis.rap.app;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Regression checks for exact, line-atomic pair console prefixes. */
@SuppressWarnings("try")
public final class PairConsoleContextSelfTest {
    private PairConsoleContextSelfTest() {}

    public static void main(String[] args) throws Exception {
        PrintStream originalOut = System.out;
        PrintStream originalErr = System.err;
        ByteArrayOutputStream stdoutBytes = new ByteArrayOutputStream();
        ByteArrayOutputStream stderrBytes = new ByteArrayOutputStream();
        System.setOut(new PrintStream(
                stdoutBytes, true, StandardCharsets.UTF_8));
        System.setErr(new PrintStream(
                stderrBytes, true, StandardCharsets.UTF_8));

        PairConsoleContext.install();
        try (PairConsoleContext.Scope ignored =
                     PairConsoleContext.command("batch-embed")) {
            System.out.println("BATCH_LINE");
        }
        try (PairConsoleContext.Scope ignored =
                     PairConsoleContext.batchPair(
                             1, 53_250,
                             Path.of("000001.png"),
                             Path.of("001.msg"))) {
            System.out.print("PAIR_LINE_ONE\nPAIR_LINE_TWO\n");
            System.err.println("PAIR_ERROR");
        }
        try (PairConsoleContext.Scope ignored =
                     PairConsoleContext.batchPair(
                             2, 53_250,
                             Path.of("0001.png"),
                             Path.of("001.msg"))) {
            System.out.println("SHORT_TEXT_ID");
        }

        ExecutorService executor = Executors.newFixedThreadPool(8);
        List<Future<?>> futures = new ArrayList<>();
        try {
            for (int i = 1; i <= 64; i++) {
                final int pair = i;
                futures.add(executor.submit(() -> {
                    Path image = Path.of(String.format(
                            "%06d.png", pair));
                    Path message = Path.of(String.format(
                            "%03d.msg", (pair % 15) + 1));
                    try (PairConsoleContext.Scope ignored =
                                 PairConsoleContext.batchPair(
                                         pair, 64, image, message)) {
                        System.out.println("TOKEN_" + pair);
                    }
                }));
            }
            for (Future<?> future : futures) future.get();
        } finally {
            executor.shutdownNow();
        }

        System.out.flush();
        System.err.flush();
        String stdout = stdoutBytes.toString(StandardCharsets.UTF_8);
        String stderr = stderrBytes.toString(StandardCharsets.UTF_8);
        System.setOut(originalOut);
        System.setErr(originalErr);

        require(stdout.contains(
                        "[pair_id=BATCH_EMBED]+[image_id=-]+[msg_id=-] BATCH_LINE"),
                "batch command prefix is missing");
        require(stdout.contains(
                        "[pair_id=000001]+[image_id=000001]+[msg_id=001] PAIR_LINE_ONE"),
                "full textual pair prefix is missing");
        require(stdout.contains(
                        "[pair_id=000002]+[image_id=0001]+[msg_id=001] SHORT_TEXT_ID"),
                "leading-zero textual identity was changed");
        require(stderr.contains(
                        "[pair_id=000001]+[image_id=000001]+[msg_id=001] PAIR_ERROR"),
                "stderr pair prefix is missing");

        int tokenLines = 0;
        for (String line : stdout.split("\\R")) {
            if (line.isEmpty()) continue;
            require(line.startsWith("[pair_id="),
                    "unprefixed stdout line: " + line);
            if (line.contains("TOKEN_")) {
                tokenLines++;
                require(line.indexOf("TOKEN_")
                                == line.lastIndexOf("TOKEN_"),
                        "concurrent console lines were spliced: " + line);
            }
        }
        for (String line : stderr.split("\\R")) {
            if (line.isEmpty()) continue;
            require(line.startsWith("[pair_id="),
                    "unprefixed stderr line: " + line);
        }
        require(tokenLines == 64,
                "expected 64 complete concurrent token lines, got "
                        + tokenLines);
        originalOut.println(
                "PairConsoleContextSelfTest: ALL TESTS PASSED");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
