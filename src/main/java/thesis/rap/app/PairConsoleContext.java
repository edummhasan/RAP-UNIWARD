package thesis.rap.app;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Adds an immutable pair identity to every application console line.
 *
 * <p>The wrapper buffers one line per Java thread and emits the prefix plus
 * the complete line while holding one shared output lock. This prevents
 * fragments from different pair workers from being spliced together. The
 * identifiers are always treated as text; for example, image IDs
 * {@code 000001} and {@code 0001} remain distinct.</p>
 */
final class PairConsoleContext {
    private static final Object OUTPUT_LOCK = new Object();
    private static final ThreadLocal<Identity> CURRENT = new ThreadLocal<>();
    private static final Identity UNSCOPED =
            new Identity("UNSCOPED", "-", "-");
    private static boolean installed;

    private PairConsoleContext() {}

    /** Installs the prefixing streams exactly once for this JVM. */
    static synchronized void install() {
        if (installed) return;
        PrintStream originalOut = System.out;
        PrintStream originalErr = System.err;
        System.setOut(new PrintStream(
                new PrefixingOutputStream(originalOut), true,
                StandardCharsets.UTF_8));
        System.setErr(new PrintStream(
                new PrefixingOutputStream(originalErr), true,
                StandardCharsets.UTF_8));
        installed = true;
    }

    static Scope command(String command) {
        String pairId = command == null || command.isBlank()
                ? "COMMAND"
                : command.trim().toUpperCase(Locale.ROOT)
                        .replace('-', '_');
        return open(new Identity(pairId, "-", "-"));
    }

    static Scope singlePair(Path image, Path message) {
        return open(new Identity(
                "SINGLE", exactStem(image), exactStem(message)));
    }

    static Scope extraction(Path image) {
        return open(new Identity(
                "EXTRACT", exactStem(image), "-"));
    }

    static Scope batchPair(
            long pairIndex,
            long totalPairs,
            Path image,
            Path message
    ) {
        if (pairIndex < 1L) {
            throw new IllegalArgumentException(
                    "pairIndex must be positive");
        }
        if (totalPairs < pairIndex) {
            throw new IllegalArgumentException(
                    "totalPairs cannot be less than pairIndex");
        }
        int width = Math.max(
                6, Long.toString(totalPairs).length());
        String pairId = String.format(
                Locale.ROOT, "%0" + width + "d", pairIndex);
        return open(new Identity(
                pairId, exactStem(image), exactStem(message)));
    }

    private static Scope open(Identity identity) {
        Identity previous = CURRENT.get();
        CURRENT.set(identity);
        return new Scope(Thread.currentThread(), previous);
    }

    private static Identity currentIdentity() {
        Identity identity = CURRENT.get();
        return identity == null ? UNSCOPED : identity;
    }

    static String exactStem(Path path) {
        String name = path.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    static final class Scope implements AutoCloseable {
        private final Thread owner;
        private final Identity previous;
        private boolean closed;

        private Scope(Thread owner, Identity previous) {
            this.owner = owner;
            this.previous = previous;
        }

        @Override
        public void close() {
            if (closed) return;
            if (Thread.currentThread() != owner) {
                throw new IllegalStateException(
                        "Console context must close on its opening thread");
            }
            if (previous == null) CURRENT.remove();
            else CURRENT.set(previous);
            closed = true;
        }
    }

    private static final class Identity {
        private final byte[] prefix;

        private Identity(
                String pairId,
                String imageId,
                String messageId
        ) {
            String value = "[pair_id=" + safeField(pairId) + "]"
                    + "+[image_id=" + safeField(imageId) + "]"
                    + "+[msg_id=" + safeField(messageId) + "] ";
            prefix = value.getBytes(StandardCharsets.UTF_8);
        }

        private static String safeField(String value) {
            if (value == null) return "-";
            StringBuilder safe = new StringBuilder(value.length());
            for (int i = 0; i < value.length(); i++) {
                char ch = value.charAt(i);
                switch (ch) {
                    case '\\' -> safe.append("\\\\");
                    case ']' -> safe.append("\\]");
                    case '\r' -> safe.append("\\r");
                    case '\n' -> safe.append("\\n");
                    case '\t' -> safe.append("\\t");
                    default -> safe.append(Character.isISOControl(ch)
                            ? '?' : ch);
                }
            }
            return safe.toString();
        }
    }

    private static final class PrefixingOutputStream
            extends OutputStream {
        private final PrintStream target;
        private final ThreadLocal<LineState> lines =
                ThreadLocal.withInitial(LineState::new);

        private PrefixingOutputStream(PrintStream target) {
            this.target = target;
        }

        @Override
        public void write(int value) throws IOException {
            LineState line = lines.get();
            line.beginIfNecessary();
            line.bytes.write(value);
            if ((value & 0xff) == '\n') emit(line);
        }

        @Override
        public void write(byte[] bytes, int offset, int length)
                throws IOException {
            if (bytes == null) throw new NullPointerException("bytes");
            if (offset < 0 || length < 0
                    || offset > bytes.length - length) {
                throw new IndexOutOfBoundsException();
            }
            LineState line = lines.get();
            int end = offset + length;
            int cursor = offset;
            while (cursor < end) {
                line.beginIfNecessary();
                int newline = cursor;
                while (newline < end && bytes[newline] != '\n') {
                    newline++;
                }
                int chunkEnd = newline < end ? newline + 1 : end;
                line.bytes.write(bytes, cursor, chunkEnd - cursor);
                if (newline < end) emit(line);
                cursor = chunkEnd;
            }
        }

        @Override
        public void flush() {
            synchronized (OUTPUT_LOCK) {
                target.flush();
            }
        }

        @Override
        public void close() {
            flush();
        }

        private void emit(LineState line) throws IOException {
            byte[] content = line.bytes.toByteArray();
            Identity identity = line.identity;
            line.reset();
            synchronized (OUTPUT_LOCK) {
                target.write(identity.prefix);
                target.write(content);
                target.flush();
            }
        }
    }

    private static final class LineState {
        private final ByteArrayOutputStream bytes =
                new ByteArrayOutputStream(256);
        private Identity identity;

        private void beginIfNecessary() {
            if (identity == null) identity = currentIdentity();
        }

        private void reset() {
            bytes.reset();
            identity = null;
        }
    }
}
