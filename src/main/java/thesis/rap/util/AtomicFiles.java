package thesis.rap.util;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Small helper for atomically publishing one complete artifact. */
public final class AtomicFiles {
    private AtomicFiles() {}

    public static void writeString(
            Path target, String content, Charset charset
    ) throws IOException {
        writeBytes(target, content.getBytes(charset));
    }

    public static void writeBytes(
            Path target, byte[] content
    ) throws IOException {
        Path absolute = target.toAbsolutePath();
        Path parent = absolute.getParent();
        if (parent != null) Files.createDirectories(parent);
        Path directory = parent == null
                ? Path.of(".").toAbsolutePath() : parent;
        Path temp = Files.createTempFile(
                directory,
                "." + absolute.getFileName() + ".tmp-",
                ".part");
        boolean published = false;
        try {
            Files.write(temp, content);
            try {
                Files.move(
                        temp, absolute,
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(
                        temp, absolute,
                        StandardCopyOption.REPLACE_EXISTING);
            }
            published = true;
        } finally {
            if (!published) Files.deleteIfExists(temp);
        }
    }
}
