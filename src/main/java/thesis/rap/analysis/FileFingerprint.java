package thesis.rap.analysis;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;

/** Streaming SHA-256 helper; does not load large datasets into memory. */
public final class FileFingerprint {
    private FileFingerprint() {}

    public static String sha256(Path path) throws IOException {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream in = Files.newInputStream(path)) {
                byte[] buffer = new byte[1024 * 1024];
                int n;
                while ((n = in.read(buffer)) >= 0) {
                    if (n > 0) md.update(buffer, 0, n);
                }
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }
}
