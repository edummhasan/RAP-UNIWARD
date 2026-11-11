package thesis.rap.app;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/** Compact exception identity for CSV records; complete stack traces go to the log. */
final class FailureDiagnostics {
    private FailureDiagnostics() {}

    record Info(String className, String message, String stackHash, String origin) {}

    static Info info(Throwable failure) {
        if (failure == null) return new Info("", "", "", "");
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(failure.getClass().getName().getBytes(StandardCharsets.UTF_8));
            StackTraceElement[] stack = failure.getStackTrace();
            for (StackTraceElement frame : stack) {
                digest.update(("\n" + frame).getBytes(StandardCharsets.UTF_8));
            }
            return new Info(failure.getClass().getName(),
                    failure.getMessage() == null ? "" : failure.getMessage(),
                    HexFormat.of().formatHex(digest.digest()),
                    stack.length == 0 ? "" : stack[0].toString());
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    static void log(Throwable failure) {
        Info info = info(failure);
        System.err.printf("EXCEPTION_DIAGNOSTIC class=%s stack_hash=%s origin=%s%n",
                info.className(), info.stackHash(), info.origin());
        failure.printStackTrace(System.err);
    }
}
