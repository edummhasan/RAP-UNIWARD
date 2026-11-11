package thesis.rap.app;

import java.util.Locale;

/**
 * Selects the top-level audit data model without changing any embedding,
 * extraction, JPEG, candidate, cost, balance, or STC algorithm.
 */
public enum AuditMode {
    /** One detailed selected/last-QF row; no intermediate-QF history. */
    RESULT_ONLY,
    /** Complete QF-trial and balance-attempt evidence for forensic analysis. */
    QF_FORENSIC;

    public static AuditMode parse(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "audit mode must be RESULT_ONLY or QF_FORENSIC");
        }
        String normalized = value.trim()
                .toUpperCase(Locale.ROOT)
                .replace('-', '_');
        return switch (normalized) {
            case "RESULT", "REGULAR", "RESULT_ONLY" -> RESULT_ONLY;
            case "QF", "FORENSIC", "QF_FORENSIC" -> QF_FORENSIC;
            default -> throw new IllegalArgumentException(
                    "audit mode must be RESULT_ONLY or QF_FORENSIC: "
                            + value);
        };
    }

    public boolean writesQfForensics() {
        return this == QF_FORENSIC;
    }
}
