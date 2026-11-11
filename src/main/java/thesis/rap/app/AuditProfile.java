package thesis.rap.app;

import java.util.Locale;

/**
 * Controls evidence volume without changing candidate construction, costs,
 * carrier allocation, STC embedding, JPEG write-back, or verification.
 */
public enum AuditProfile {
    /** Compact grouped candidate evidence and stream hashes. */
    THESIS,
    /** THESIS evidence plus one detailed row per modified coefficient. */
    CHANGES,
    /** CHANGES evidence plus the complete candidate universe as gzip CSV. */
    FULL_AUDIT;

    public static AuditProfile parse(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "audit profile must be THESIS, CHANGES, or FULL_AUDIT");
        }
        String normalized = value.trim()
                .toUpperCase(Locale.ROOT)
                .replace('-', '_');
        return switch (normalized) {
            case "THESIS" -> THESIS;
            case "CHANGES" -> CHANGES;
            case "FULL", "FULL_AUDIT" -> FULL_AUDIT;
            default -> throw new IllegalArgumentException(
                    "audit profile must be THESIS, CHANGES, or FULL_AUDIT: "
                            + value);
        };
    }

    /** Compact evidence is the default within QF_FORENSIC mode. */
    public static AuditProfile forensicDefault() {
        return THESIS;
    }

    
    public static AuditProfile uncappedDefault() {
        return FULL_AUDIT;
    }

    public boolean writesChanges() {
        return this == CHANGES || this == FULL_AUDIT;
    }

    public boolean writesFullCandidateUniverse() {
        return this == FULL_AUDIT;
    }

    public boolean writesExpandedAggregateTables() {
        return this == FULL_AUDIT;
    }
}
