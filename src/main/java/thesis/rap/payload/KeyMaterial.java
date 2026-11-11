package thesis.rap.payload;

/** Transport-safe holder for the three distinct keys required by the conformance profile. */
public record KeyMaterial(String encryptionKey, String xorKey, String stcKey) {
    private static final String SEPARATOR = "\u001f";

    public KeyMaterial {
        require(encryptionKey, "encryptionKey");
        require(xorKey, "xorKey");
        require(stcKey, "stcKey");
        if (encryptionKey.equals(xorKey) || encryptionKey.equals(stcKey) || xorKey.equals(stcKey)) {
            throw new IllegalArgumentException("encryptionKey, xorKey, and stcKey must be distinct");
        }
    }

    public String encode() { return encryptionKey + SEPARATOR + xorKey + SEPARATOR + stcKey; }

    public static KeyMaterial decode(String encoded) {
        if (encoded == null) throw new IllegalArgumentException("encoded key material cannot be null");
        String[] parts = encoded.split(SEPARATOR, -1);
        if (parts.length == 3) return new KeyMaterial(parts[0], parts[1], parts[2]);
        throw new IllegalArgumentException("Strict conformance requires three independently supplied keys");
    }

    private static void require(String value, String name) {
        if (value == null || value.isBlank() || value.contains(SEPARATOR)) {
            throw new IllegalArgumentException(name + " must be nonblank and may not contain the key separator");
        }
    }
}
