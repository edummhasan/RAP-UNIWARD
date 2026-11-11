package thesis.rap.payload;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import thesis.rap.image.SamplingGeometry;

/**
 * Packs and unpacks the authenticated payload carried by the JPEG coefficient channel.
 *
 * <p>The detailed API exposes the exact byte accounting used by the real payload path so
 * evaluation code never has to infer or simulate payload overhead.</p>
 */
public final class PayloadPackage {
    public static final int MIN_SUPPORTED_QF = 75;
    public static final int MAX_SUPPORTED_QF = 100;
    
    public static final int PUBLIC_MARKER_LEN = 0;
    
    public static final int QF_LEN = 0;
    /** Compact wire length field; the authenticated AAD continues to encode the length as uint32. */
    public static final int CIPHERTEXT_LENGTH_FIELD_LEN = Short.BYTES;
    public static final int MAX_CIPHERTEXT_LEN = 0xFFFF;
    /** NIST-recommended 96-bit GCM IV/nonce length retained unchanged. */
    public static final int NONCE_LEN = 12;
    
    public static final int TAG_LEN = 12;
    public static final int PREPROCESSING_METADATA_LEN = 1 + 4 + 4;

    private static final int AAD_QF_LEN = 1;
    private static final int AAD_CIPHERTEXT_LENGTH_FIELD_LEN = Integer.BYTES;
    private static final byte[] AAD_DOMAIN =
            "AHP-PAYLOAD-AAD-V14.2.0".getBytes(StandardCharsets.US_ASCII);
    public static final int AAD_DOMAIN_LEN = AAD_DOMAIN.length;
    public static final int XOR_KEY_COMMITMENT_LEN = 32;
    
    public static final int AAD_LEN = AAD_DOMAIN_LEN + AAD_QF_LEN
            + AAD_CIPHERTEXT_LENGTH_FIELD_LEN + XOR_KEY_COMMITMENT_LEN;
    private static final byte[] GEOMETRY_AAD_DOMAIN =
            "AHP-PAYLOAD-AAD-V14.2.0-GEOM-V1"
                    .getBytes(StandardCharsets.US_ASCII);
    public static final int GEOMETRY_SIGNATURE_LEN = Integer.BYTES;
    /** Production geometry-bound AAD remains 72 bytes in the compact-header profile. */
    public static final int GEOMETRY_AAD_LEN = GEOMETRY_AAD_DOMAIN.length
            + AAD_QF_LEN + GEOMETRY_SIGNATURE_LEN
            + AAD_CIPHERTEXT_LENGTH_FIELD_LEN + XOR_KEY_COMMITMENT_LEN;
    private static final SecureRandom RNG = new SecureRandom();

    private PayloadPackage() {}

    public static int headerBytes() {
        return CIPHERTEXT_LENGTH_FIELD_LEN + NONCE_LEN + TAG_LEN;
    }

    /**
     * QF-independent payload preprocessing. Huffman selection and framing are
     * deterministic for one message and therefore must not be repeated for
     * every adaptive-QF trial.
     */
    public static PreparedMessage prepareMessage(byte[] message) {
        if (message == null) throw new IllegalArgumentException("message cannot be null");
        PreparedPayload prepared = preparePlainPayload(message);
        return new PreparedMessage(prepared.bytes, prepared.layout);
    }

    /** Exact deterministic payload-size accounting before encryption randomness is generated. */
    public static PayloadLayout inspectLayout(byte[] message) {
        return prepareMessage(message).layout();
    }

    public static PayloadLayout inspectLayout(PreparedMessage message) {
        if (message == null) throw new IllegalArgumentException("message cannot be null");
        return message.layout();
    }

    /** Builds the real encrypted payload and returns the exact observed layout. */
    public static PackResult packDetailed(byte[] message, String encodedKeys, int qf) throws GeneralSecurityException {
        return packDetailed(prepareMessage(message), encodedKeys, qf);
    }

    /** Production geometry-bound payload packing; geometry is authenticated as AAD. */
    public static PackResult packDetailed(
            byte[] message, String encodedKeys, int qf, SamplingGeometry geometry
    ) throws GeneralSecurityException {
        return packDetailed(prepareMessage(message), encodedKeys, qf, geometry);
    }

    /** Production geometry-bound payload packing for a preprocessed message. */
    public static PackResult packDetailed(
            PreparedMessage message, String encodedKeys, int qf,
            SamplingGeometry geometry
    ) throws GeneralSecurityException {
        if (geometry == null) throw new IllegalArgumentException("geometry cannot be null");
        KeyMaterial keys = KeyMaterial.decode(encodedKeys);
        return packDetailedGeometryBound(
                message, keys.encryptionKey(), keys.xorKey(), qf, geometry);
    }

    public static PackResult packDetailed(
            PreparedMessage message, String encodedKeys, int qf
    ) throws GeneralSecurityException {
        KeyMaterial keys = KeyMaterial.decode(encodedKeys);
        return packDetailed(message, keys.encryptionKey(), keys.xorKey(), qf);
    }

    public static PackResult packDetailed(byte[] message, String encryptionKey, String xorKey, int qf) throws GeneralSecurityException {
        return packDetailed(prepareMessage(message), encryptionKey, xorKey, qf);
    }

    public static PackResult packDetailed(
            PreparedMessage message,
            String encryptionKey,
            String xorKey,
            int qf
    ) throws GeneralSecurityException {
        validateQf(qf);
        if (message == null) throw new IllegalArgumentException("message cannot be null");
        byte[] prepared = xorPrepared(message.preparedBytes, xorKey);

        byte[] nonce = new byte[NONCE_LEN];
        RNG.nextBytes(nonce);

        int cipherLen = prepared.length;
        requireSupportedCiphertextLength(cipherLen);

        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(
                Cipher.ENCRYPT_MODE,
                new SecretKeySpec(key(encryptionKey), "AES"),
                new GCMParameterSpec(TAG_LEN * 8, nonce)
        );
        cipher.updateAAD(aadFor(qf, cipherLen, xorKey));

        byte[] encryptedWithTag = cipher.doFinal(prepared);
        int observedCipherLen = encryptedWithTag.length - TAG_LEN;
        if (observedCipherLen != cipherLen) {
            throw new IllegalStateException("AES-GCM ciphertext length differs from prepared plaintext length");
        }
        byte[] cipherText = Arrays.copyOfRange(encryptedWithTag, 0, cipherLen);
        byte[] tag = Arrays.copyOfRange(encryptedWithTag, cipherLen, encryptedWithTag.length);

        int packedLength = Math.addExact(
                headerBytes(), cipherLen);
        ByteBuffer bb = ByteBuffer.allocate(packedLength)
                .order(ByteOrder.BIG_ENDIAN);
        putUnsignedShort(bb, cipherLen);
        bb.put(nonce);
        bb.put(tag);
        bb.put(cipherText);

        byte[] packed = bb.array();
        PayloadLayout observed = message.layout.withPackedLength(packed.length, cipherLen);
        return new PackResult(packed, observed);
    }

    private static PackResult packDetailedGeometryBound(
            PreparedMessage message,
            String encryptionKey,
            String xorKey,
            int qf,
            SamplingGeometry geometry
    ) throws GeneralSecurityException {
        validateQf(qf);
        if (message == null || geometry == null) {
            throw new IllegalArgumentException("message and geometry cannot be null");
        }
        byte[] prepared = xorPrepared(message.preparedBytes, xorKey);
        byte[] nonce = new byte[NONCE_LEN];
        RNG.nextBytes(nonce);
        int cipherLen = prepared.length;
        requireSupportedCiphertextLength(cipherLen);

        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(
                Cipher.ENCRYPT_MODE,
                new SecretKeySpec(key(encryptionKey), "AES"),
                new GCMParameterSpec(TAG_LEN * 8, nonce));
        cipher.updateAAD(aadForGeometry(qf, cipherLen, xorKey, geometry));
        byte[] encryptedWithTag = cipher.doFinal(prepared);
        int observedCipherLen = encryptedWithTag.length - TAG_LEN;
        if (observedCipherLen != cipherLen) {
            throw new IllegalStateException(
                    "AES-GCM ciphertext length differs from prepared plaintext length");
        }
        byte[] cipherText = Arrays.copyOfRange(encryptedWithTag, 0, cipherLen);
        byte[] tag = Arrays.copyOfRange(
                encryptedWithTag, cipherLen, encryptedWithTag.length);
        int packedLength = Math.addExact(headerBytes(), cipherLen);
        ByteBuffer bb = ByteBuffer.allocate(packedLength)
                .order(ByteOrder.BIG_ENDIAN);
        putUnsignedShort(bb, cipherLen);
        bb.put(nonce);
        bb.put(tag);
        bb.put(cipherText);
        byte[] packed = bb.array();
        PayloadLayout observed = message.layout.withPackedLength(
                packed.length, cipherLen);
        return new PackResult(packed, observed);
    }

    /** Compact geometry-bound production packing API. */
    public static byte[] pack(
            byte[] message, String encodedKeys, int qf, SamplingGeometry geometry
    ) throws GeneralSecurityException {
        return packDetailed(message, encodedKeys, qf, geometry).packed();
    }

    /** Compact packing API using the strict encoded three-key tuple. */
    public static byte[] pack(byte[] message, String encodedKeys, int qf) throws GeneralSecurityException {
        return packDetailed(message, encodedKeys, qf).packed();
    }

    
    public static byte[] unpack(
            byte[] packed, String encodedKeys, int qf, SamplingGeometry geometry
    ) throws GeneralSecurityException {
        if (geometry == null) throw new IllegalArgumentException("geometry cannot be null");
        KeyMaterial keys = KeyMaterial.decode(encodedKeys);
        return unpackGeometryBound(
                packed, keys.encryptionKey(), keys.xorKey(), qf, geometry);
    }

    /** Non-geometry compact unpacking with explicit trial QF. */
    public static byte[] unpack(
            byte[] packed, String encodedKeys, int qf
    ) throws GeneralSecurityException {
        KeyMaterial keys = KeyMaterial.decode(encodedKeys);
        return unpack(packed, keys.encryptionKey(), keys.xorKey(), qf);
    }

    /**
     * Compatibility helper for non-production callers that do not already know the QF.
     * The production extractor must use the explicit-QF overload because it already
     * owns the adaptive-QF trial context.
     */
    public static byte[] unpack(byte[] packed, String encodedKeys) throws GeneralSecurityException {
        KeyMaterial keys = KeyMaterial.decode(encodedKeys);
        return unpackAutoQf(packed, keys.encryptionKey(), keys.xorKey());
    }

    /** Compatibility helper for geometry-bound callers outside the adaptive extractor. */
    public static byte[] unpack(
            byte[] packed, String encodedKeys, SamplingGeometry geometry
    ) throws GeneralSecurityException {
        if (geometry == null) throw new IllegalArgumentException("geometry cannot be null");
        KeyMaterial keys = KeyMaterial.decode(encodedKeys);
        GeneralSecurityException last = null;
        for (int qf = MIN_SUPPORTED_QF; qf <= MAX_SUPPORTED_QF; qf++) {
            try {
                return unpackGeometryBound(
                        packed, keys.encryptionKey(), keys.xorKey(), qf, geometry);
            } catch (GeneralSecurityException rejected) {
                last = rejected;
            }
        }
        if (last != null) throw last;
        throw new GeneralSecurityException("No supported QF authenticated the payload");
    }

    public static byte[] unpack(
            byte[] packed, String encryptionKey, String xorKey, int qf
    ) throws GeneralSecurityException {
        validateQf(qf);
        ParsedHeader parsed = parseHeader(packed);
        int cipherLen = parsed.cipherLen();
        byte[] encryptedWithTag = new byte[Math.addExact(cipherLen, TAG_LEN)];
        System.arraycopy(parsed.cipherText(), 0, encryptedWithTag, 0, cipherLen);
        System.arraycopy(parsed.tag(), 0, encryptedWithTag, cipherLen, TAG_LEN);

        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(
                Cipher.DECRYPT_MODE,
                new SecretKeySpec(key(encryptionKey), "AES"),
                new GCMParameterSpec(TAG_LEN * 8, parsed.nonce())
        );
        cipher.updateAAD(aadFor(qf, cipherLen, xorKey));

        byte[] prepared = cipher.doFinal(encryptedWithTag);
        xorPreparedInPlace(prepared, xorKey);
        return recoverPlainPayload(prepared);
    }

    
    public static byte[] unpack(
            byte[] packed, String encryptionKey, String xorKey
    ) throws GeneralSecurityException {
        return unpackAutoQf(packed, encryptionKey, xorKey);
    }

    private static byte[] unpackAutoQf(
            byte[] packed, String encryptionKey, String xorKey
    ) throws GeneralSecurityException {
        GeneralSecurityException last = null;
        for (int qf = MIN_SUPPORTED_QF; qf <= MAX_SUPPORTED_QF; qf++) {
            try {
                return unpack(packed, encryptionKey, xorKey, qf);
            } catch (GeneralSecurityException rejected) {
                last = rejected;
            }
        }
        if (last != null) throw last;
        throw new GeneralSecurityException("No supported QF authenticated the payload");
    }

    private static byte[] unpackGeometryBound(
            byte[] packed, String encryptionKey, String xorKey, int qf,
            SamplingGeometry geometry
    ) throws GeneralSecurityException {
        validateQf(qf);
        if (geometry == null) throw new IllegalArgumentException("geometry cannot be null");
        ParsedHeader parsed = parseHeader(packed);
        int cipherLen = parsed.cipherLen();
        byte[] encryptedWithTag = new byte[Math.addExact(cipherLen, TAG_LEN)];
        System.arraycopy(parsed.cipherText(), 0, encryptedWithTag, 0, cipherLen);
        System.arraycopy(parsed.tag(), 0, encryptedWithTag, cipherLen, TAG_LEN);

        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(
                Cipher.DECRYPT_MODE,
                new SecretKeySpec(key(encryptionKey), "AES"),
                new GCMParameterSpec(TAG_LEN * 8, parsed.nonce()));
        cipher.updateAAD(aadForGeometry(qf, cipherLen, xorKey, geometry));
        byte[] prepared = cipher.doFinal(encryptedWithTag);
        xorPreparedInPlace(prepared, xorKey);
        return recoverPlainPayload(prepared);
    }

    
    @Deprecated
    public static int qfFromHeader(byte[] header) {
        throw new UnsupportedOperationException(
                "RAP-UNIWARD compact header does not embed QF");
    }

    public static int ciphertextLengthFromHeader(byte[] header) {
        if (header == null || header.length < CIPHERTEXT_LENGTH_FIELD_LEN) {
            throw new IllegalArgumentException("Header too short for ciphertext length.");
        }
        return Short.toUnsignedInt(
                ByteBuffer.wrap(header, 0, CIPHERTEXT_LENGTH_FIELD_LEN)
                        .order(ByteOrder.BIG_ENDIAN).getShort());
    }

    private static ParsedHeader parseHeader(byte[] packed) {
        if (packed == null || packed.length < headerBytes()) {
            throw new IllegalStateException(
                    "Packed payload is shorter than the RAP-UNIWARD compact fixed header.");
        }
        ByteBuffer bb = ByteBuffer.wrap(packed).order(ByteOrder.BIG_ENDIAN);
        int cipherLen = Short.toUnsignedInt(bb.getShort());
        requireSupportedCiphertextLength(cipherLen);
        byte[] nonce = new byte[NONCE_LEN];
        bb.get(nonce);
        byte[] tag = new byte[TAG_LEN];
        bb.get(tag);
        if (bb.remaining() != cipherLen) {
            throw new IllegalStateException(
                    "Packed payload length does not match compact ciphertext length field.");
        }
        byte[] cipherText = new byte[cipherLen];
        bb.get(cipherText);
        return new ParsedHeader(cipherLen, nonce, tag, cipherText);
    }

    private static void putUnsignedShort(ByteBuffer bb, int value) {
        requireSupportedCiphertextLength(value);
        bb.putShort((short) (value & 0xFFFF));
    }

    private static void requireSupportedCiphertextLength(int cipherLen) {
        if (cipherLen < PREPROCESSING_METADATA_LEN
                || cipherLen > MAX_CIPHERTEXT_LEN) {
            throw new IllegalArgumentException(
                    "RAP-UNIWARD compact ciphertext length must be in "
                            + PREPROCESSING_METADATA_LEN + "-"
                            + MAX_CIPHERTEXT_LEN + " bytes: " + cipherLen);
        }
    }

    private static void validateQf(int qf) {
        if (qf < MIN_SUPPORTED_QF || qf > MAX_SUPPORTED_QF) {
            throw new IllegalArgumentException("QF must be in " + MIN_SUPPORTED_QF + "-" + MAX_SUPPORTED_QF + ": " + qf);
        }
    }

    private static byte[] aadFor(
            int qf, int cipherLen, String independentXorKey
    ) throws GeneralSecurityException {
        if (independentXorKey == null) {
            throw new IllegalArgumentException("XOR key cannot be null");
        }
        byte[] commitment = MessageDigest.getInstance("SHA-256")
                .digest(("xor-aad:" + independentXorKey)
                        .getBytes(StandardCharsets.UTF_8));
        return ByteBuffer.allocate(AAD_LEN).order(ByteOrder.BIG_ENDIAN)
                .put(AAD_DOMAIN)
                .put((byte) qf)
                .putInt(cipherLen)
                .put(commitment)
                .array();
    }

    private static byte[] aadForGeometry(
            int qf, int cipherLen, String independentXorKey,
            SamplingGeometry geometry
    ) throws GeneralSecurityException {
        if (independentXorKey == null || geometry == null) {
            throw new IllegalArgumentException("XOR key and geometry cannot be null");
        }
        byte[] commitment = MessageDigest.getInstance("SHA-256")
                .digest(("xor-aad:" + independentXorKey)
                        .getBytes(StandardCharsets.UTF_8));
        return ByteBuffer.allocate(GEOMETRY_AAD_LEN)
                .order(ByteOrder.BIG_ENDIAN)
                .put(GEOMETRY_AAD_DOMAIN)
                .put((byte) qf)
                .putInt(geometry.signatureCode())
                .putInt(cipherLen)
                .put(commitment)
                .array();
    }

    private static PreparedPayload preparePlainPayload(byte[] message) {
        byte[] huff = HuffmanCodec.encode(message);
        boolean useHuffman = huff.length < message.length;
        byte mode = useHuffman ? (byte) 1 : (byte) 0;
        byte[] content = useHuffman ? huff : message;

        ByteBuffer meta = ByteBuffer.allocate(PREPROCESSING_METADATA_LEN).order(ByteOrder.BIG_ENDIAN);
        meta.put(mode);
        meta.putInt(message.length);
        meta.putInt(content.length);

        int preparedLength = Math.addExact(
                PREPROCESSING_METADATA_LEN, content.length);
        ByteArrayOutputStream out =
                new ByteArrayOutputStream(preparedLength);
        out.writeBytes(meta.array());
        out.writeBytes(content);
        byte[] prepared = out.toByteArray();
        requireSupportedCiphertextLength(prepared.length);

        PayloadLayout layout = new PayloadLayout(
                message.length,
                huff.length,
                useHuffman,
                content.length,
                PREPROCESSING_METADATA_LEN,
                prepared.length,
                PUBLIC_MARKER_LEN,
                QF_LEN,
                CIPHERTEXT_LENGTH_FIELD_LEN,
                NONCE_LEN,
                TAG_LEN,
                headerBytes(),
                prepared.length,
                Math.addExact(headerBytes(), prepared.length),
                Math.multiplyExact(
                        Math.addExact(
                                headerBytes(), prepared.length),
                        8)
        );
        return new PreparedPayload(prepared, layout);
    }

    private static byte[] xorPrepared(byte[] prepared, String independentXorKey) throws GeneralSecurityException {
        byte[] copy = Arrays.copyOf(prepared, prepared.length);
        xorPreparedInPlace(copy, independentXorKey);
        return copy;
    }

    private static void xorPreparedInPlace(byte[] prepared, String independentXorKey) throws GeneralSecurityException {
        byte[] xorKeyBytes = MessageDigest.getInstance("SHA-256")
                .digest(("xor:" + independentXorKey).getBytes(StandardCharsets.UTF_8));
        for (int i = 0; i < prepared.length; i++) prepared[i] = (byte) (prepared[i] ^ xorKeyBytes[i % xorKeyBytes.length]);
    }

    private static byte[] recoverPlainPayload(byte[] prepared) {
        if (prepared.length < PREPROCESSING_METADATA_LEN) {
            throw new IllegalStateException("Recovered prepared payload is shorter than preprocessing metadata.");
        }
        ByteBuffer bb = ByteBuffer.wrap(prepared).order(ByteOrder.BIG_ENDIAN);
        byte mode = bb.get();
        int rawLength = bb.getInt();
        int contentLength = bb.getInt();
        if (rawLength < 0 || contentLength < 0 || contentLength > bb.remaining()) {
            throw new IllegalStateException("Invalid recovered payload lengths.");
        }

        byte[] content = new byte[contentLength];
        bb.get(content);
        if (bb.hasRemaining()) {
            throw new IllegalStateException("Recovered prepared payload has unexpected trailing bytes.");
        }

        byte[] plain;
        if (mode == 1) plain = HuffmanCodec.decode(content);
        else if (mode == 0) plain = content;
        else throw new IllegalStateException("Unknown payload preprocessing mode: " + mode);

        if (plain.length != rawLength) {
            throw new IllegalStateException("Recovered message length mismatch. expected=" + rawLength + ", got=" + plain.length);
        }
        return plain;
    }

    private static byte[] key(String independentEncryptionKey) throws GeneralSecurityException {
        if (independentEncryptionKey == null) throw new IllegalArgumentException("encryption key cannot be null");
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        return Arrays.copyOf(md.digest(("aes-gcm:" + independentEncryptionKey)
                .getBytes(StandardCharsets.UTF_8)), 16);
    }

    private record ParsedHeader(
            int cipherLen, byte[] nonce, byte[] tag, byte[] cipherText
    ) {
        ParsedHeader {
            nonce = Arrays.copyOf(nonce, nonce.length);
            tag = Arrays.copyOf(tag, tag.length);
            cipherText = Arrays.copyOf(cipherText, cipherText.length);
        }
        @Override public byte[] nonce() { return Arrays.copyOf(nonce, nonce.length); }
        @Override public byte[] tag() { return Arrays.copyOf(tag, tag.length); }
        @Override public byte[] cipherText() { return Arrays.copyOf(cipherText, cipherText.length); }
    }

    private record PreparedPayload(byte[] bytes, PayloadLayout layout) {}

    /** Immutable run-scoped result of QF-independent payload preprocessing. */
    public record PreparedMessage(byte[] preparedBytes, PayloadLayout layout) {
        public PreparedMessage {
            if (preparedBytes == null || layout == null) {
                throw new IllegalArgumentException("prepared message fields cannot be null");
            }
            preparedBytes = Arrays.copyOf(preparedBytes, preparedBytes.length);
        }

        @Override public byte[] preparedBytes() {
            return Arrays.copyOf(preparedBytes, preparedBytes.length);
        }
    }


    public record PackResult(byte[] packed, PayloadLayout layout) {
        public PackResult {
            packed = Arrays.copyOf(packed, packed.length);
        }
        @Override public byte[] packed() { return Arrays.copyOf(packed, packed.length); }
    }

    /** Exact byte accounting for one payload format instance. */
    public record PayloadLayout(
            int rawMessageBytes,
            int huffmanCandidateBytes,
            boolean huffmanUsed,
            int contentBytes,
            int preprocessingMetadataBytes,
            int preparedPlaintextBytes,
            int protocolMarkerBytes,
            int qfFieldBytes,
            int ciphertextLengthFieldBytes,
            int nonceBytes,
            int authenticationTagBytes,
            int fixedHeaderBytes,
            int ciphertextBytes,
            int totalPayloadBytes,
            int totalPayloadBits
    ) {
        private PayloadLayout withPackedLength(int packedBytes, int observedCiphertextBytes) {
            if (observedCiphertextBytes != preparedPlaintextBytes) {
                throw new IllegalStateException("AES-GCM ciphertext length unexpectedly differs from prepared plaintext length.");
            }
            return new PayloadLayout(rawMessageBytes, huffmanCandidateBytes, huffmanUsed, contentBytes,
                    preprocessingMetadataBytes, preparedPlaintextBytes, protocolMarkerBytes, qfFieldBytes,
                    ciphertextLengthFieldBytes, nonceBytes, authenticationTagBytes, fixedHeaderBytes,
                    observedCiphertextBytes, packedBytes, Math.multiplyExact(packedBytes, 8));
        }

        /** Header plus encrypted preprocessing metadata; independent of compression. */
        public int fixedFramingOverheadBytes() {
            return Math.addExact(
                    fixedHeaderBytes, preprocessingMetadataBytes);
        }

        /** Net payload growth relative to the raw message; may be negative after compression. */
        public int netPayloadExpansionBytes() {
            return totalPayloadBytes - rawMessageBytes;
        }

        public double expansionRatio() { return rawMessageBytes == 0 ? Double.NaN : totalPayloadBytes / (double) rawMessageBytes; }
        public double compressionRatio() { return rawMessageBytes == 0 ? Double.NaN : contentBytes / (double) rawMessageBytes; }
    }
}
