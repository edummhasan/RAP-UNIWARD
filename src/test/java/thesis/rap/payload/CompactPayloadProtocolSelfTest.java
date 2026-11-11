package thesis.rap.payload;

import thesis.rap.image.SamplingGeometry;
import javax.crypto.AEADBadTagException;
import java.util.Arrays;
import java.util.Random;


public final class CompactPayloadProtocolSelfTest {
    private CompactPayloadProtocolSelfTest() {}

    public static void main(String[] args) throws Exception {
        assert PayloadPackage.headerBytes() == 26;
        assert PayloadPackage.QF_LEN == 0;
        assert PayloadPackage.CIPHERTEXT_LENGTH_FIELD_LEN == 2;
        assert PayloadPackage.NONCE_LEN == 12;
        assert PayloadPackage.TAG_LEN == 12;
        assert PayloadPackage.GEOMETRY_AAD_LEN == 72;

        String keys = new KeyMaterial(
                "enc-standalone-compact", "xor-standalone-compact",
                "stc-standalone-compact").encode();
        byte[] message = "compact authenticated payload".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        int qf = 83;
        SamplingGeometry geometry = SamplingGeometry.JPEG_420;
        PayloadPackage.PackResult packed = PayloadPackage.packDetailed(message, keys, qf, geometry);
        byte[] header = Arrays.copyOfRange(packed.packed(), 0, PayloadPackage.headerBytes());
        assert PayloadPackage.ciphertextLengthFromHeader(header)
                == packed.layout().preparedPlaintextBytes();
        assert Arrays.equals(message, PayloadPackage.unpack(packed.packed(), keys, qf, geometry));

        boolean rejected = false;
        try { PayloadPackage.unpack(packed.packed(), keys, qf + 1, geometry); }
        catch (AEADBadTagException expected) { rejected = true; }
        assert rejected : "wrong trial QF must fail AAD authentication";

        rejected = false;
        try { PayloadPackage.unpack(packed.packed(), keys, qf, SamplingGeometry.JPEG_444); }
        catch (AEADBadTagException expected) { rejected = true; }
        assert rejected : "wrong geometry must fail AAD authentication";

        byte[] oversized = new byte[70_000];
        new Random(7L).nextBytes(oversized);
        rejected = false;
        try { PayloadPackage.inspectLayout(oversized); }
        catch (IllegalArgumentException expected) { rejected = true; }
        assert rejected : "prepared ciphertext beyond uint16 range must fail closed";

        System.out.println("CompactPayloadProtocolSelfTest: ALL TESTS PASSED");
    }
}
