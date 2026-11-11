package thesis.rap.app;

import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/** Dependency-free integrity test for versioned batch resume metadata. */
public final class VerifiedOutputManifestSelfTest {
    private VerifiedOutputManifestSelfTest() {}

    public static void main(String[] args) throws Exception {
        Path dir = Files.createTempDirectory("rap-resume-manifest-");
        Path png = dir.resolve("cover.png");
        Path message = dir.resolve("message.msg");
        Path stego = dir.resolve("stego.jpg");
        Files.write(png, new byte[]{1, 2, 3});
        Files.write(message, new byte[]{4, 5, 6});
        Files.write(stego, new byte[]{7, 8, 9});
        VerifiedOutputManifest.write(
                stego, png, message, 75, 100, "1.0.0.1", 10,
                SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_444);
        assert VerifiedOutputManifest.matches(
                stego, png, message, 75, 100, "1.0.0.1", 10,
                SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_444);
        String manifest = Files.readString(
                VerifiedOutputManifest.sidecarFor(stego), StandardCharsets.UTF_8);
        assert manifest.contains("format=RAP-VERIFIED-OUTPUT-1\n");
        String profileLine = "balance_profile="
                + ThreeStateEmbeddingEngine.Profile.configured().code() + "\n";
        assert manifest.contains(profileLine);
        assert manifest.contains("balance_zero_budget=");
        assert manifest.contains("balance_max_attempts=");
        assert manifest.contains(
                "stc_numeric_profile=PORTABLE_SCALAR_BINARY64_STC_V1\n");
        assert manifest.contains("geometry_mode=ALL_GEOMETRY\n");
        assert manifest.contains("sampling_geometry=4:4:4\n");
        assert manifest.contains("geometry_signature_hex=111111\n");
        assert manifest.contains("carrier_order_domain=ahp-order-v1\n");
        assert manifest.contains("sampling_factors=Y=1x1;Cb=1x1;Cr=1x1\n");
        assert manifest.contains("reference_template_writer=ImageIO JPEG writer class=");
        assert manifest.contains(", provider=");
        assert manifest.contains(", vendor=");
        assert manifest.contains(", version=");
        assert !VerifiedOutputManifest.matches(
                stego, png, message, 75, 100, "wrong-version", 10,
                SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_444);
        assert !VerifiedOutputManifest.matches(
                stego, png, message, 75, 100, "1.0.0.1", 10,
                SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_420);
        // A sidecar from a different balancing profile must not authorize resume.
        String otherProfile = ThreeStateEmbeddingEngine.Profile.configured()
                == ThreeStateEmbeddingEngine.Profile.STRICT ? "U" : "S";
        for (String incompatible : new String[]{
                manifest.replace(profileLine, "balance_profile=" + otherProfile + "\n"),
                manifest.replace(profileLine, ""),
                manifest.replace("RAP-VERIFIED-OUTPUT-1", "RAP-VERIFIED-OUTPUT-7")}) {
            Files.writeString(VerifiedOutputManifest.sidecarFor(stego), incompatible);
            assert !VerifiedOutputManifest.matches(
                    stego, png, message, 75, 100, "1.0.0.1", 10,
                    SamplingGeometryMode.ALL_GEOMATRY, SamplingGeometry.JPEG_444);
        }
        Files.writeString(VerifiedOutputManifest.sidecarFor(stego), manifest);
        Files.write(message, new byte[]{4, 5, 0});
        assert !VerifiedOutputManifest.matches(
                stego, png, message, 75, 100, "1.0.0.1", 10,
                SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_444);
        System.out.println("VerifiedOutputManifestSelfTest: ALL TESTS PASSED");
    }
}
