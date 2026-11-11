package thesis.rap.app;

import java.util.Map;

/** Ensures height A/B profiles hold the declared carrier counts constant. */
public final class StcHeightProfileSelfTest {
    private StcHeightProfileSelfTest() {}

    public static void main(String[] args) {
        int headerBits = thesis.rap.payload.PayloadPackage.headerBytes() * 8;
        int bodyBits = 64;
        int candidates = 100_000;
        StcCarrierPlan h10 = StcCarrierPlan.createForHeight(
                candidates, headerBits, bodyBits, 10);
        StcCarrierPlan h12 = StcCarrierPlan.createForHeight(
                candidates, headerBits, bodyBits, 12);

        assert h10.bodyWidthCap() == h12.bodyWidthCap();
        assert h10.headerCarrierCount() == h12.headerCarrierCount();
        assert h10.bodyCarrierCount() == h12.bodyCarrierCount();
        assert h10.totalCarrierCount() == h12.totalCarrierCount();
        assert h12.nativePathBytesEstimate()
                == 4L * h10.nativePathBytesEstimate();
        assert h12.nativePeakPathBytesEstimate()
                == 4L * h10.nativePeakPathBytesEstimate();
        assert h10.nativePeakPathBytesEstimate()
                <= h10.nativePathBytesEstimate();
        assert h10.nativeMetricBytesEstimate() == 16_384L;
        assert h12.nativeMetricBytesEstimate() == 65_536L;
        assert h12.nativeMetricBytesEstimate()
                == 4L * h10.nativeMetricBytesEstimate();
        assert RapUniward.embeddingStcHeight(Map.of()) == 10;
        assert RapUniward.embeddingStcHeight(
                Map.of("--stcHeight", "12")) == 12;
        int[] automatic = RapUniward.extractionStcHeights(
                Map.of("--stcHeight", "auto"));
        assert automatic.length == 2;
        assert automatic[0] == 10 && automatic[1] == 12;

        boolean rejected = false;
        try {
            RapUniward.embeddingStcHeight(
                    Map.of("--stcHeight", "14"));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected : "unbounded research heights must fail closed";
        System.out.println(
                "StcHeightProfileSelfTest: ALL TESTS PASSED");
    }
}
