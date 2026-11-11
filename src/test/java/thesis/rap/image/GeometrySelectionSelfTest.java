package thesis.rap.image;

import java.util.List;


public final class GeometrySelectionSelfTest {
    private GeometrySelectionSelfTest() {}

    public static void main(String[] args) {
        GeometrySelection single = GeometrySelection.parse("420");
        assert single.geometries().equals(List.of(SamplingGeometry.JPEG_420));
        assert !single.isMultiple();

        GeometrySelection subset = GeometrySelection.parse("444,420,411,410");
        assert subset.geometries().equals(List.of(
                SamplingGeometry.JPEG_444,
                SamplingGeometry.JPEG_420,
                SamplingGeometry.JPEG_411,
                SamplingGeometry.JPEG_410));
        assert subset.isMultiple();
        assert !subset.includesAll();

        GeometrySelection labels = GeometrySelection.parse("4:4:0,4:4:1");
        assert labels.geometries().equals(List.of(
                SamplingGeometry.JPEG_440,
                SamplingGeometry.JPEG_441));

        GeometrySelection all = GeometrySelection.parse("ALL_GEOMETRY");
        assert all.includesAll();
        assert all.geometries().equals(List.of(SamplingGeometry.values()));

        boolean rejected = false;
        try { GeometrySelection.parse("424"); }
        catch (IllegalArgumentException expected) { rejected = true; }
        assert rejected : "Undeclared geometry must be rejected";

        System.out.println("GeometrySelectionSelfTest: ALL TESTS PASSED");
    }
}
