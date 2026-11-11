package thesis.rap.image;

import java.util.List;

/** Public geometry selector for the complete color-subsampling study. */
public enum SamplingGeometryMode {
    ALL_GEOMATRY(List.of(SamplingGeometry.values())),
    GEOMETRY_444(List.of(SamplingGeometry.JPEG_444)),
    GEOMETRY_422(List.of(SamplingGeometry.JPEG_422)),
    GEOMETRY_420(List.of(SamplingGeometry.JPEG_420)),
    GEOMETRY_440(List.of(SamplingGeometry.JPEG_440)),
    GEOMETRY_411(List.of(SamplingGeometry.JPEG_411)),
    GEOMETRY_441(List.of(SamplingGeometry.JPEG_441)),
    GEOMETRY_410(List.of(SamplingGeometry.JPEG_410));

    private final List<SamplingGeometry> geometries;
    SamplingGeometryMode(List<SamplingGeometry> geometries) { this.geometries = List.copyOf(geometries); }
    public List<SamplingGeometry> geometries() { return geometries; }
    public boolean includesAll() { return this == ALL_GEOMATRY; }
    public String token() {
        return switch (this) {
            case ALL_GEOMATRY -> "ALL_GEOMETRY";
            case GEOMETRY_444 -> "444GEOMETRY";
            case GEOMETRY_422 -> "422GEOMETRY";
            case GEOMETRY_420 -> "420GEOMETRY";
            case GEOMETRY_440 -> "440GEOMETRY";
            case GEOMETRY_411 -> "411GEOMETRY";
            case GEOMETRY_441 -> "441GEOMETRY";
            case GEOMETRY_410 -> "410GEOMETRY";
        };
    }
    public static SamplingGeometryMode parse(String raw) {
        if (raw == null || raw.isBlank()) return ALL_GEOMATRY;
        return switch (raw.trim().toUpperCase(java.util.Locale.ROOT)) {
            case "ALL_GEOMATRY", "ALL_GEOMETRY" -> ALL_GEOMATRY;
            case "444GEOMATRY", "444GEOMETRY" -> GEOMETRY_444;
            case "422GEOMATRY", "422GEOMETRY" -> GEOMETRY_422;
            case "420GEOMATRY", "420GEOMETRY" -> GEOMETRY_420;
            case "440GEOMATRY", "440GEOMETRY" -> GEOMETRY_440;
            case "411GEOMATRY", "411GEOMETRY" -> GEOMETRY_411;
            case "441GEOMATRY", "441GEOMETRY" -> GEOMETRY_441;
            case "410GEOMATRY", "410GEOMETRY" -> GEOMETRY_410;
            default -> throw new IllegalArgumentException(
                    "--geometryMode must be ALL_GEOMATRY or one of " +
                    "444/422/420/440/411/441/410 GEOMATRY: " + raw);
        };
    }
}
