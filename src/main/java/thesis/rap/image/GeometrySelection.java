package thesis.rap.image;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Immutable CLI-level geometry selection supporting one, several, or all of
 * the seven thesis sampling geometries. Internal embedding trials still run
 * against exactly one concrete {@link SamplingGeometry} at a time.
 */
public final class GeometrySelection {
    private final List<SamplingGeometry> geometries;
    private final String token;

    private GeometrySelection(List<SamplingGeometry> geometries, String token) {
        if (geometries == null || geometries.isEmpty()) {
            throw new IllegalArgumentException("At least one geometry is required");
        }
        this.geometries = List.copyOf(geometries);
        this.token = token;
    }

    public static GeometrySelection all() {
        return new GeometrySelection(List.of(SamplingGeometry.values()), "ALL_GEOMETRY");
    }

    public static GeometrySelection of(SamplingGeometry geometry) {
        if (geometry == null) throw new IllegalArgumentException("geometry cannot be null");
        return new GeometrySelection(List.of(geometry), geometry.modeToken());
    }

    public static GeometrySelection parse(String raw) {
        if (raw == null || raw.isBlank()) return all();
        String normalized = raw.trim();
        String upper = normalized.toUpperCase(Locale.ROOT);
        if (upper.equals("ALL_GEOMETRY") || upper.equals("ALL_GEOMATRY") || upper.equals("ALL")) {
            return all();
        }

        String[] parts = normalized.split(",");
        Set<SamplingGeometry> ordered = new LinkedHashSet<>();
        for (String part : parts) {
            String p = part.trim().toUpperCase(Locale.ROOT)
                    .replace(" ", "")
                    .replace(":", "")
                    .replace("_", "");
            if (p.isEmpty()) continue;
            SamplingGeometry g = switch (p) {
                case "444", "444GEOMETRY", "444GEOMATRY", "G444" -> SamplingGeometry.JPEG_444;
                case "422", "422GEOMETRY", "422GEOMATRY", "G422" -> SamplingGeometry.JPEG_422;
                case "420", "420GEOMETRY", "420GEOMATRY", "G420" -> SamplingGeometry.JPEG_420;
                case "440", "440GEOMETRY", "440GEOMATRY", "G440" -> SamplingGeometry.JPEG_440;
                case "411", "411GEOMETRY", "411GEOMATRY", "G411" -> SamplingGeometry.JPEG_411;
                case "441", "441GEOMETRY", "441GEOMATRY", "G441" -> SamplingGeometry.JPEG_441;
                case "410", "410GEOMETRY", "410GEOMATRY", "G410" -> SamplingGeometry.JPEG_410;
                default -> throw new IllegalArgumentException(
                        "Unknown geometry '" + part + "'. Expected ALL or a comma-separated subset of "
                                + "444,422,420,440,411,441,410");
            };
            ordered.add(g);
        }
        if (ordered.isEmpty()) {
            throw new IllegalArgumentException("At least one geometry must be selected");
        }
        List<SamplingGeometry> selected = new ArrayList<>(ordered);
        String token = selected.size() == 1
                ? selected.get(0).modeToken()
                : "MULTI_GEOMETRY[" + selected.stream()
                .map(g -> g.label().replace(":", ""))
                .reduce((a, b) -> a + "," + b).orElse("") + "]";
        return new GeometrySelection(selected, token);
    }

    public List<SamplingGeometry> geometries() { return geometries; }
    public String token() { return token; }
    public boolean includesAll() { return geometries.size() == SamplingGeometry.values().length; }
    public boolean isMultiple() { return geometries.size() > 1; }
    public boolean contains(SamplingGeometry geometry) { return geometries.contains(geometry); }

    public SamplingGeometryMode singleModeFor(SamplingGeometry geometry) {
        if (!contains(geometry)) {
            throw new IllegalArgumentException("Geometry " + geometry + " is not selected by " + token);
        }
        return switch (geometry) {
            case JPEG_444 -> SamplingGeometryMode.GEOMETRY_444;
            case JPEG_422 -> SamplingGeometryMode.GEOMETRY_422;
            case JPEG_420 -> SamplingGeometryMode.GEOMETRY_420;
            case JPEG_440 -> SamplingGeometryMode.GEOMETRY_440;
            case JPEG_411 -> SamplingGeometryMode.GEOMETRY_411;
            case JPEG_441 -> SamplingGeometryMode.GEOMETRY_441;
            case JPEG_410 -> SamplingGeometryMode.GEOMETRY_410;
        };
    }
}
