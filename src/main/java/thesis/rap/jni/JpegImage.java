package thesis.rap.jni;

import java.util.Arrays;
import java.util.Objects;

public class JpegImage {
    public final int width;
    public final int height;
    public final JpegComponent[] components;

    public JpegImage(int width, int height, JpegComponent[] components) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException(
                    "JPEG image dimensions must be positive");
        }
        Objects.requireNonNull(components, "components");
        if (components.length == 0) {
            throw new IllegalArgumentException(
                    "JPEG image must contain at least one component");
        }
        if (Arrays.stream(components).anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException(
                    "JPEG components cannot contain null");
        }
        this.width = width;
        this.height = height;
        this.components = components;
    }

    public JpegComponent component(int index) { return components[index]; }
}
