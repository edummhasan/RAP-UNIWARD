package thesis.rap.image;

import java.io.Serial;

/**
 * Identifies a canonical reference that exceeds the baseline JPEG AC bound at
 * one geometry/QF trial.  This type distinguishes trial infeasibility from I/O
 * and internal-consistency failures.
 *
 * <p>For a fixed image and geometry, an AC overflow at one QF cannot disappear
 * at a later, higher QF under the declared monotone quantization tables.  The
 * adaptive controller may still record later trials uniformly, but it must not
 * claim that a higher QF repairs this particular condition.</p>
 */
public final class UnencodableReferenceException extends RuntimeException {
    @Serial
    private static final long serialVersionUID = 1L;

    private final int qf;
    private final SamplingGeometry geometry;

    public UnencodableReferenceException(
            int qf,
            SamplingGeometry geometry,
            String message
    ) {
        super(message);
        this.qf = qf;
        this.geometry = geometry;
    }

    public int qf() {
        return qf;
    }

    public SamplingGeometry geometry() {
        return geometry;
    }
}
