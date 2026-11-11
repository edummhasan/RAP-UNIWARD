package thesis.rap.stc;

/** Expected infeasibility under fixed wet constraints, distinct from native failures. */
public final class NoFiniteStcSolutionException extends IllegalStateException {
    private static final long serialVersionUID = 1L;
    public NoFiniteStcSolutionException(String message) { super(message); }
}
