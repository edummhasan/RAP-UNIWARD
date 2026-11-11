package thesis.rap.app;

import thesis.rap.alg.CandidateGenerator;
import thesis.rap.alg.CandidateSet;
import thesis.rap.image.CanonicalReferenceJpeg;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SourceImageContext;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegImage;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;


public final class TrueParallelStateCache {
    private static final String ENABLE_PROPERTY = "rap.true.parallel.cache";
    private static final String MAX_STATES_PROPERTY = "rap.true.parallel.cache.states";
    private static final String MAX_SOURCE_IMAGES_PROPERTY = "rap.true.parallel.source.images";
    private static final int DEFAULT_MAX_STATES = 32;
    private static final int DEFAULT_MAX_SOURCE_IMAGES = 2;

    private static final Object LOCK = new Object();
    private static final LinkedHashMap<Key, CompletableFuture<State>> CACHE =
            new LinkedHashMap<>(64, 0.75f, true);
    private static final LinkedHashMap<SourceKey, SourceImageContext> SOURCE_CACHE =
            new LinkedHashMap<>(8, 0.75f, true);
    private static volatile Path cacheDirectory;
    private static volatile boolean cleanupHookInstalled;
    private static final ExecutorService PREFETCH_EXECUTOR =
            Executors.newFixedThreadPool(
                    prefetchWorkers(), new PrefetchThreadFactory());

    private TrueParallelStateCache() {}

    public static boolean enabled() {
        String property = System.getProperty(ENABLE_PROPERTY);
        String env = System.getenv("RAP_TRUE_PARALLEL_CACHE");
        String value = property != null ? property : (env == null ? "true" : env);
        return Boolean.parseBoolean(value);
    }

    public static int maximumStates() {
        String raw = System.getProperty(MAX_STATES_PROPERTY,
                System.getenv().getOrDefault("RAP_PREPARED_CACHE_STATES",
                        Integer.toString(DEFAULT_MAX_STATES)));
        int parsed;
        try {
            parsed = Integer.parseInt(raw.trim());
        } catch (NumberFormatException bad) {
            throw new IllegalArgumentException(
                    "RAP_PREPARED_CACHE_STATES must be an integer", bad);
        }
        if (parsed < 1 || parsed > 4096) {
            throw new IllegalArgumentException(
                    "RAP_PREPARED_CACHE_STATES must be in 1..4096: " + parsed);
        }
        return parsed;
    }

    public static int prefetchDepth() {
        String raw = System.getProperty(
                "rap.true.parallel.qf.prefetch.depth",
                System.getenv().getOrDefault(
                        "RAP_QF_PREFETCH_DEPTH", "4"));
        int parsed;
        try {
            parsed = Integer.parseInt(raw.trim());
        } catch (NumberFormatException bad) {
            throw new IllegalArgumentException(
                    "RAP_QF_PREFETCH_DEPTH must be an integer", bad);
        }
        if (parsed < 0 || parsed > 26) {
            throw new IllegalArgumentException(
                    "RAP_QF_PREFETCH_DEPTH must be in 0..26: " + parsed);
        }
        return parsed;
    }

    private static int prefetchWorkers() {
        String raw = System.getProperty(
                "rap.true.parallel.qf.prefetch.workers",
                System.getenv().getOrDefault(
                        "RAP_QF_PREFETCH_WORKERS", "4"));
        try {
            int parsed = Integer.parseInt(raw.trim());
            return Math.max(1, Math.min(16, parsed));
        } catch (NumberFormatException ignored) {
            return 4;
        }
    }

    /**
     * Speculatively prepares the next bounded QF window.  Only immutable
     * (image,geometry,QF) state is built; message capacity, selected-QF,
     * ciphertext, STC and verification decisions remain in the caller.
     */
    public static void prefetchAhead(
            Path png,
            SourceImageContext source,
            CandidateGenerator.Context candidateContext,
            int firstQf,
            int maximumQf,
            SamplingGeometry geometry
    ) {
        if (!enabled()) return;
        int depth = prefetchDepth();
        if (depth <= 0 || firstQf > maximumQf) return;
        int end = Math.min(maximumQf, firstQf + depth - 1);
        for (int qf = firstQf; qf <= end; qf++) {
            final int scheduledQf = qf;
            PREFETCH_EXECUTOR.execute(() -> {
                try {
                    prepare(
                            png, source, candidateContext,
                            scheduledQf, geometry);
                } catch (Throwable ignored) {
                    // The foreground QF trial remains authoritative and will
                    // retry/propagate according to the execution policy.
                }
            });
        }
    }

    public static int maximumSourceImages() {
        String raw = System.getProperty(MAX_SOURCE_IMAGES_PROPERTY,
                System.getenv().getOrDefault("RAP_SOURCE_CACHE_IMAGES",
                        Integer.toString(DEFAULT_MAX_SOURCE_IMAGES)));
        int parsed;
        try {
            parsed = Integer.parseInt(raw.trim());
        } catch (NumberFormatException bad) {
            throw new IllegalArgumentException(
                    "RAP_SOURCE_CACHE_IMAGES must be an integer", bad);
        }
        if (parsed < 1 || parsed > 128) {
            throw new IllegalArgumentException(
                    "RAP_SOURCE_CACHE_IMAGES must be in 1..128: " + parsed);
        }
        return parsed;
    }

    /**
     * Returns one shared immutable source context per stable PNG identity.
     * This makes RGB->YCbCr, geometry resampling, raw DCT and block variance
     * reusable across all message workers and QF states for the same image.
     */
    public static SourceImageContext sharedSourceContext(
            Path png, SourceImageContext verifiedContext
    ) throws Exception {
        if (!enabled()) return verifiedContext;
        if (verifiedContext == null) {
            throw new IllegalArgumentException("verifiedContext cannot be null");
        }
        SourceKey key = SourceKey.of(png, verifiedContext);
        synchronized (LOCK) {
            SourceImageContext existing = SOURCE_CACHE.get(key);
            if (existing != null) return existing;
            SOURCE_CACHE.put(key, verifiedContext);
            while (SOURCE_CACHE.size() > maximumSourceImages()) {
                var eldest = SOURCE_CACHE.entrySet().iterator().next();
                SOURCE_CACHE.remove(eldest.getKey());
            }
            return verifiedContext;
        }
    }

    public static State prepare(
            Path png,
            SourceImageContext source,
            CandidateGenerator.Context candidateContext,
            int qf,
            SamplingGeometry geometry
    ) throws Exception {
        if (!enabled()) {
            throw new IllegalStateException("True-parallel state cache is disabled");
        }
        if (candidateContext == null || candidateContext.source() != source) {
            throw new IllegalArgumentException("Candidate context must use the supplied source snapshot");
        }
        Key key = Key.of(png, source, qf, geometry);
        CompletableFuture<State> future;
        boolean builder = false;
        synchronized (LOCK) {
            future = CACHE.get(key);
            if (future == null) {
                future = new CompletableFuture<>();
                CACHE.put(key, future);
                builder = true;
                trimLocked(maximumStates());
            }
        }

        if (builder) {
            try {
                State state = build(key, source, candidateContext, qf, geometry);
                future.complete(state);
                synchronized (LOCK) {
                    trimLocked(maximumStates());
                }
            } catch (Throwable failure) {
                future.completeExceptionally(failure);
                synchronized (LOCK) {
                    CACHE.remove(key, future);
                }
            }
        }

        try {
            return future.join();
        } catch (CompletionException wrapped) {
            Throwable cause = wrapped.getCause();
            if (cause instanceof Exception exception) throw exception;
            if (cause instanceof Error error) throw error;
            throw new IllegalStateException("Prepared-state construction failed", cause);
        }
    }

    public static int residentStates() {
        synchronized (LOCK) {
            return CACHE.size();
        }
    }

    public static void clearMemoryIndex() {
        synchronized (LOCK) {
            CACHE.clear();
            SOURCE_CACHE.clear();
        }
    }

    public static Path cacheDirectory() throws Exception {
        Path local = cacheDirectory;
        if (local != null) return local;
        synchronized (LOCK) {
            local = cacheDirectory;
            if (local == null) {
                String configured = System.getenv().getOrDefault(
                        "RAP_PREPARED_CACHE_DIR", "build/true_parallel_prepared_cache");
                local = Path.of(configured).toAbsolutePath().normalize();
                Files.createDirectories(local);
                cacheDirectory = local;
                installCleanupHookLocked();
            }
            return local;
        }
    }


    private static void installCleanupHookLocked() {
        if (cleanupHookInstalled) return;
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (keepPreparedCache()) return;
            Path directory = cacheDirectory;
            if (directory != null) {
                try { Files.deleteIfExists(directory); }
                catch (Exception ignored) { /* do not delete nonempty shared dirs */ }
            }
        }, "rap-standalone-prepared-cache-cleanup"));
        cleanupHookInstalled = true;
    }

    private static boolean keepPreparedCache() {
        return Boolean.parseBoolean(System.getProperty(
                "rap.true.parallel.keep.cache",
                System.getenv().getOrDefault(
                        "RAP_KEEP_PREPARED_CACHE", "false")));
    }

    private static State build(
            Key key,
            SourceImageContext source,
            CandidateGenerator.Context candidateContext,
            int qf,
            SamplingGeometry geometry
    ) throws Exception {
        Path directory = cacheDirectory();
        String stem = "ref_" + Integer.toUnsignedString(key.hashCode(), 16)
                + "_" + geometry.outputTag() + "_qf" + qf;
        Path template = Files.createTempFile(directory, stem + "_", ".jpg");
        Files.deleteIfExists(template);
        final JpegImage canonical;
        final byte[] templateBytes;
        try {
            canonical = CanonicalReferenceJpeg.build(
                    source, template, qf, geometry);
            templateBytes = Files.readAllBytes(template);
        } finally {
            Files.deleteIfExists(template);
        }
        CandidateSet candidateSet = CandidateGenerator.generateSet(
                candidateContext, canonical);
        return new State(templateBytes, canonical, candidateSet);
    }

    private static void trimLocked(int maxStates) {
        while (CACHE.size() > maxStates) {
            Key removable = null;
            for (Map.Entry<Key, CompletableFuture<State>> entry
                    : CACHE.entrySet()) {
                // Never evict an in-flight construction.  Evicting an
                // incomplete future allows a second worker to rebuild the
                // same expensive (image,geometry,QF) state concurrently,

                // oversubscription is trimmed at completion and insertion.
                if (entry.getValue().isDone()) {
                    removable = entry.getKey();
                    break;
                }
            }
            if (removable == null) return;
            CACHE.remove(removable);
        }
    }

    public record State(
            byte[] canonicalTemplateBytes,
            JpegImage canonicalReference,
            CandidateSet candidateSet
    ) {
        public State {
            if (canonicalTemplateBytes == null || canonicalTemplateBytes.length == 0
                    || canonicalReference == null || candidateSet == null) {
                throw new IllegalArgumentException("Prepared state cannot contain nulls");
            }
            canonicalTemplateBytes = Arrays.copyOf(
                    canonicalTemplateBytes, canonicalTemplateBytes.length);
            canonicalReference = deepCopy(canonicalReference);
        }

        @Override public byte[] canonicalTemplateBytes() {
            return Arrays.copyOf(
                    canonicalTemplateBytes, canonicalTemplateBytes.length);
        }

        @Override public JpegImage canonicalReference() {
            return deepCopy(canonicalReference);
        }

        public JpegImage freshReferenceImage() {
            return deepCopy(canonicalReference);
        }

        /** Writes the byte-identical canonical JPEG to the caller's own file. */
        public void materializeReference(Path target) throws Exception {
            Files.write(target, canonicalTemplateBytes);
        }
    }

    public static JpegImage deepCopy(JpegImage source) {
        JpegComponent[] components = new JpegComponent[source.components.length];
        for (int i = 0; i < components.length; i++) {
            JpegComponent c = source.component(i);
            components[i] = new JpegComponent(
                    c.componentIndex,
                    c.widthBlocks,
                    c.heightBlocks,
                    c.hSamplingFactor,
                    c.vSamplingFactor,
                    c.quantTableIndex,
                    Arrays.copyOf(c.coefficients, c.coefficients.length),
                    Arrays.copyOf(c.quantTable, c.quantTable.length));
        }
        return new JpegImage(source.width, source.height, components);
    }

    private record SourceKey(
            String normalizedPath,
            String sha256
    ) {
        static SourceKey of(Path png, SourceImageContext source) {
            Path normalized = png.toAbsolutePath().normalize();
            if (!normalized.equals(source.sourcePath())) {
                throw new IllegalArgumentException("PNG path does not match the supplied source snapshot");
            }
            return new SourceKey(normalized.toString(), source.sourceSha256());
        }
    }

    private record Key(
            SourceKey source,
            int qf,
            SamplingGeometry geometry
    ) {
        static Key of(Path png, SourceImageContext source, int qf, SamplingGeometry geometry) {
            return new Key(SourceKey.of(png, source), qf, geometry);
        }
    }

    private static final class PrefetchThreadFactory implements ThreadFactory {
        private final AtomicInteger sequence = new AtomicInteger();

        @Override
        public Thread newThread(Runnable task) {
            Thread thread = new Thread(
                    task,
                    "rap-standalone-qf-prefetch-" + sequence.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        }
    }

}
