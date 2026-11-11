package thesis.rap.app;

import thesis.rap.stc.BitPacking;
import thesis.rap.stc.CudaStcCodec;
import thesis.rap.stc.NoFiniteStcSolutionException;
import thesis.rap.stc.OriginalStcCodec;
import thesis.rap.stc.StcCodec;
import thesis.rap.runtime.ExecutionBackend;
import thesis.rap.runtime.GpuStageStats;


public final class StcPipelineIntegration implements StcEmbeddingBackend {
    static final int DEFAULT_CUDA_MIN_CARRIERS = 4096;

    private final StcCodec codec;
    private final StcCodec cpuReference;
    private final boolean cudaSelected;
    private final boolean cpuFallbackAllowed;
    private final int cudaMinCarriers;
    private final int constraintHeight;
    private final String backendName;

    public StcPipelineIntegration(int constraintHeight) {
        this(constraintHeight, selectCodec());
    }

    private StcPipelineIntegration(int constraintHeight, BackendSelection selection) {
        this(constraintHeight, selection.name(), selection.codec(),
                selection.codec() instanceof CudaStcCodec
                        ? new OriginalStcCodec() : null,
                !ExecutionBackend.cudaFailureIsFatal(), cudaMinCarriers());
    }

    /**
     * Routing core.  {@code cpuReference == null} means the selected codec is
     * already the CPU reference; otherwise {@code accelerated} is the GPU codec.
     * Package-private so the routing can be tested without a GPU.
     */
    StcPipelineIntegration(
            int constraintHeight,
            String selectionName,
            StcCodec accelerated,
            StcCodec cpuReference,
            boolean cpuFallbackAllowed,
            int cudaMinCarriers
    ) {
        StcCarrierPlan.validateSupportedHeight(constraintHeight);
        this.codec = accelerated;
        this.cudaSelected = cpuReference != null;
        this.cpuReference = cudaSelected ? cpuReference : accelerated;
        this.cpuFallbackAllowed = cpuFallbackAllowed;
        this.cudaMinCarriers = cudaMinCarriers;
        this.backendName = cudaSelected
                ? selectionName + "(min_carriers=" + cudaMinCarriers
                        + ",cpu_fallback=" + cpuFallbackAllowed + ")"
                : selectionName;
        this.constraintHeight = constraintHeight;
    }

    public String backendName() {
        return backendName;
    }

    private static BackendSelection selectCodec() {
        String requested;
        if (ExecutionBackend.explicitlyConfigured()) {
            requested = switch (ExecutionBackend.mode()) {
                case CPU -> "CPU";
                case CUDA -> "CUDA";
                case AUTO -> "AUTO";
                case HIP -> "CPU"; // STC has no HIP implementation; preserve semantics on CPU.
            };
        } else {
            requested = System.getProperty(
                    "rap.stc.backend",
                    System.getenv().getOrDefault("RAP_STC_BACKEND", "AUTO"))
                    .trim().toUpperCase(java.util.Locale.ROOT);
        }

        return switch (requested) {
            case "CUDA" -> {
                if (CudaStcCodec.isAvailable()) {
                    yield new BackendSelection(new CudaStcCodec(), "CUDA");
                }
                if (ExecutionBackend.explicitAcceleratorFallbackEnabled()) {
                    yield new BackendSelection(new OriginalStcCodec(), "CPU_FALLBACK_FROM_CUDA");
                }
                throw new IllegalStateException(
                        "CUDA backend requested but CUDA STC is unavailable.");
            }
            case "CPU" -> new BackendSelection(new OriginalStcCodec(), "CPU");
            case "AUTO" -> {
                if (ExecutionBackend.cudaPermitted() && CudaStcCodec.isAvailable()) {
                    yield new BackendSelection(new CudaStcCodec(), "CUDA");
                }
                yield new BackendSelection(new OriginalStcCodec(), "CPU");
            }
            default -> throw new IllegalArgumentException(
                    "RAP_STC_BACKEND must be AUTO, CUDA, or CPU; got " + requested);
        };
    }

    private static int cudaMinCarriers() {
        String raw = System.getProperty("rap.stc.cuda.min.carriers",
                System.getenv().getOrDefault("RAP_STC_CUDA_MIN_CARRIERS",
                        Integer.toString(DEFAULT_CUDA_MIN_CARRIERS)));
        try {
            int parsed = Integer.parseInt(raw.trim());
            if (parsed < 0) throw new NumberFormatException("negative");
            return parsed;
        } catch (NumberFormatException bad) {
            throw new IllegalArgumentException(
                    "RAP_STC_CUDA_MIN_CARRIERS must be a nonnegative integer: " + raw, bad);
        }
    }

    private record BackendSelection(StcCodec codec, String name) {}

    @Override
    public int constraintHeight() {
        return constraintHeight;
    }

    public StcEmbedResult embedPayloadWithTrace(byte[] coverBits, double[] costs, byte[] payloadBytes) {
        byte[] messageBits = BitPacking.bytesToBits(payloadBytes);
        byte[] stegoBits = embedBits(coverBits, costs, messageBits);
        if (stegoBits.length != coverBits.length) {
            throw new IllegalStateException("Native STC returned a stego vector with the wrong length.");
        }

        int changed = 0;
        double weightedDistortion = 0.0;
        for (int i = 0; i < coverBits.length; i++) {
            if (coverBits[i] != stegoBits[i]) {
                changed++;
                weightedDistortion += costs[i];
            }
        }
        return new StcEmbedResult(stegoBits,
                new StcEmbeddingTrace(coverBits.length, messageBits.length, changed, weightedDistortion));
    }

    public byte[] embedPayload(byte[] coverBits, double[] costs, byte[] payloadBytes) {
        return embedPayloadWithTrace(coverBits, costs, payloadBytes).stegoBits();
    }

    public byte[] extractPayload(byte[] stegoBits, int payloadByteLength) {
        int messageBitLength = Math.multiplyExact(payloadByteLength, 8);
        return BitPacking.bitsToBytes(extractBits(stegoBits, messageBitLength));
    }

    private byte[] embedBits(byte[] coverBits, double[] costs, byte[] messageBits) {
        if (!cudaSelected || coverBits.length < cudaMinCarriers) {
            return cpuReference.embed(coverBits, costs, messageBits, constraintHeight);
        }
        try {
            byte[] stego = codec.embed(coverBits, costs, messageBits, constraintHeight);
            GpuStageStats.gpu(GpuStageStats.Stage.STC);
            return stego;
        } catch (NoFiniteStcSolutionException | IllegalArgumentException methodOutcome) {
            throw methodOutcome;
        } catch (RuntimeException | LinkageError | OutOfMemoryError gpuFailure) {
            if (!cpuFallbackAllowed) throw gpuFailure;
            GpuStageStats.fallback(GpuStageStats.Stage.STC, gpuFailure);
            return cpuReference.embed(coverBits, costs, messageBits, constraintHeight);
        }
    }

    private byte[] extractBits(byte[] stegoBits, int messageBitLength) {
        if (!cudaSelected || stegoBits.length < cudaMinCarriers) {
            return cpuReference.extract(stegoBits, messageBitLength, constraintHeight);
        }
        try {
            byte[] message = codec.extract(stegoBits, messageBitLength, constraintHeight);
            GpuStageStats.gpu(GpuStageStats.Stage.STC);
            return message;
        } catch (IllegalArgumentException methodOutcome) {
            throw methodOutcome;
        } catch (RuntimeException | LinkageError | OutOfMemoryError gpuFailure) {
            if (!cpuFallbackAllowed) throw gpuFailure;
            GpuStageStats.fallback(GpuStageStats.Stage.STC, gpuFailure);
            return cpuReference.extract(stegoBits, messageBitLength, constraintHeight);
        }
    }
}
