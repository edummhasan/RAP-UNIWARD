package thesis.rap.app;

import thesis.rap.runtime.GpuStageStats;
import thesis.rap.stc.NoFiniteStcSolutionException;
import thesis.rap.stc.StcCodec;

import java.util.Arrays;


public final class StcCpuFallbackSelfTest {
    private StcCpuFallbackSelfTest() {}

    public static void main(String[] args) {
        gpuFailureFallsBackToCpuWhenAllowed();
        gpuFailureIsFatalWhenFailClosed();
        methodOutcomesAreNeverRetriedOnCpu();
        smallCoversNeverTouchTheGpu();
        cpuOnlySelectionUsesOneCodec();
        System.out.println("StcCpuFallbackSelfTest: ALL TESTS PASSED");
    }

    private static final byte[] COVER = new byte[5000];
    private static final double[] COSTS = new double[5000];
    private static final byte[] PAYLOAD = {1, 2};

    private static void gpuFailureFallsBackToCpuWhenAllowed() {
        Fake gpu = Fake.throwing(new IllegalStateException("cudaErrorIllegalAddress"));
        Fake cpu = Fake.working();
        long before = GpuStageStats.fallbackCount(GpuStageStats.Stage.STC);
        StcPipelineIntegration stc = new StcPipelineIntegration(
                10, "CUDA", gpu, cpu, true, 4096);
        stc.embedPayload(COVER, COSTS, PAYLOAD);
        stc.extractPayload(COVER, PAYLOAD.length);
        require(gpu.calls == 2 && cpu.calls == 2, "embed+extract must retry on CPU: " + gpu + cpu);
        require(GpuStageStats.fallbackCount(GpuStageStats.Stage.STC) == before + 2,
                "each CPU fallback must be counted");
        Fake oom = Fake.throwing(new OutOfMemoryError("device allocation"));
        new StcPipelineIntegration(10, "CUDA", oom, Fake.working(), true, 4096)
                .embedPayload(COVER, COSTS, PAYLOAD);
    }

    private static void gpuFailureIsFatalWhenFailClosed() {
        Fake gpu = Fake.throwing(new IllegalStateException("cudaErrorLaunchFailure"));
        Fake cpu = Fake.working();
        StcPipelineIntegration strict = new StcPipelineIntegration(
                10, "CUDA", gpu, cpu, false, 4096);
        try {
            strict.embedPayload(COVER, COSTS, PAYLOAD);
            throw new AssertionError("strict CUDA must not fall back");
        } catch (IllegalStateException expected) {
            require(cpu.calls == 0, "strict mode ran the CPU codec");
        }
    }

    private static void methodOutcomesAreNeverRetriedOnCpu() {
        Fake infeasible = Fake.throwing(new NoFiniteStcSolutionException("no finite path"));
        Fake cpu = Fake.working();
        try {
            new StcPipelineIntegration(10, "CUDA", infeasible, cpu, true, 4096)
                    .embedPayload(COVER, COSTS, PAYLOAD);
            throw new AssertionError("NoFiniteStcSolutionException was swallowed");
        } catch (NoFiniteStcSolutionException expected) {
            require(cpu.calls == 0, "an infeasible QF must be rejected, not retried on CPU");
        }
        Fake invalid = Fake.throwing(new IllegalArgumentException("bad dimensions"));
        try {
            new StcPipelineIntegration(10, "CUDA", invalid, cpu, true, 4096)
                    .extractPayload(COVER, PAYLOAD.length);
            throw new AssertionError("IllegalArgumentException was swallowed");
        } catch (IllegalArgumentException expected) {
            require(cpu.calls == 0, "an argument error must not be retried on CPU");
        }
    }

    private static void smallCoversNeverTouchTheGpu() {
        Fake gpu = Fake.working();
        Fake cpu = Fake.working();
        StcPipelineIntegration stc = new StcPipelineIntegration(
                10, "CUDA", gpu, cpu, true, 4096);
        stc.embedPayload(new byte[240], new double[240], PAYLOAD);
        require(gpu.calls == 0 && cpu.calls == 1, "a 240-carrier header must run on CPU");
        stc.embedPayload(COVER, COSTS, PAYLOAD);
        require(gpu.calls == 1, "a 5000-carrier body must run on the GPU");
    }

    private static void cpuOnlySelectionUsesOneCodec() {
        Fake cpu = Fake.working();
        StcPipelineIntegration stc = new StcPipelineIntegration(
                12, "CPU", cpu, null, true, 0);
        stc.embedPayload(COVER, COSTS, PAYLOAD);
        require(cpu.calls == 1 && "CPU".equals(stc.backendName()), "CPU selection misrouted");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static final class Fake implements StcCodec {
        private final Throwable failure;
        int calls;

        private Fake(Throwable failure) { this.failure = failure; }
        static Fake working() { return new Fake(null); }
        static Fake throwing(Throwable failure) { return new Fake(failure); }

        @Override
        public byte[] embed(byte[] cover, double[] costs, byte[] message, int height) {
            calls++;
            fail();
            return Arrays.copyOf(cover, cover.length);
        }

        @Override
        public byte[] extract(byte[] stego, int messageBits, int height) {
            calls++;
            fail();
            return new byte[messageBits];
        }

        private void fail() {
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
        }

        @Override public String toString() { return "calls=" + calls + " "; }
    }
}
