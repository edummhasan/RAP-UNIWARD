package thesis.rap.app;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


final class BatchExecutorSupport {
    /** Sentinel used only by virtual mode: derive a bounded in-flight window automatically. */
    static final int ALL_DISCOVERED_PAIR_TASKS = 0;
    /** Auto mode keeps two waves of work ahead of the real resource bottleneck. */
    private static final int AUTO_IN_FLIGHT_MULTIPLIER = 2;
    /** Guardrail for physical/resource parallelism, never for virtual tasks. */
    static final int MAX_RESOURCE_PARALLELISM = 4096;
    private static final String SCHEDULER_PARALLELISM =
            "jdk.virtualThreadScheduler.parallelism";
    private static final String SCHEDULER_MAX_POOL_SIZE =
            "jdk.virtualThreadScheduler.maxPoolSize";
    private BatchExecutorSupport() {}

    static String currentWorkerLabel() {
        Thread thread = Thread.currentThread();
        String name = thread.getName();
        if (name != null && !name.isBlank()) return name;
        String kind = isVirtual(thread)
                ? "rap-vworker" : "rap-worker";
        return kind + "-" + threadIdentifier(thread);
    }

    static String observedCpuAffinity() {
        Path processStatus = Path.of("/proc", "self", "status");
        if (!Files.isRegularFile(processStatus)) return "UNAVAILABLE";
        try {
            for (String line : Files.readAllLines(processStatus)) {
                if (line.startsWith("Cpus_allowed_list:")) {
                    String value = line.substring(
                            "Cpus_allowed_list:".length()).trim();
                    return value.isBlank() ? "UNAVAILABLE" : value;
                }
            }
        } catch (Exception ignored) {
            return "UNAVAILABLE";
        }
        return "UNAVAILABLE";
    }

    enum ThreadMode {
        VIRTUAL,
        PLATFORM;

        static ThreadMode parse(String value) {
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException("thread mode cannot be blank");
            }
            return switch (value.trim().toUpperCase(Locale.ROOT)) {
                case "VIRTUAL" -> VIRTUAL;
                case "PLATFORM" -> PLATFORM;
                default -> throw new IllegalArgumentException(
                        "--threadMode must be virtual or platform: " + value);
            };
        }
    }

    record Config(
            int workers,
            int carrierParallelism,
            int cpuSlots,
            ThreadMode mode
    ) {
        Config {
            if (mode == null) {
                throw new IllegalArgumentException("thread mode cannot be null");
            }
            if (carrierParallelism < 1
                    || carrierParallelism > MAX_RESOURCE_PARALLELISM) {
                throw new IllegalArgumentException(
                        "carrierParallelism must be from 1 through "
                                + MAX_RESOURCE_PARALLELISM);
            }
            if (cpuSlots < 1
                    || cpuSlots > MAX_RESOURCE_PARALLELISM) {
                throw new IllegalArgumentException(
                        "cpuSlots must be from 1 through "
                                + MAX_RESOURCE_PARALLELISM);
            }
            if (mode == ThreadMode.VIRTUAL) {
                if (workers < ALL_DISCOVERED_PAIR_TASKS) {
                    throw new IllegalArgumentException(
                            "virtual workers must be auto or a positive "
                                    + "in-flight task-window size");
                }
            } else {
                if (workers < 1
                        || workers > MAX_RESOURCE_PARALLELISM) {
                    throw new IllegalArgumentException(
                            "platform workers must be from 1 through "
                                    + MAX_RESOURCE_PARALLELISM);
                }
                if (cpuSlots > workers) {
                    throw new IllegalArgumentException(
                            "platform cpuSlots cannot exceed workers");
                }
            }
        }

        boolean allDiscoveredPairTasks() {
            return mode == ThreadMode.VIRTUAL
                    && workers == ALL_DISCOVERED_PAIR_TASKS;
        }

        boolean boundedVirtualTasks() {
            return mode == ThreadMode.VIRTUAL
                    && workers > ALL_DISCOVERED_PAIR_TASKS;
        }

        int submissionLimit(int discoveredPairCount) {
            if (discoveredPairCount < 0) {
                throw new IllegalArgumentException(
                        "discoveredPairCount cannot be negative");
            }
            if (allDiscoveredPairTasks()) {
                int resourceWidth = Math.max(cpuSlots, carrierParallelism);
                int automaticWindow = Math.multiplyExact(
                        AUTO_IN_FLIGHT_MULTIPLIER, resourceWidth);
                return Math.min(discoveredPairCount, automaticWindow);
            }
            return Math.min(workers, discoveredPairCount);
        }

        boolean usesMultipleTasks(int discoveredPairCount) {
            return submissionLimit(discoveredPairCount) > 1;
        }

        String taskPolicy() {
            if (allDiscoveredPairTasks()) {
                int automaticWindow = AUTO_IN_FLIGHT_MULTIPLIER
                        * Math.max(cpuSlots, carrierParallelism);
                return "VIRTUAL_AUTO_IN_FLIGHT_WINDOW_" + automaticWindow;
            }
            if (boundedVirtualTasks()) {
                return "VIRTUAL_IN_FLIGHT_WINDOW_" + workers;
            }
            return "PLATFORM_BOUNDED_" + workers;
        }
    }

    static ExecutorService create(Config config) {
        return switch (config.mode()) {
            case PLATFORM -> Executors.newFixedThreadPool(
                    config.workers(), namedPlatformFactory());
            case VIRTUAL -> createVirtualExecutor(config);
        };
    }

    private static ExecutorService createVirtualExecutor(Config config) {
        setOrValidatePositiveProperty(
                SCHEDULER_PARALLELISM, config.carrierParallelism());
        setOrValidatePositiveProperty(
                SCHEDULER_MAX_POOL_SIZE, config.carrierParallelism());

        try {
            Object executor = Executors.class
                    .getMethod("newVirtualThreadPerTaskExecutor")
                    .invoke(null);
            if (executor instanceof ExecutorService service) {
                return service;
            }
            throw new IllegalStateException(
                    "Java virtual-thread factory returned an unexpected type");
        } catch (ReflectiveOperationException unavailable) {
            throw new IllegalStateException(
                    "Virtual batch mode requires Java 22 or newer",
                    unavailable);
        }
    }

    private static boolean isVirtual(Thread thread) {
        try {
            return (Boolean) Thread.class.getMethod("isVirtual")
                    .invoke(thread);
        } catch (ReflectiveOperationException unavailable) {
            return false;
        }
    }

    private static long threadIdentifier(Thread thread) {
        try {
            return (Long) Thread.class.getMethod("threadId")
                    .invoke(thread);
        } catch (ReflectiveOperationException unavailable) {
            return Integer.toUnsignedLong(
                    System.identityHashCode(thread));
        }
    }

    /**
     * Fair semaphore limiting the number of complete pair kernels that may
     * execute simultaneously. Waiting virtual threads unmount normally and do
     * not consume a CPU slot until a permit is acquired.
     */
    static final class CpuSlotGate {
        private final int limit;
        private final Semaphore slots;
        private final AtomicInteger active = new AtomicInteger();
        private final AtomicInteger peakActive = new AtomicInteger();

        CpuSlotGate(int limit) {
            if (limit < 1 || limit > MAX_RESOURCE_PARALLELISM) {
                throw new IllegalArgumentException(
                        "CPU slot limit must be from 1 through "
                                + MAX_RESOURCE_PARALLELISM);
            }
            this.limit = limit;
            this.slots = new Semaphore(limit, true);
        }

        CpuPermit acquire() throws InterruptedException {
            slots.acquire();
            int now = active.incrementAndGet();
            if (now > limit) {
                active.decrementAndGet();
                slots.release();
                throw new IllegalStateException(
                        "CPU-slot semaphore admitted more than " + limit
                                + " active pair kernels");
            }
            peakActive.accumulateAndGet(now, Math::max);
            return new CpuPermit(this);
        }

        int limit() {
            return limit;
        }

        int active() {
            return active.get();
        }

        int availablePermits() {
            return slots.availablePermits();
        }

        int queuedThreads() {
            return slots.getQueueLength();
        }

        int peakActive() {
            return peakActive.get();
        }

        private void release() {
            int remaining = active.decrementAndGet();
            if (remaining < 0) {
                active.incrementAndGet();
                throw new IllegalStateException(
                        "CPU-slot semaphore permit released twice");
            }
            slots.release();
        }
    }

    static final class CpuPermit implements AutoCloseable {
        private final CpuSlotGate gate;
        private boolean closed;

        private CpuPermit(CpuSlotGate gate) {
            this.gate = gate;
        }

        @Override
        public void close() {
            if (closed) {
                throw new IllegalStateException(
                        "CPU-slot permit released twice");
            }
            closed = true;
            gate.release();
        }
    }

    private static ThreadFactory namedPlatformFactory() {
        AtomicLong sequence = new AtomicLong();
        return task -> {
            Thread thread = new Thread(
                    task, "rap-pworker-" + sequence.getAndIncrement());
            thread.setDaemon(false);
            return thread;
        };
    }

    private static void setOrValidatePositiveProperty(
            String property, int requested
    ) {
        String existing = System.getProperty(property);
        if (existing == null || existing.isBlank()) {
            System.setProperty(property, Integer.toString(requested));
            return;
        }
        final int observed;
        try {
            observed = Integer.parseInt(existing);
        } catch (NumberFormatException invalid) {
            throw new IllegalArgumentException(
                    "Invalid JVM property -D" + property + "=" + existing,
                    invalid);
        }
        if (observed != requested) {
            throw new IllegalArgumentException(
                    "Conflicting virtual-thread scheduler parallelism: --carrierParallelism="
                            + requested + " but -D" + property + "=" + observed);
        }
    }

    static void close(
            ExecutorService executor, boolean completedNormally
    ) {
        close(executor, completedNormally, 60, 60, TimeUnit.SECONDS);
    }

    static void closePreservingFailure(ExecutorService executor,
            boolean completedNormally, Throwable primaryFailure) {
        try {
            close(executor, completedNormally);
        } catch (RuntimeException | Error shutdownFailure) {
            if (primaryFailure == null) throw shutdownFailure;
            primaryFailure.addSuppressed(shutdownFailure);
            FailureDiagnostics.log(primaryFailure);
        }
    }

    static void close(ExecutorService executor, boolean completedNormally,
            long grace, long force, TimeUnit unit) {
        int abandoned = 0;
        if (completedNormally) executor.shutdown();
        else abandoned = executor.shutdownNow().size();
        try {
            if (!executor.awaitTermination(grace, unit)) {
                abandoned += executor.shutdownNow().size();
                if (!executor.awaitTermination(force, unit)) {
                    throw new IllegalStateException(
                            "Batch worker executor did not terminate; abandoned=" + abandoned);
                }
            }
        } catch (InterruptedException interrupted) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
            throw new IllegalStateException(
                    "Interrupted while stopping batch workers", interrupted);
        }
    }
}
