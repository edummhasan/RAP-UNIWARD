package thesis.rap.app;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Regression checks for unbounded virtual tasks and bounded CPU admission. */
public final class BatchExecutorSupportSelfTest {
    private BatchExecutorSupportSelfTest() {}

    public static void main(String[] args) throws Exception {
        configurationParsing();
        configurableVirtualTopology();
        auditSwitchParsing();
        leadingZeroIdentifiersRemainText();
        virtualAutoPolicyBoundsInFlightWork();
        platformConcurrencyIsBounded();
        cpuSemaphoreLimits64Tasks();
        if (virtualApisAvailable()) {
            virtualModeContract();
        } else {
            System.out.println(
                    "BatchExecutorSupportSelfTest: SKIP virtual API "
                            + "contract (audit host is below Java 22)");
        }
        System.out.println(
                "BatchExecutorSupportSelfTest: ALL TESTS PASSED");
    }

    private static void configurationParsing() {
        BatchExecutorSupport.Config defaults =
                RapUniward.batchThreading(Map.of());
        require(defaults.workers() == 1,
                "direct CLI default must remain one worker");
        require(defaults.mode()
                        == BatchExecutorSupport.ThreadMode.PLATFORM,
                "direct CLI default must remain platform-compatible");
        require(defaults.cpuSlots() == 1,
                "direct CLI default must retain one CPU slot");

        BatchExecutorSupport.Config requested =
                RapUniward.batchThreading(Map.of(
                        "--workers", "auto",
                        "--carrierParallelism", "8",
                        "--cpuSlots", "3",
                        "--threadMode", "virtual"));
        require(requested.workers()
                        == BatchExecutorSupport.ALL_DISCOVERED_PAIR_TASKS,
                "virtual auto sentinel was not retained");
        require(requested.carrierParallelism() == 8,
                "8-way carrier parallelism was not retained");
        require(requested.cpuSlots() == 3,
                "independent CPU-slot permits were not retained");
        require(requested.mode()
                        == BatchExecutorSupport.ThreadMode.VIRTUAL,
                "virtual mode was not retained");
        require(requested.allDiscoveredPairTasks(),
                "virtual auto sentinel was not recognized");

        expectFailure(() -> RapUniward.batchThreading(Map.of(
                        "--workers", "0")),
                "zero workers must be rejected");
        expectFailure(() -> RapUniward.batchThreading(Map.of(
                        "--workers", "auto",
                        "--threadMode", "platform")),
                "platform mode must require a finite worker count");
        expectFailure(() -> RapUniward.batchThreading(Map.of(
                        "--threadMode", "imaginary")),
                "unknown thread mode must be rejected");
    }

    private static void configurableVirtualTopology() {
        BatchExecutorSupport.Config bounded =
                RapUniward.batchThreading(Map.of(
                        "--workers", "64",
                        "--carrierParallelism", "11",
                        "--cpuSlots", "5",
                        "--threadMode", "virtual"));
        require(bounded.boundedVirtualTasks(),
                "numeric virtual workers did not select a bounded window");
        require(!bounded.allDiscoveredPairTasks(),
                "numeric virtual workers were mislabeled as auto");
        require(bounded.submissionLimit(1_000) == 64,
                "numeric virtual in-flight window was not retained");
        require(bounded.carrierParallelism() == 11,
                "carrier parallelism was not independently retained");
        require(bounded.cpuSlots() == 5,
                "CPU-slot count was not independently retained");
        require("VIRTUAL_IN_FLIGHT_WINDOW_64".equals(
                        bounded.taskPolicy()),
                "bounded virtual policy label is inaccurate");

        BatchExecutorSupport.Config largerThanCorpus =
                RapUniward.batchThreading(Map.of(
                        "--workers", "100000",
                        "--carrierParallelism", "2",
                        "--cpuSlots", "9",
                        "--threadMode", "virtual"));
        require(largerThanCorpus.submissionLimit(17) == 17,
                "bounded window submitted nonexistent input pairs");
    }

    private static void virtualAutoPolicyBoundsInFlightWork() {
        BatchExecutorSupport.Config config =
                RapUniward.batchThreading(Map.of(
                        "--workers", "auto",
                        "--carrierParallelism", "8",
                        "--cpuSlots", "8",
                        "--threadMode", "virtual"));
        int discoveredPairs = 53_250;
        require(config.submissionLimit(discoveredPairs) == 16,
                "virtual auto mode must keep a two-wave resource window");
        require(config.submissionLimit(discoveredPairs)
                        < discoveredPairs,
                "virtual auto mode unnecessarily submitted the entire corpus");
        require("VIRTUAL_AUTO_IN_FLIGHT_WINDOW_16".equals(
                        config.taskPolicy()),
                "virtual auto task policy label is inaccurate");

        BatchExecutorSupport.Config asymmetric =
                RapUniward.batchThreading(Map.of(
                        "--workers", "auto",
                        "--carrierParallelism", "14",
                        "--cpuSlots", "6",
                        "--threadMode", "virtual"));
        require(asymmetric.submissionLimit(discoveredPairs) == 28,
                "auto window must follow the larger carrier/resource width");
        require(asymmetric.submissionLimit(7) == 7,
                "auto window must not invent pair tasks");
    }

    private static void auditSwitchParsing() {
        String environment = System.getenv("RAP_AUDIT_ENABLED");
        if (environment == null || environment.isBlank()) {
            require(RapUniward.auditEnabled(Map.of()),
                    "auditing must remain enabled by default");
        }
        require(!RapUniward.auditEnabled(
                        Map.of("--auditEnabled", "false")),
                "explicit audit disable was ignored");
        require(RapUniward.auditEnabled(
                        Map.of("--auditEnabled", "true")),
                "explicit audit enable was ignored");
        expectFailure(() -> RapUniward.auditEnabled(
                        Map.of("--auditEnabled", "maybe")),
                "non-boolean audit option must be rejected");

        String modeEnvironment = System.getenv("RAP_AUDIT_MODE");
        if (modeEnvironment == null || modeEnvironment.isBlank()) {
            require(RapUniward.auditMode(Map.of())
                            == AuditMode.RESULT_ONLY,
                    "result-only auditing must be the storage-safe default");
        }
        require(RapUniward.auditMode(
                        Map.of("--auditMode", "QF_FORENSIC"))
                        == AuditMode.QF_FORENSIC,
                "explicit QF forensic mode was ignored");
        expectFailure(() -> RapUniward.auditMode(
                        Map.of("--auditMode", "both")),
                "unknown audit mode must be rejected");
    }

    private static void leadingZeroIdentifiersRemainText() {
        Path pngRoot = Path.of("/tmp/rap-input");
        Path messageRoot = Path.of("/tmp/rap-message");
        String longId = RapUniward.batchOutputName(
                pngRoot, pngRoot.resolve("000001.png"),
                messageRoot, messageRoot.resolve("001.msg"));
        String shortId = RapUniward.batchOutputName(
                pngRoot, pngRoot.resolve("0001.png"),
                messageRoot, messageRoot.resolve("001.msg"));
        require(!longId.equals(shortId),
                "leading-zero image IDs were treated as numbers");
        require(longId.contains("000001") && shortId.contains("0001"),
                "textual image IDs were not retained in output identity");
    }

    private static void platformConcurrencyIsBounded()
            throws Exception {
        int workers = 4;
        ExecutorService executor = BatchExecutorSupport.create(
                new BatchExecutorSupport.Config(
                        workers, 2, 2,
                        BatchExecutorSupport.ThreadMode.PLATFORM));
        CountDownLatch entered = new CountDownLatch(workers);
        CountDownLatch release = new CountDownLatch(1);
        AtomicInteger active = new AtomicInteger();
        AtomicInteger maximum = new AtomicInteger();
        List<Future<?>> futures = new ArrayList<>();
        boolean completedNormally = false;
        try {
            for (int i = 0; i < workers * 2; i++) {
                futures.add(executor.submit(() -> {
                    int now = active.incrementAndGet();
                    maximum.accumulateAndGet(now, Math::max);
                    entered.countDown();
                    try {
                        if (!release.await(10, TimeUnit.SECONDS)) {
                            throw new IllegalStateException(
                                    "test release timed out");
                        }
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException(interrupted);
                    } finally {
                        active.decrementAndGet();
                    }
                }));
            }
            require(entered.await(10, TimeUnit.SECONDS),
                    "configured workers did not start");
            require(maximum.get() == workers,
                    "executor did not reach the configured bound");
            release.countDown();
            for (Future<?> future : futures) future.get();
            require(maximum.get() <= workers,
                    "executor exceeded its configured bound");
            completedNormally = true;
        } finally {
            release.countDown();
            BatchExecutorSupport.close(executor, completedNormally);
        }
    }

    @SuppressWarnings("try")
    private static void cpuSemaphoreLimits64Tasks()
            throws Exception {
        int workers = 64;
        int slots = 8;
        ExecutorService executor = BatchExecutorSupport.create(
                new BatchExecutorSupport.Config(
                        workers, slots, slots,
                        BatchExecutorSupport.ThreadMode.PLATFORM));
        BatchExecutorSupport.CpuSlotGate gate =
                new BatchExecutorSupport.CpuSlotGate(slots);
        CountDownLatch firstWave = new CountDownLatch(slots);
        CountDownLatch release = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>();
        boolean completedNormally = false;
        try {
            for (int i = 0; i < workers; i++) {
                futures.add(executor.submit(() -> {
                    try (BatchExecutorSupport.CpuPermit ignored =
                                 gate.acquire()) {
                        firstWave.countDown();
                        if (!release.await(10, TimeUnit.SECONDS)) {
                            throw new IllegalStateException(
                                    "CPU-slot release timed out");
                        }
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException(interrupted);
                    }
                }));
            }
            require(firstWave.await(10, TimeUnit.SECONDS),
                    "eight semaphore permits were not acquired");
            require(gate.active() == slots,
                    "semaphore did not hold exactly eight active tasks");
            require(gate.peakActive() == slots,
                    "semaphore did not reach its configured limit");
            release.countDown();
            for (Future<?> future : futures) future.get();
            require(gate.peakActive() == slots,
                    "64 submitted tasks exceeded eight active CPU slots");
            require(gate.active() == 0,
                    "CPU-slot permits leaked after task completion");
            require(gate.availablePermits() == slots,
                    "CPU-slot semaphore did not recover all permits");
            completedNormally = true;
        } finally {
            release.countDown();
            BatchExecutorSupport.close(executor, completedNormally);
        }
    }

    private static void virtualModeContract() throws Exception {
        BatchExecutorSupport.Config config =
                new BatchExecutorSupport.Config(
                        BatchExecutorSupport.ALL_DISCOVERED_PAIR_TASKS,
                        1, 1,
                        BatchExecutorSupport.ThreadMode.VIRTUAL);
        ExecutorService executor = BatchExecutorSupport.create(config);
        boolean completedNormally = false;
        try {
            Future<Boolean> virtual = executor.submit(() -> {
                Thread thread = Thread.currentThread();
                return isVirtual(thread)
                        && !BatchExecutorSupport.currentWorkerLabel()
                        .isBlank();
            });
            require(virtual.get(),
                    "virtual mode did not create and label a real virtual thread");
            completedNormally = true;
        } finally {
            BatchExecutorSupport.close(executor, completedNormally);
        }
    }

    private static boolean virtualApisAvailable() {
        try {
            Thread.class.getMethod("isVirtual");
            java.util.concurrent.Executors.class.getMethod(
                    "newVirtualThreadPerTaskExecutor");
            return true;
        } catch (NoSuchMethodException unavailable) {
            return false;
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

    private static void expectFailure(
            ThrowingRunnable action, String message
    ) {
        boolean failed = false;
        try {
            action.run();
        } catch (Exception expected) {
            failed = true;
        }
        require(failed, message);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
