package thesis.rap.app;

import thesis.rap.alg.Candidate;
import thesis.rap.alg.CandidateGenerator;
import thesis.rap.alg.CandidateSet;
import thesis.rap.alg.JpegMath;
import thesis.rap.analysis.AttemptCsvWriter;
import thesis.rap.analysis.CsvUtil;
import thesis.rap.analysis.EvaluationCsvWriter;
import thesis.rap.analysis.EvaluationJsonWriter;
import thesis.rap.analysis.FileFingerprint;
import thesis.rap.analysis.EvaluationRunContext;
import thesis.rap.analysis.EvaluationSummary;
import thesis.rap.analysis.JpegMetadata;
import thesis.rap.analysis.QfSelectionMetrics;
import thesis.rap.analysis.RunReproducibility;
import thesis.rap.analysis.StcRunMetrics;
import thesis.rap.analysis.ExperimentSummaryGenerator;
import thesis.rap.analysis.DetectorMetricsAnalyzer;
import thesis.rap.analysis.FullEvaluationAnalyzer;
import thesis.rap.image.CanonicalReferenceJpeg;
import thesis.rap.image.ImageUtil;
import thesis.rap.image.GeometrySelection;
import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;
import thesis.rap.image.SourceImageContext;
import thesis.rap.image.StandardJpegQuantization;
import thesis.rap.image.UnencodableReferenceException;
import thesis.rap.jni.JpegComponent;
import thesis.rap.jni.JpegCore;
import thesis.rap.jni.JpegImage;
import thesis.rap.payload.PayloadPackage;
import thesis.rap.payload.KeyMaterial;
import thesis.rap.stc.OriginalStcCodec;
import thesis.rap.util.AtomicFiles;
import thesis.rap.runtime.ExecutionBackend;
import thesis.rap.runtime.GpuStageStats;

import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public final class RapUniward {

    /** Normative QF search range for all six algorithms. */
    private static final int DEFAULT_MIN_QF = 75;
    private static final int DEFAULT_MAX_QF = 100;
    
    private static final String APPLICATION_VERSION =
            "1.0.0.1";
    private static final int[] AUTO_EXTRACTION_HEIGHTS = {
            StcCarrierPlan.DEFAULT_STC_HEIGHT,
            StcCarrierPlan.RESEARCH_STC_HEIGHT
    };
    private static final int HEADER_BITS = PayloadPackage.headerBytes() * 8;
    private static final Path RUN_CSV = thesis.rap.runtime.RuntimePaths.results(
            "evaluation_runs.csv");
    private static final Path POSITION_CSV = thesis.rap.runtime.RuntimePaths.results(
            "evaluation_zz_positions.csv");
    private static final Path CHANNEL_CSV = thesis.rap.runtime.RuntimePaths.results(
            "evaluation_channels.csv");
    private static final Path CHANNEL_POSITION_CSV = thesis.rap.runtime.RuntimePaths.results(
            "evaluation_channel_zz_positions.csv");
    private static final Path ATTEMPT_CSV = thesis.rap.runtime.RuntimePaths.results(
            "evaluation_attempts.csv");

    private RapUniward() {}

    @SuppressWarnings("try")
    public static void main(String[] args) throws Exception {
        PairConsoleContext.install();
        if (args.length < 1 || args[0].equals("--help") || args[0].equals("help")) {
            usage();
            return;
        }

        if (args[0].equals("--version") || args[0].equals("version")) {
            System.out.println("RAP-UNIWARD " + APPLICATION_VERSION);
            return;
        }
        String mode = args[0];
        try (PairConsoleContext.Scope commandScope =
                     PairConsoleContext.command(mode)) {
            Map<String, String> o = parse(
                    Arrays.copyOfRange(args, 1, args.length));

            switch (mode) {
            case "embed" -> {
                requireOnlyOptions(o, Set.of(
                        "--png", "--msgFile", "--out",
                        "--encKey", "--xorKey", "--stcKey",
                        "--encKeyEnv", "--xorKeyEnv", "--stcKeyEnv",
                        "--minQf", "--maxQf", "--selectionPolicy",
                        "--stcHeight",
                        "--geometryMode",
                        "--auditMode", "--auditProfile", "--auditEnabled"));
                Path png = Path.of(req(o, "--png"));
                Path message = Path.of(req(o, "--msgFile"));
                try (PairConsoleContext.Scope pairScope =
                             PairConsoleContext.singlePair(png, message)) {
                    embedSelectedGeometries(
                            png,
                            message,
                            Path.of(req(o, "--out")),
                            keyMaterial(o),
                            qfRange(o),
                            selectionConstraints(o),
                            embeddingStcHeight(o),
                            auditMode(o),
                            auditProfile(o),
                            auditEnabled(o),
                            geometrySelection(o));
                }
            }

            case "batch-embed" -> {
                requireOnlyOptions(o, Set.of(
                        "--pngDir", "--msgDir", "--outDir",
                        "--recursive", "--resume", "--continueOnError",
                        "--startIndex", "--pairModulo", "--pairRemainder",
                        "--shardUnit", "--manifest",
                        "--messageExtension", "--onlyMessageName",
                        "--encKey", "--xorKey", "--stcKey",
                        "--encKeyEnv", "--xorKeyEnv", "--stcKeyEnv",
                        "--minQf", "--maxQf", "--selectionPolicy",
                        "--stcHeight",
                        "--geometryMode",
                        "--auditMode", "--auditProfile", "--auditEnabled",
                        "--workers", "--carrierParallelism", "--cpuSlots",
                        "--threadMode", "--cpuAffinity",
                        "--deleteJpegsAfterEvaluation"));
                batchEmbed(o);
            }

            case "extract" -> {
                requireOnlyOptions(o, Set.of(
                        "--png", "--stego", "--outFile",
                        "--encKey", "--xorKey", "--stcKey",
                        "--encKeyEnv", "--xorKeyEnv", "--stcKeyEnv",
                        "--minQf", "--maxQf", "--stcHeight",
                        "--geometryMode"));
                Path png = Path.of(req(o, "--png"));
                try (PairConsoleContext.Scope extractionScope =
                             PairConsoleContext.extraction(png)) {
                    extract(
                            png,
                            Path.of(req(o, "--stego")),
                            keyMaterial(o),
                            qfRange(o),
                            extractionStcHeights(o),
                            geometrySelection(o),
                            Path.of(req(o, "--outFile")));
                }
            }

            case "summarize" -> {
                requireOnlyOptions(o, Set.of(
                        "--runs", "--positions", "--channels",
                        "--channelPositions", "--attempts", "--outDir"));
                ExperimentSummaryGenerator.generate(
                        Path.of(o.getOrDefault("--runs", RUN_CSV.toString())),
                        Path.of(o.getOrDefault("--positions", POSITION_CSV.toString())),
                        Path.of(o.getOrDefault("--channels", CHANNEL_CSV.toString())),
                        Path.of(o.getOrDefault("--channelPositions", CHANNEL_POSITION_CSV.toString())),
                        Path.of(o.getOrDefault("--attempts", ATTEMPT_CSV.toString())),
                        Path.of(o.getOrDefault("--outDir", thesis.rap.runtime.RuntimePaths.results("experiment_summary").toString())));
            }

            case "detector-summary" -> {
                requireOnlyOptions(o, Set.of(
                        "--csv", "--labelColumn", "--scoreColumn", "--out"));
                DetectorMetricsAnalyzer.analyze(
                        Path.of(req(o, "--csv")),
                        o.getOrDefault("--labelColumn", "label"),
                        o.getOrDefault("--scoreColumn", "score"),
                        Path.of(o.getOrDefault("--out", thesis.rap.runtime.RuntimePaths.results("detector_summary.csv").toString()))
                ).print();
            }

            default -> throw new IllegalArgumentException(
                    "Unknown mode: " + mode
                            + ". Permitted modes are embed, batch-embed, extract, "
                            + "summarize, and detector-summary.");
            }
        }
    }

    private static void embedSelectedGeometries(
            Path png,
            Path msg,
            Path requestedOut,
            String encodedKeys,
            QfRange qfRange,
            AdaptiveQFSelector.SelectionConstraints selectionConstraints,
            int stcHeight,
            AuditMode auditMode,
            AuditProfile auditProfile,
            boolean auditEnabled,
            GeometrySelection geometrySelection
    ) throws Exception {
        int verified = 0;
        Map<SamplingGeometry, Integer> selectedQfs =
                new LinkedHashMap<>();
        List<String> unverifiedProfiles = new ArrayList<>();
        List<String> unexpectedFailures = new ArrayList<>();
        Throwable firstUnexpectedFailure = null;
        for (SamplingGeometry geometry : geometrySelection.geometries()) {
            Path geometryOutput = outputForGeometry(
                    requestedOut, geometrySelection, geometry);
            System.out.printf(
                    "GEOMETRY_PROFILE_START mode=%s geometry=%s output=%s%n",
                    geometrySelection.token(), geometry.label(),
                    geometryOutput.toAbsolutePath());
            Integer selectedQf;
            try {
                selectedQf = embed(
                        png, msg, geometryOutput, encodedKeys, qfRange,
                        selectionConstraints, stcHeight, auditMode,
                        auditProfile, auditEnabled, geometrySelection.singleModeFor(geometry), geometry);
            } catch (Exception | LinkageError failure) {
                if (!geometrySelection.isMultiple()) throw failure;
                if (firstUnexpectedFailure == null) {
                    firstUnexpectedFailure = failure;
                }
                String failureText = geometry.label() + "="
                        + failure.getClass().getSimpleName() + ":"
                        + safeMessage(failure);
                unexpectedFailures.add(failureText);
                selectedQfs.put(geometry, null);
                System.err.printf(
                        "GEOMETRY_RESULT geometry=%s selected_qf=none "
                                + "verified_output=false "
                                + "status=UNEXPECTED_FAILURE detail=%s%n",
                        geometry.label(), failureText);
                continue;
            }
            selectedQfs.put(geometry, selectedQf);
            boolean produced = selectedQf != null
                    && Files.isRegularFile(geometryOutput)
                    && Files.size(geometryOutput) > 0;
            if (produced) {
                verified++;
            } else {
                unverifiedProfiles.add(geometry.label());
            }
            System.out.printf(
                    "GEOMETRY_PROFILE_END mode=%s geometry=%s status=%s output=%s%n",
                    geometrySelection.token(), geometry.label(),
                    produced ? "VERIFIED_SUCCESS" : "NO_VERIFIED_QF",
                    geometryOutput.toAbsolutePath());
            System.out.printf(
                    "GEOMETRY_RESULT geometry=%s selected_qf=%s "
                            + "verified_output=%s%n",
                    geometry.label(),
                    selectedQf == null ? "none" : selectedQf,
                    produced);
        }
        if (geometrySelection.isMultiple()) {
            System.out.printf(
                    "GEOMETRY_SELECTION_SUMMARY verified_geometries=%d "
                            + "requested_geometries=%d selected_qfs=%s%n",
                    verified, geometrySelection.geometries().size(),
                    selectedQfs.entrySet().stream()
                            .map(entry -> entry.getKey().label() + "="
                                    + (entry.getValue() == null
                                    ? "none" : entry.getValue()))
                            .toList());
            if (!unexpectedFailures.isEmpty()) {
                throw new IllegalStateException(
                        "Geometry selection completed every geometry, but "
                                + "one or more profiles had unexpected "
                                + "failures: " + unexpectedFailures,
                        firstUnexpectedFailure);
            }
        }
        if (!unverifiedProfiles.isEmpty()) {
            throw new IllegalStateException(
                    "Every selected geometry completed its inclusive QF search, "
                            + "but no verified output was found through maxQf for: "
                            + unverifiedProfiles);
        }
    }

    private static Integer embed(
            Path png,
            Path msg,
            Path out,
            String encodedKeys,
            QfRange qfRange,
            AdaptiveQFSelector.SelectionConstraints selectionConstraints,
            int stcHeight,
            AuditMode auditMode,
            AuditProfile auditProfile,
            boolean auditEnabled,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry
    ) throws Exception {
        String runId = UUID.randomUUID().toString();
        EmbedRunState runState = new EmbedRunState();
        runState.auditMode = auditMode;
        runState.auditProfile = auditProfile;
        runState.auditEnabled = auditEnabled;
        runState.stcHeight = stcHeight;
        runState.geometryMode = geometryMode;
        runState.geometry = geometry;
        runState.finalQf.configureStcHeight(stcHeight);
        System.setProperty(
                ThreeStateEmbeddingEngine.VERBOSE_ATTEMPTS_PROPERTY,
                Boolean.toString(
                        auditEnabled
                                && auditMode.writesQfForensics()
                                && auditProfile == AuditProfile.FULL_AUDIT));
        try {
            return embedInternalConformant(
                    runId, png, msg, out, encodedKeys, qfRange,
                    selectionConstraints, runState);
        } catch (Exception | LinkageError e) {
            String failureCode = e instanceof java.io.IOException
                    ? "FAILED_" + runState.operation : failureReasonCode(e);
            String failureDetail = e.getClass().getSimpleName() + ": "
                    + safeMessage(e);
            runState.failure = e;
            FailureDiagnostics.log(e);
            runState.finalQf.markRunFailure(
                    failureCode, failureDetail);
            if (runState.selectedReference != null) {
                try {
                    Files.deleteIfExists(runState.selectedReference);
                    runState.selectedReference = null;
                } catch (Exception cleanupError) {
                    e.addSuppressed(cleanupError);
                }
            }
            if (runState.outputCommitted && runState.outputEvidence == null) {
                try {
                    Files.deleteIfExists(out);
                    VerifiedOutputManifest.delete(out);
                } catch (Exception cleanupError) {
                    e.addSuppressed(cleanupError);
                }
            }
            if (runState.outputCommitted && runState.outputEvidence != null) {
                System.err.println("VERIFIED_OUTPUT_PRESERVED finalization_failure=" + failureCode);
            }
            if (runState.auditEnabled
                    && runState.auditMode.writesQfForensics()
                    && !runState.auditPublished
                    && runState.profile != null
                    && runState.payloadLayout != null) {
                try {
                    if (runState.inputEvidence != null) {
                        verifyProductionInputStability(
                                png, msg, runState.inputEvidence);
                    }
                    RunAuditWriter.writeFailure(
                            runId, APPLICATION_VERSION,
                            runState.startedUtc, png, msg,
                            runState.profile, runState.payloadLayout,
                            runState.qfRows, runState.balanceAttempts,
                            runState.stcHeight,
                            failureCode,
                            failureDetail,
                            runState.auditProfile,
                            runState.geometryMode,
                            runState.geometry);
                    runState.auditPublished = true;
                } catch (Exception auditError) {
                    e.addSuppressed(auditError);
                }
            }
            if (runState.auditEnabled
                    && runState.auditMode.writesQfForensics()) {
                try {
                    AttemptCsvWriter.append(ATTEMPT_CSV, runId,
                            safeFileName(png), safeFileName(msg),
                            runState.geometryMode,
                            runState.geometry,
                            runState.stcHeight,
                            qfRange.minQf(), qfRange.maxQf(),
                            -1, null, failureCode,
                            failureDetail);
                } catch (Exception ledgerError) {
                    e.addSuppressed(ledgerError);
                }
            }
            if (runState.auditEnabled
                    && runState.auditMode == AuditMode.RESULT_ONLY
                    && !runState.resultAuditPublished) {
                try {
                    appendResultAudit(
                            runId, png, msg, out, qfRange,
                            selectionConstraints, runState, "FAILED",
                            failureCode, failureDetail, null,
                            "RUN_FAILED_BEFORE_COMPLETE_FINAL_QF_EVALUATION");
                    runState.resultAuditPublished = true;
                } catch (Exception ledgerError) {
                    e.addSuppressed(ledgerError);
                }
            }
            System.out.println("Embedding success: false");
            System.out.println("Failure reason code: " + failureCode);
            if (runState.auditEnabled
                    && runState.auditMode == AuditMode.RESULT_ONLY) {
                System.out.println(runState.finalQf.consoleSummary());
                System.out.println("Result audit ledger: "
                        + ResultAuditWriter.DEFAULT_CSV.toAbsolutePath());
            }
            System.out.println(
                    "===== RAP EMBEDDING END status=FAILED "
                            + "selectedQF="
                            + (runState.finalQf.qf() == null
                            ? "none" : runState.finalQf.qf())
                            + " =====");
            throw e;
        }
    }

    private static Integer embedInternalConformant(
            String runId,
            Path png,
            Path msg,
            Path out,
            String encodedKeys,
            QfRange qfRange,
            AdaptiveQFSelector.SelectionConstraints selectionConstraints,
            EmbedRunState runState
    ) throws Exception {
        int stcHeight = runState.stcHeight;
        StcCarrierPlan.validateSupportedHeight(stcHeight);
        SamplingGeometry geometry = java.util.Objects.requireNonNull(
                runState.geometry, "run sampling geometry");
        SamplingGeometryMode geometryMode = java.util.Objects.requireNonNull(
                runState.geometryMode, "run geometry mode");
        System.out.println("===== RAP EMBEDDING START =====");
        System.out.println("Run ID: " + runId);
        System.out.println("Start UTC: " + runState.startedUtc);
        System.out.println("Input image: " + png.toAbsolutePath());
        System.out.println("Message file: " + msg.toAbsolutePath());
        System.out.println("Requested output: " + out.toAbsolutePath());
        System.out.println("Geometry mode: " + geometryMode.token());
        System.out.println("Sampling geometry: " + geometry.label()
                + " | signature=0x" + geometry.signatureHex());
        System.out.println("Audit enabled: " + runState.auditEnabled);
        System.out.println(
                "Audit mode: "
                        + (runState.auditEnabled
                        ? runState.auditMode.name() : "DISABLED"));
        if (runState.auditEnabled
                && runState.auditMode.writesQfForensics()) {
            System.out.println(
                    "QF forensic detail profile: "
                            + runState.auditProfile.name());
        } else if (runState.auditEnabled) {
            System.out.println(
                    "Result audit scope: selected/last QF plus final-QF "
                            + "run and Cb/Cr channel aggregates");
            System.out.println("Result audit ledger: "
                    + ResultAuditWriter.DEFAULT_CSV.toAbsolutePath());
        }
        validateReadableFile(png, "Input PNG");
        // An empty semantic message is valid: the authenticated frame still
        // contains nine preprocessing bytes and the 26-byte compact header.
        validateReadableFile(msg, "Message file", true);
        VerifiedProductionInput verifiedInput =
                verifyProductionInputs(png, msg);
        ProductionInputEvidence inputEvidence = verifiedInput.evidence();
        runState.inputEvidence = inputEvidence;
        System.out.printf(
                "INPUT_FILE_VERIFIED source=filesystem image_format=PNG "
                        + "image_bytes=%d width=%d height=%d image_sha256=%s "
                        + "message_bytes=%d message_sha256=%s%n",
                inputEvidence.imageBytes(), inputEvidence.width(),
                inputEvidence.height(), inputEvidence.imageSha256(),
                inputEvidence.messageBytes(), inputEvidence.messageSha256());
        System.out.printf(
                "NATIVE_JPEG_BACKEND_VERIFIED libjpeg_version=%d "
                        + "java_coefficient_fallback=false "
                        + "native_stc_required=true input_origin=UNATTESTED%n",
                inputEvidence.libjpegVersion());
        // Cost-model provenance is part of the method, not a diagnostic: the
        // impact-window alignment changes every candidate cost and therefore the
        // frozen carrier order, so it is printed in every audit mode.
        System.out.printf(
                "COST_MODEL_PROVENANCE uniward_impact_window=%s offset=%d "
                        + "sigma=2^-6 si_offset=1e-4 wet_sentinel=1e13 "
                        + "band=%s statistical_model=%s "
                        + "normalization=joint_cbcr_qf_p95 weights=%.2f/%.2f/%.2f/%.2f "
                        + "geometry_mode=%s sampling_geometry=%s "
                        + "stc_height=%d stc_numeric_profile=%s "
                        + "header_reserve=%d body_width_cap=%d%n",
                thesis.rap.alg.SiUniwardCostEngine.IMPACT_WINDOW_ALIGNMENT,
                thesis.rap.alg.SiUniwardCostEngine.IMPACT_WINDOW_OFFSET,
                JpegMath.targetBandLabel(),
                thesis.rap.alg.StatisticalTransitionCostModel.MODEL_VERSION,
                thesis.rap.alg.StatisticalTransitionCostModel.MSE_WEIGHT,
                thesis.rap.alg.StatisticalTransitionCostModel.SSIM_WEIGHT,
                thesis.rap.alg.StatisticalTransitionCostModel.KL_WEIGHT,
                thesis.rap.alg.StatisticalTransitionCostModel.ENTROPY_WEIGHT,
                geometryMode.token(), geometry.label(),
                stcHeight,
                OriginalStcCodec.NUMERIC_PROFILE,
                StcCarrierPlan.DEFAULT_HEADER_RESERVE_BITS,
                StcCarrierPlan.bodyWidthCap(stcHeight));
        boolean cudaPermitted = ExecutionBackend.cudaPermitted();
        boolean hipPermitted = ExecutionBackend.hipPermitted();
        boolean cudaAvailable = cudaPermitted
                && thesis.rap.alg.SiUniwardCostEngine.cudaAvailable();
        boolean hipAvailable = hipPermitted
                && thesis.rap.alg.SiUniwardCostEngine.hipAvailable();
        System.out.printf(
                "SI_ACCELERATION execution_policy=%s configured_backend=%s hip_available=%s "
                        + "hip_device=%s cuda_available=%s cuda_device=%s gpu_slots=%s%n",
                ExecutionBackend.description(),
                thesis.rap.alg.SiUniwardCostEngine.configuredBackendName(),
                hipAvailable,
                hipAvailable
                        ? thesis.rap.alg.SiUniwardCostEngine.hipDeviceName().replace(' ', '_')
                        : "disabled_or_unavailable",
                cudaAvailable,
                cudaAvailable
                        ? thesis.rap.alg.SiUniwardCostEngine.cudaDeviceName().replace(' ', '_')
                        : "disabled_or_unavailable",
                System.getProperty(
                        "rap.gpu.slots",
                        System.getenv().getOrDefault("RAP_GPU_SLOTS", "1")));
        ensureParentDirectory(out);
        if (Files.exists(out)
                || Files.exists(VerifiedOutputManifest.sidecarFor(out))) {
            throw new IllegalArgumentException(
                    "Refusing to overwrite an existing output or "
                            + "verification sidecar: "
                            + out.toAbsolutePath());
        }
        KeyMaterial.decode(encodedKeys);

        long embedStartNanos = System.nanoTime();
        runState.embeddingStartNanos = embedStartNanos;
        byte[] message = Files.readAllBytes(msg);
        PayloadPackage.PreparedMessage preparedMessage =
                PayloadPackage.prepareMessage(message);
        PayloadPackage.PayloadLayout estimatedLayout =
                PayloadPackage.inspectLayout(preparedMessage);
        SourceImageContext sourceContext = TrueParallelStateCache.sharedSourceContext(
                png, verifiedInput.sourceContext());
        CandidateGenerator.Context candidateContext =
                new CandidateGenerator.Context(sourceContext);
        if (TrueParallelStateCache.enabled()) {
            System.out.printf(
                    "TRUE_PARALLEL_PREPARED_CACHE enabled=true max_states=%d directory=%s%n",
                    TrueParallelStateCache.maximumStates(),
                    TrueParallelStateCache.cacheDirectory());
        }
        runState.payloadLayout = estimatedLayout;
        int requiredBits = estimatedLayout.totalPayloadBits();
        ThreeStateEmbeddingEngine.Profile profile =
                ThreeStateEmbeddingEngine.Profile.configured();
        runState.profile = profile;
        List<QfBandCoefficientReport.Row> qfRows = runState.qfRows;
        List<ThreeStateEmbeddingEngine.Attempt> balanceAttempts =
                runState.balanceAttempts;
        StcPipelineIntegration stc = new StcPipelineIntegration(stcHeight);
        if (TrueParallelStateCache.enabled()) {
            TrueParallelStateCache.prefetchAhead(
                    png, sourceContext, candidateContext,
                    qfRange.minQf(), qfRange.maxQf(), geometry);
        }

        AdaptiveQfSearch.Result<EmbeddingTrialResult> search =
                AdaptiveQfSearch.firstVerified(
                        qfRange.minQf(), qfRange.maxQf(),
                        qf -> attemptEmbeddingAtQf(
                                runId, png, out, encodedKeys, message,
                                preparedMessage, sourceContext, candidateContext,
                                requiredBits, qf, qfRange.maxQf(), profile, stc,
                                geometryMode, geometry,
                                qfRows, balanceAttempts,
                                runState.auditEnabled
                                        && runState.auditMode
                                        .writesQfForensics(),
                                runState.auditEnabled
                                        && runState.auditMode
                                        == AuditMode.RESULT_ONLY
                                        ? runState.finalQf : null));

        if (runState.auditEnabled
                && runState.auditMode.writesQfForensics()
                && runState.auditProfile == AuditProfile.FULL_AUDIT) {
            Path qfBandCsv = writeAdaptiveQfBandCsv(
                    png, msg, runId, qfRows);
            System.out.println(
                    "Expanded QF trial report CSV: "
                            + qfBandCsv.toAbsolutePath());
        }

        if (!search.verified()) {
            runState.embeddingElapsedMillis =
                    nanosToMillis(System.nanoTime() - embedStartNanos);
            verifyProductionInputStability(
                    png, msg, inputEvidence);
            Path auditDirectory = null;
            if (runState.auditEnabled
                    && runState.auditMode.writesQfForensics()) {
                auditDirectory = RunAuditWriter.writeFailure(
                        runId, APPLICATION_VERSION, runState.startedUtc,
                        png, msg, profile, estimatedLayout,
                        qfRows, balanceAttempts,
                        stcHeight,
                        "NO_VERIFIED_QF",
                        "No QF passed the complete fixed-QF kernel "
                                + "and authenticated verification",
                        runState.auditProfile,
                        geometryMode, geometry);
                runState.auditPublished = true;
                AttemptCsvWriter.append(
                        ATTEMPT_CSV, runId,
                        png.getFileName().toString(),
                        msg.getFileName().toString(),
                        geometryMode, geometry,
                        stcHeight,
                        qfRange.minQf(), qfRange.maxQf(),
                        requiredBits, null, "NO_VERIFIED_QF",
                        "No QF passed capacity, fixed-QF embedding, balance, "
                                + "write-back, authenticated extraction, and byte equality");
            } else if (runState.auditEnabled
                    && runState.auditMode == AuditMode.RESULT_ONLY) {
                appendResultAudit(
                        runId, png, msg, out, qfRange,
                        selectionConstraints, runState,
                        "UNABLE_TO_EMBED_OR_EXTRACT",
                        "NO_VERIFIED_QF",
                        "No QF passed capacity, the fixed-QF kernel, "
                                + "balance, coefficient write-back, "
                                + "authenticated extraction, and exact "
                                + "message equality. The terminal-QF "
                                + "columns describe only the last tested QF.",
                        null, "NO_VERIFIED_QF_HAS_NO_STEGO_EVALUATION");
                runState.resultAuditPublished = true;
            }
            System.out.println("Embedding success: false");
            System.out.println("Extraction success: false");
            System.out.println("AES-GCM verified: false");
            if (auditDirectory != null) {
                System.out.println("Audit directory: "
                        + auditDirectory.toAbsolutePath());
            } else if (runState.auditEnabled) {
                System.out.println("Result audit ledger: "
                        + ResultAuditWriter.DEFAULT_CSV.toAbsolutePath());
            } else {
                System.out.println(
                        "Audit output: disabled by --auditEnabled=false");
            }
            if (runState.auditEnabled
                    && runState.auditMode == AuditMode.RESULT_ONLY) {
                System.out.println("Failure reason code: NO_VERIFIED_QF");
                System.out.println(runState.finalQf.consoleSummary());
                System.out.println(
                        "===== RAP EMBEDDING END "
                                + "status=UNABLE_TO_EMBED_OR_EXTRACT "
                                + "lastTestedQF="
                                + (runState.finalQf.qf() == null
                                ? "none" : runState.finalQf.qf())
                                + " =====");
            } else {
                System.out.println(
                        "===== RAP EMBEDDING END status=NO_VERIFIED_QF "
                                + "selectedQF=none =====");
            }
            return null;
        }

        EmbeddingTrialResult selected = search.value();
        int selectedQf = selected.qf();
        Path referenceJpeg = selected.referenceJpeg();
        runState.selectedReference = referenceJpeg;
        runState.operation = "OUTPUT_COMMIT";
        atomicReplace(selected.stagedStego(), out);
        runState.outputCommitted = true;
        runState.embeddingSucceeded = true;

        FixedQfEmbeddingKernel.Result kernel = selected.kernel();
        CandidateSet candidateSet = kernel.candidateSet();
        List<Candidate> dryCandidates = candidateSet.dryBaseCostSorted();
        List<Candidate> activeCandidates = kernel.activeCandidates();
        StcCarrierPlan carrierPlan = kernel.carrierPlan();
        ChannelAllocationEstimate allocation =
                kernel.channelAllocation();
        ThreeStateEmbeddingEngine.Result balanced = kernel.balance();
        PayloadPackage.PayloadLayout payloadLayout =
                selected.packedPayload().layout();
        int payloadBits = payloadLayout.totalPayloadBits();
        int engineeringCapacity =
                carrierPlan.engineeringMessageCapacityBits();
        StcRunMetrics stcMetrics = new StcRunMetrics(
                stcHeight, carrierPlan,
                balanced.header().trace(), balanced.body().trace());
        long embeddingElapsedMillis =
                nanosToMillis(System.nanoTime() - embedStartNanos);
        runState.embeddingElapsedMillis = embeddingElapsedMillis;

        if (runState.auditEnabled
                && runState.auditMode.writesQfForensics()) {
            System.out.println("Adaptive QF selected: " + selectedQf);
            System.out.println(
                    "Carrier universe: every Cb/Cr address, including zero, in "
                            + JpegMath.targetBandLabel());
            System.out.println(
                    "Dry order: minimum base directional cost, component, block, natural index; "
                            + "HMAC ordering only within frozen 64-candidate windows");
            System.out.println(
                    "Body cap: min(Nq-" + carrierPlan.headerCarrierCount() + ", "
                            + carrierPlan.bodyWidthCap()
                            + "*mB); STC height=" + stcHeight
                            + "; aggregate sequential native path bytes="
                            + carrierPlan.nativePathBytesEstimate()
                            + " bytes; binary64 metric-array bytes="
                            + carrierPlan.nativeMetricBytesEstimate()
                            + " bytes per call");
            System.out.printf(
                    java.util.Locale.US,
                    "Balance profile=%s | attempts=%d | delta h(0)=%+d | "
                            + "final balance parameter=%.9f | scale=%.9f%n",
                    balanced.profile().code(), balanced.iterations(),
                    balanced.zeroBinDelta(), balanced.finalMultiplier(),
                    balanced.scale());
            System.out.println("Payload bits: " + payloadBits);
            System.out.println("Universe addresses: "
                    + candidateSet.universeCount());
            System.out.println("Dry candidates: " + candidateSet.dryCount());
            System.out.println("Fully wet addresses: "
                    + candidateSet.fullyWetCount());
            System.out.println("Active STC carriers: "
                    + activeCandidates.size());
            System.out.println("RAP engineering capacity C_eng: "
                    + engineeringCapacity);
            System.out.println("Reference JPEG: "
                    + referenceJpeg.toAbsolutePath());
            System.out.println("Output stego JPEG: "
                    + out.toAbsolutePath());
            System.out.printf(
                    java.util.Locale.US,
                    "Cb/Cr allocation evidence: active=%.6f/%.6f | "
                            + "idealized_entropy_contribution=%.6f/%.6f | "
                            + "actual_modifications=%.6f/%.6f%n",
                    allocation.activeCbShare(), allocation.activeCrShare(),
                    allocation.cbEntropyContributionShare(),
                    allocation.crEntropyContributionShare(),
                    allocation.actualModifiedCbShare(),
                    allocation.actualModifiedCrShare());
        }

        VerificationResult verification = selected.verification();
        if (!verification.extractionSuccess()
                || !verification.aesGcmVerified()
                || !verification.messageMatches()) {
            Files.deleteIfExists(out);
            VerifiedOutputManifest.delete(out);
            Files.deleteIfExists(referenceJpeg);
            throw new VerificationFailure(
                    "Selected trial lost verification after finalization: "
                            + verification.detail());
        }

        /*
         * The verification sidecar is operational integrity state, not an
         * optional statistical audit. It is therefore retained in both audit
         * modes and is required before --resume may trust an existing output.
         */
        ProductionOutputEvidence outputEvidence =
                verifyCommittedProductionOutput(
                        out, png, msg, inputEvidence);
        runState.outputEvidence = outputEvidence;
        runState.operation = "VERIFIED_MARKER_WRITE";
        VerifiedOutputManifest.write(
                out, png, msg, qfRange.minQf(), qfRange.maxQf(),
                APPLICATION_VERSION, stcHeight,
                geometryMode, geometry);
        System.out.printf(
                "OUTPUT_FILE_VERIFIED stego_bytes=%d width=%d height=%d "
                        + "components=%d stego_sha256=%s input_stable=true "
                        + "coefficient_writeback=true native_stc_round_trip=true "
                        + "aes_gcm=true exact_message=true%n",
                outputEvidence.stegoBytes(), outputEvidence.width(),
                outputEvidence.height(), outputEvidence.components(),
                outputEvidence.stegoSha256());

        if (!runState.auditEnabled) {
            cleanupSelectedReference(referenceJpeg, runState);
            System.out.println(
                    "Audit/evaluation artifacts: disabled by --auditEnabled=false");
            System.out.println(
                    "Core checks retained: coefficient write-back equality, "
                            + "STC syndrome extraction, AES-GCM/AAD authentication, "
                            + "and exact recovered-message equality.");
            System.out.printf(
                    "SELECTED_QF=%d payload_bits=%d dry_candidates=%d "
                            + "active_carriers=%d engineering_capacity_bits=%d "
                            + "balance_profile=%s balance_attempts=%d delta_h0=%+d%n",
                    selectedQf, payloadBits, dryCandidates.size(),
                    activeCandidates.size(), engineeringCapacity,
                    balanced.profile().code(), balanced.iterations(),
                    balanced.zeroBinDelta());
            System.out.println("End UTC: " + Instant.now());
            System.out.println("Run ID: " + runId);
            System.out.println(
                    "===== RAP EMBEDDING END status=VERIFIED_SUCCESS "
                            + "selectedQF=" + selectedQf + " =====");
            return selectedQf;
        }

        runState.operation = "EVALUATION_OR_AUDIT_WRITE";
        if (runState.auditMode == AuditMode.RESULT_ONLY) {
            long resultEvaluationStartNanos = System.nanoTime();
            EvaluationSummary finalQfEvaluation =
                    FullEvaluationAnalyzer.evaluate(
                            png.toString(), referenceJpeg.toString(),
                            out.toString(), payloadBits, selectedQf,
                            dryCandidates, activeCandidates,
                            engineeringCapacity, stcMetrics,
                            true, true, true, true);
            runState.evaluationElapsedMillis = nanosToMillis(
                    System.nanoTime() - resultEvaluationStartNanos);
            appendResultAudit(
                    runId, png, msg, out, qfRange,
                    selectionConstraints, runState,
                    "VERIFIED_SUCCESS", "", "OK",
                    finalQfEvaluation, "");
            runState.resultAuditPublished = true;
            cleanupSelectedReference(referenceJpeg, runState);
            System.out.println("Embedding success: true");
            System.out.println("Extraction success: true");
            System.out.println("Exact message verified: true");
            System.out.printf(
                    java.util.Locale.US,
                    "FINAL_QF_EVALUATION selected_qf=%d modified=%d "
                            + "reference_stego_psnr=%.9f "
                            + "reference_stego_ssim=%.12f "
                            + "target_band_js=%.12g "
                            + "stc_weighted_distortion=%.12g%n",
                    selectedQf,
                    finalQfEvaluation.coefficientReport
                            .modifiedCoefficients,
                    finalQfEvaluation.referenceVsStego.psnr,
                    finalQfEvaluation.referenceVsStego.ssim,
                    finalQfEvaluation.coefficientReport
                            .targetBandJensenShannon,
                    finalQfEvaluation.coefficientReport
                            .stcWeightedDistortion);
            System.out.printf(
                    "SELECTED_QF=%d payload_bits=%d dry_candidates=%d "
                            + "active_carriers=%d engineering_capacity_bits=%d "
                            + "balance_profile=%s balance_attempts=%d delta_h0=%+d%n",
                    selectedQf, payloadBits, dryCandidates.size(),
                    activeCandidates.size(), engineeringCapacity,
                    balanced.profile().code(), balanced.iterations(),
                    balanced.zeroBinDelta());
            System.out.println(runState.finalQf.consoleSummary());
            System.out.println("Result audit ledger: "
                    + ResultAuditWriter.DEFAULT_CSV.toAbsolutePath());
            System.out.println("End UTC: " + Instant.now());
            System.out.println("Run ID: " + runId);
            System.out.println(
                    "===== RAP EMBEDDING END status=VERIFIED_SUCCESS "
                            + "selectedQF=" + selectedQf + " =====");
            return selectedQf;
        }

        long evaluationStartNanos = System.nanoTime();
        EvaluationSummary eval = FullEvaluationAnalyzer.evaluate(
                png.toString(),
                referenceJpeg.toString(),
                out.toString(),
                payloadBits,
                selectedQf,
                dryCandidates,
                activeCandidates,
                engineeringCapacity,
                stcMetrics,
                true,
                true,
                true,
                true);
        long evaluationElapsedMillis =
                nanosToMillis(System.nanoTime() - evaluationStartNanos);

        if (runState.auditProfile == AuditProfile.FULL_AUDIT) {
            eval.print();
        } else {
            System.out.printf(
                    java.util.Locale.US,
                    "Compact evaluation: QF=%d payload_bits=%d "
                            + "modified=%d ref_stego_psnr=%.9f "
                            + "ref_stego_ssim=%.12f%n",
                    selectedQf, payloadBits,
                    eval.coefficientReport.modifiedCoefficients,
                    eval.referenceVsStego.psnr,
                    eval.referenceVsStego.ssim);
            System.out.println(
                    "Numeric fidelity values are diagnostics, "
                            + "not security grades.");
        }
        Files.createDirectories(thesis.rap.runtime.RuntimePaths.results());
        AdaptiveQFSelector.QFDecision decision =
                new AdaptiveQFSelector.QFDecision(
                        selectedQf, dryCandidates.size(),
                        engineeringCapacity, true, qfRows);
        QfSelectionMetrics qfSelectionMetrics = qfSelectionMetrics(
                decision, payloadBits, dryCandidates.size(),
                engineeringCapacity);
        JpegMetadata jpegMetadata = eval.jpegMetadata;
        if (runState.auditProfile == AuditProfile.FULL_AUDIT) {
            printPayloadBreakdown(payloadLayout);
            printJpegMetadata(jpegMetadata);
            printQfSelectionMargin(qfSelectionMetrics);
        } else {
            System.out.printf(
                    "Payload bytes=%d | fixed framing=%d | "
                            + "net expansion=%+d | JPEG profile=%s | "
                            + "QFs tested=%d%n",
                    payloadLayout.totalPayloadBytes(),
                    payloadLayout.fixedFramingOverheadBytes(),
                    payloadLayout.netPayloadExpansionBytes(),
                    jpegMetadata.subsampling(),
                    qfSelectionMetrics.qfsTested());
        }

        RunReproducibility reproducibility = new RunReproducibility(
                runState.startedUtc,
                APPLICATION_VERSION,
                FileFingerprint.sha256(png),
                FileFingerprint.sha256(msg),
                FileFingerprint.sha256(referenceJpeg),
                FileFingerprint.sha256(out),
                System.getProperty("java.version", "unknown"),
                System.getProperty("os.name", "unknown"),
                System.getProperty("os.arch", "unknown"),
                JpegCore.libjpegVersion(),
                ImageUtil.jpegWriterDescription()
                        + "; canonical coefficient-derived JPEG reference; "
                        + "native JPEG coefficient I/O");

        EvaluationRunContext ctx = new EvaluationRunContext(
                runId,
                png.getFileName().toString(),
                msg.getFileName().toString(),
                geometryMode.token(),
                geometry.label(),
                qfRange.minQf(),
                qfRange.maxQf(),
                selectionConstraints.policyName(),
                selectionConstraints.maxCandidateLoading(),
                selectionConstraints.maxStcUtilization(),
                payloadLayout,
                jpegMetadata,
                qfSelectionMetrics,
                reproducibility,
                embeddingElapsedMillis,
                evaluationElapsedMillis);
        TransitionCounts transitions =
                countTransitions(candidateSet.universe(), kernel.intendedStego());
        Path runJson = thesis.rap.runtime.RuntimePaths.results(
                "evaluation_json", runId + ".json");
        runState.operation = "AUDIT_WRITE";
        Path auditDirectory = RunAuditWriter.writeSuccess(
                runId, APPLICATION_VERSION, runState.startedUtc,
                png, msg, referenceJpeg, out, profile,
                payloadLayout, selectedQf, qfRows,
                balanceAttempts, kernel, runState.auditProfile,
                geometryMode, geometry);
        runState.auditPublished = true;

        boolean aggregateLogsPublished = true;
        try {
            if (runState.auditProfile.writesExpandedAggregateTables()) {
                EvaluationCsvWriter.append(
                        RUN_CSV, POSITION_CSV, CHANNEL_CSV,
                        CHANNEL_POSITION_CSV, ctx, eval);
            } else {
                EvaluationCsvWriter.appendRun(RUN_CSV, ctx, eval);
            }
            EvaluationJsonWriter.write(runJson, ctx, eval);
            AttemptCsvWriter.append(
                    ATTEMPT_CSV, runId,
                    png.getFileName().toString(),
                    msg.getFileName().toString(),
                    geometryMode, geometry,
                    stcHeight,
                    qfRange.minQf(), qfRange.maxQf(),
                    payloadBits, selectedQf, "SUCCESS", "");
        } catch (Exception aggregateFailure) {
            aggregateLogsPublished = false;
            System.err.println(
                    "Aggregate-log publication warning: "
                            + safeMessage(aggregateFailure));
            System.err.println(
                    "The atomic six-file audit bundle remains the authoritative "
                            + "verified run record.");
        }

        System.out.println("========== THREE-STATE TRANSITION AUDIT ==========");
        System.out.printf(
                "0->+1=%d | 0->-1=%d | +1->0=%d | -1->0=%d | "
                        + "other +1=%d | other -1=%d%n",
                transitions.zeroToPlus(), transitions.zeroToMinus(),
                transitions.plusToZero(), transitions.minusToZero(),
                transitions.otherPlus(), transitions.otherMinus());
        System.out.printf(
                "zero activations=%d | zero deactivations=%d | "
                        + "net delta h(0)=%+d%n",
                transitions.activations(), transitions.deactivations(),
                transitions.zeroBinDelta());
        System.out.println("==================================================");
        cleanupSelectedReference(referenceJpeg, runState);
        if (aggregateLogsPublished) {
            System.out.println("Run-level evaluation CSV updated: "
                    + RUN_CSV.toAbsolutePath());
            System.out.println("Structured per-run JSON written: "
                    + runJson.toAbsolutePath());
        }
        System.out.println("Audit directory: "
                + auditDirectory.toAbsolutePath());
        System.out.println("End UTC: " + Instant.now());
        System.out.println("Run ID: " + runId);
        System.out.println(
                "===== RAP EMBEDDING END status=VERIFIED_SUCCESS "
                        + "selectedQF=" + selectedQf + " =====");
        return selectedQf;
    }

    private static AdaptiveQfSearch.Trial<EmbeddingTrialResult>
    attemptEmbeddingAtQf(
            String runId,
            Path png,
            Path requestedOutput,
            String encodedKeys,
            byte[] message,
            PayloadPackage.PreparedMessage preparedMessage,
            SourceImageContext sourceContext,
            CandidateGenerator.Context candidateContext,
            int requiredBits,
            int qf,
            int maximumQf,
            ThreeStateEmbeddingEngine.Profile profile,
            StcPipelineIntegration stc,
            SamplingGeometryMode geometryMode,
            SamplingGeometry geometry,
            List<QfBandCoefficientReport.Row> qfRows,
            List<ThreeStateEmbeddingEngine.Attempt> allBalanceAttempts,
            boolean qfForensicAudit,
            FinalQfAuditState finalQfAudit
    ) throws Exception {
        if (finalQfAudit != null) {
            finalQfAudit.begin(qf, requiredBits, profile);
        }
        if (qfForensicAudit) {
            System.out.println(
                    "===== QF TRIAL START geometry="
                            + geometry.label() + " q=" + qf + " =====");
        }
        if (TrueParallelStateCache.enabled()) {
            TrueParallelStateCache.prefetchAhead(
                    png, sourceContext, candidateContext,
                    qf + 1, maximumQf, geometry);
        }
        Path reference = Files.createTempFile(
                "rap-ref-" + runId + "-" + geometry.outputTag()
                        + "-qf" + qf + "-", ".jpg");
        Path staged = null;
        QfBandCoefficientReport.Row row = null;
        boolean endPrinted = false;
        try {
            final JpegImage referenceImage;
            final CandidateSet candidateSet;
            if (TrueParallelStateCache.enabled()) {
                TrueParallelStateCache.State preparedState =
                        TrueParallelStateCache.prepare(
                                png, sourceContext, candidateContext,
                                qf, geometry);
                preparedState.materializeReference(reference);
                referenceImage = preparedState.freshReferenceImage();
                candidateSet = preparedState.candidateSet();
            } else {
                referenceImage = CanonicalReferenceJpeg.build(
                        sourceContext, reference, qf, geometry);
                candidateSet = CandidateGenerator.generateSet(
                        candidateContext, referenceImage);
            }
            if (finalQfAudit != null) {
                finalQfAudit.markReferenceBuilt();
            }
            row = QfBandCoefficientReport.analyze(
                    referenceImage, requiredBits, qf,
                    geometryMode, geometry);
            QfBandCoefficientReport.applyCandidateSet(
                    row, candidateSet, requiredBits);
            if (finalQfAudit != null) {
                finalQfAudit.captureCandidateRow(row);
            }
            if (qfForensicAudit) qfRows.add(row);
            if (!row.capacityPass) {
                row.terminalStatus = "CAPACITY_REJECTED";
                row.reason = "dry_count_minus_32_below_required_bits";
                if (finalQfAudit != null) {
                    finalQfAudit.captureCandidateRow(row);
                    finalQfAudit.finish(
                            row.terminalStatus, row.reason, false);
                }
                printQfTrialEnd(
                        qfForensicAudit, geometry, qf,
                        "CAPACITY_REJECTED");
                endPrinted = true;
                Files.deleteIfExists(reference);
                return AdaptiveQfSearch.Trial.capacityRejected(
                        qf, row.reason);
            }

            row.kernelRan = true;
            PayloadPackage.PackResult packed =
                    PayloadPackage.packDetailed(preparedMessage, encodedKeys, qf, geometry);
            if (packed.layout().totalPayloadBits() != requiredBits) {
                throw new IllegalStateException(
                        "Deterministic payload-size mismatch at QF " + qf);
            }

            FixedQfEmbeddingKernel.Result kernel;
            try {
                kernel = FixedQfEmbeddingKernel.embed(
                        referenceImage, candidateSet, packed.packed(),
                        encodedKeys, qf, geometry, profile, stc);
            } catch (ThreeStateEmbeddingEngine.NoFeasibleBalanceException e) {
                if (qfForensicAudit) {
                    allBalanceAttempts.addAll(e.attempts());
                }
                row.balanceAttemptCount = e.attempts().size();
                row.terminalStatus = e.reasonCode();
                row.reason = safeMessage(e);
                if (finalQfAudit != null) {
                    finalQfAudit.captureCandidateRow(row);
                    finalQfAudit.captureBalanceAttempts(e.attempts());
                    finalQfAudit.finish(
                            row.terminalStatus, row.reason, false);
                }
                printQfTrialEnd(
                        qfForensicAudit, geometry, qf,
                        e.reasonCode());
                endPrinted = true;
                Files.deleteIfExists(reference);
                return AdaptiveQfSearch.Trial.trialRejected(
                        qf, row.reason);
            }

            if (qfForensicAudit) {
                allBalanceAttempts.addAll(kernel.balance().attempts());
            }
            QfBandCoefficientReport.applyKernel(row, kernel);
            row.balanceAttemptCount = kernel.balance().attempts().size();
            if (finalQfAudit != null) {
                finalQfAudit.captureCandidateRow(row);
                finalQfAudit.captureKernel(kernel);
            }
            staged = createStagedOutput(requestedOutput, runId, qf);
            JpegImage writtenStego;
            try {
                JpegCore.writeCoefficients(
                        reference.toString(), staged.toString(),
                        kernel.intendedStego());
                writtenStego = validateWrittenCoefficientState(
                        staged, candidateSet.universe(),
                        kernel.intendedStego());
                if (finalQfAudit != null) {
                    finalQfAudit.markCoefficientWritebackVerified();
                }
            } catch (RuntimeException writebackFailure) {
                row.terminalStatus = "WRITEBACK_ERROR";
                row.reason = writebackFailure.getClass().getSimpleName()
                        + ":" + safeMessage(writebackFailure);
                throw writebackFailure;
            }
            VerificationResult verification = verifyEmbeddedPayload(
                    sourceContext, candidateContext, writtenStego,
                    encodedKeys, qf, message,
                    qfForensicAudit, stc.constraintHeight(), geometry);
            if (finalQfAudit != null) {
                finalQfAudit.captureVerification(
                        verification.extractionSuccess(),
                        verification.aesGcmVerified(),
                        verification.messageMatches(),
                        verification.detail());
            }
            if (!verification.extractionSuccess()
                    || !verification.aesGcmVerified()
                    || !verification.messageMatches()) {
                row.terminalStatus = "AUTHENTICATION_REJECTED";
                row.reason = "authenticated_round_trip:"
                        + verification.detail();
                if (finalQfAudit != null) {
                    finalQfAudit.captureCandidateRow(row);
                    finalQfAudit.finish(
                            row.terminalStatus, row.reason, false);
                }
                deleteOptionalTemporary(staged);
                Files.deleteIfExists(reference);
                printQfTrialEnd(
                        qfForensicAudit, geometry, qf,
                        "AUTHENTICATION_REJECTED");
                endPrinted = true;
                return AdaptiveQfSearch.Trial.trialRejected(
                        qf, row.reason);
            }

            row.selectedQfCandidate = true;
            row.terminalStatus = "VERIFIED";
            row.reason = "";
            if (finalQfAudit != null) {
                finalQfAudit.captureCandidateRow(row);
                finalQfAudit.finish("VERIFIED", "", true);
            }
            EmbeddingTrialResult value = new EmbeddingTrialResult(
                    qf, reference, staged, packed, kernel, verification);
            printQfTrialEnd(
                    qfForensicAudit, geometry, qf, "VERIFIED");
            endPrinted = true;
            return AdaptiveQfSearch.Trial.verified(qf, value);
        } catch (UnencodableReferenceException unencodable) {
            String reason = "unencodable_canonical_reference:"
                    + safeMessage(unencodable);
            if (row == null) {
                row = new QfBandCoefficientReport.Row();
                row.qf = qf;
                row.requiredPayloadBits = requiredBits;
                row.geometryMode = geometryMode.token();
                row.samplingGeometry = geometry.label();
                row.kernelRan = false;
                if (qfForensicAudit) qfRows.add(row);
            }
            row.terminalStatus = "UNENCODABLE_REFERENCE_REJECTED";
            row.reason = reason;
            if (finalQfAudit != null) {
                finalQfAudit.captureCandidateRow(row);
                finalQfAudit.finish(
                        row.terminalStatus, row.reason, false);
            }
            deleteOptionalTemporary(staged);
            Files.deleteIfExists(reference);
            printQfTrialEnd(
                    qfForensicAudit, geometry, qf,
                    row.terminalStatus);
            endPrinted = true;
            return AdaptiveQfSearch.Trial.trialRejected(qf, reason);
        } catch (RuntimeException | java.io.IOException failure) {
            if (row == null) {
                if (finalQfAudit != null) {
                    finalQfAudit.markRunFailure(
                            failure instanceof java.io.IOException
                                    ? "IO_ERROR" : "INTERNAL_ERROR",
                            failure.getClass().getSimpleName() + ":"
                                    + safeMessage(failure));
                }
                throw failure;
            }
            row.kernelRan = true;
            row.terminalStatus = failure instanceof java.io.IOException
                    ? "IO_ERROR" : "INTERNAL_ERROR";
            row.reason = failure.getClass().getSimpleName()
                    + ":" + safeMessage(failure);
            if (finalQfAudit != null) {
                finalQfAudit.captureCandidateRow(row);
                finalQfAudit.finish(
                        row.terminalStatus, row.reason, false);
            }
            try {
                deleteOptionalTemporary(staged);
                Files.deleteIfExists(reference);
            } catch (java.io.IOException cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            printQfTrialEnd(qfForensicAudit, geometry, qf, row.terminalStatus);
            endPrinted = true;
            throw failure;
        } finally {
            if (!endPrinted) {
                try {
                    deleteOptionalTemporary(staged);
                    Files.deleteIfExists(reference);
                } catch (java.io.IOException cleanupFailure) {
                    if (qfForensicAudit) {
                        System.err.println(
                                "QF cleanup warning q=" + qf + ": "
                                        + safeMessage(cleanupFailure));
                    }
                }
                printQfTrialEnd(
                        qfForensicAudit, geometry, qf, "FAILED");
            }
        }
    }

    static void deleteOptionalTemporary(Path path) throws java.io.IOException {
        if (path != null) Files.deleteIfExists(path);
    }

    private static void printQfTrialEnd(
            boolean enabled,
            SamplingGeometry geometry,
            int qf,
            String status
    ) {
        if (!enabled) return;
        System.out.println(
                "===== QF TRIAL END geometry=" + geometry.label()
                        + " q=" + qf
                        + " status=" + status + " =====");
    }

    private static Path createStagedOutput(
            Path requestedOutput, String runId, int qf
    ) throws Exception {
        Path absolute = requestedOutput.toAbsolutePath();
        Path parent = absolute.getParent();
        if (parent == null) parent = Path.of(".").toAbsolutePath();
        Files.createDirectories(parent);
        Path staged = Files.createTempFile(
                parent,
                "." + fileStem(absolute.getFileName().toString())
                        + "." + runId + ".qf" + qf + ".",
                ".tmp.jpg");
        deleteOptionalTemporary(staged);
        return staged;
    }

    private static void atomicReplace(Path staged, Path target)
            throws Exception {
        try {
            Files.move(
                    staged, target,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(staged, target);
        }
    }

    private static void cleanupSelectedReference(
            Path referenceJpeg, EmbedRunState runState
    ) {
        try {
            Files.deleteIfExists(referenceJpeg);
        } catch (java.io.IOException cleanupFailure) {
            referenceJpeg.toFile().deleteOnExit();
            System.err.println(
                    "Reference cleanup warning: "
                            + safeMessage(cleanupFailure));
        } finally {
            runState.selectedReference = null;
        }
    }

    private record EmbeddingTrialResult(
            int qf,
            Path referenceJpeg,
            Path stagedStego,
            PayloadPackage.PackResult packedPayload,
            FixedQfEmbeddingKernel.Result kernel,
            VerificationResult verification
    ) {}

    private static final class EmbedRunState {
        private Throwable failure;
        private String operation = "EMBED";
        private final String startedUtc = Instant.now().toString();
        private final FinalQfAuditState finalQf =
                new FinalQfAuditState();
        private final List<QfBandCoefficientReport.Row> qfRows =
                new ArrayList<>();
        private final List<ThreeStateEmbeddingEngine.Attempt>
                balanceAttempts = new ArrayList<>();
        private ThreeStateEmbeddingEngine.Profile profile;
        private PayloadPackage.PayloadLayout payloadLayout;
        private ProductionInputEvidence inputEvidence;
        private AuditMode auditMode = AuditMode.RESULT_ONLY;
        private AuditProfile auditProfile = AuditProfile.THESIS;
        private boolean auditEnabled = true;
        private int stcHeight = StcCarrierPlan.DEFAULT_STC_HEIGHT;
        private SamplingGeometryMode geometryMode =
                SamplingGeometryMode.GEOMETRY_420;
        private SamplingGeometry geometry = SamplingGeometry.JPEG_420;
        private Path selectedReference;
        private boolean auditPublished;
        private boolean resultAuditPublished;
        private boolean outputCommitted;
        private boolean embeddingSucceeded;
        private long embeddingStartNanos;
        private Long embeddingElapsedMillis;
        private Long evaluationElapsedMillis;
        private ProductionOutputEvidence outputEvidence;
    }

    private static void extract(
            Path png,
            Path stego,
            String encodedKeys,
            QfRange qfRange,
            int[] stcHeights,
            GeometrySelection geometrySelection,
            Path outFile
    ) throws Exception {
        if (Files.exists(outFile)) {
            throw new IllegalArgumentException(
                    "Refusing to overwrite an existing recovered-message file: "
                            + outFile.toAbsolutePath());
        }
        validateReadableFile(stego, "Stego JPEG");
        JpegImage stegoImage =
                JpegCore.readCoefficients(stego.toString());
        SamplingGeometry geometry = geometryForStego(
                stegoImage, geometrySelection);
        SourceImageContext sourceContext = TrueParallelStateCache.sharedSourceContext(
                png, SourceImageContext.load(png));
        CandidateGenerator.Context candidateContext =
                new CandidateGenerator.Context(sourceContext);
        RecoveredPayload recovered = recoverPayload(
                sourceContext, candidateContext, stegoImage, encodedKeys,
                qfRange, stcHeights, geometry);
        System.out.println("Recovered sampling geometry: "
                + recovered.geometry().label());
        System.out.println("Recovered QF: " + recovered.qf());
        System.out.println("Recovered STC height: "
                + recovered.stcHeight());
        System.out.println("Carrier layout: reserved header plus capped body active set");
        System.out.println("Payload bytes recovered: " + recovered.packedBytes());
        System.out.println("AES-GCM and authenticated header verified: true");
        AtomicFiles.writeBytes(outFile, recovered.plaintext());
        System.out.println("Exact recovered message written to: "
                + outFile.toAbsolutePath());
    }

    /** Algorithm 5: try every public QF and accept only a fully authenticated result. */
    private static RecoveredPayload recoverPayload(
            Path png, Path stego, String encodedKeys, QfRange qfRange) throws Exception {
        validateReadableFile(stego, "Stego JPEG");
        SamplingGeometry detected = SamplingGeometry.detect(
                JpegCore.readCoefficients(stego.toString()));
        return recoverPayload(
                png, stego, encodedKeys, qfRange,
                AUTO_EXTRACTION_HEIGHTS, detected);
    }

    private static RecoveredPayload recoverPayload(
            Path png,
            Path stego,
            String encodedKeys,
            QfRange qfRange,
            int[] requestedHeights,
            SamplingGeometry geometry
    ) throws Exception {
        validateReadableFile(png, "Shared PNG");
        validateReadableFile(stego, "Stego JPEG");
        JpegImage stegoImg = JpegCore.readCoefficients(stego.toString());
        geometry.require(stegoImg);
        SourceImageContext sourceContext = TrueParallelStateCache.sharedSourceContext(
                png, SourceImageContext.load(png));
        CandidateGenerator.Context candidateContext =
                new CandidateGenerator.Context(sourceContext);
        return recoverPayload(
                sourceContext, candidateContext, stegoImg, encodedKeys,
                qfRange, requestedHeights, geometry);
    }

    /**
     * Algorithm 5 with run-scoped preprocessing.  Fixed-QF reference/candidate
     * preparation is performed once per QF and reused across all requested STC
     * heights; STC height changes only coding/extraction, not the cost surface or
     * keyed carrier order.
     */
    private static RecoveredPayload recoverPayload(
            SourceImageContext sourceContext,
            CandidateGenerator.Context candidateContext,
            JpegImage stegoImg,
            String encodedKeys,
            QfRange qfRange,
            int[] requestedHeights,
            SamplingGeometry geometry
    ) throws Exception {
        geometry.require(stegoImg);
        int[] heights = normalizedStcHeights(requestedHeights);

        for (int qf = qfRange.minQf(); qf <= qfRange.maxQf(); qf++) {
            RecoveryQfContext prepared = prepareRecoveryAtQf(
                    sourceContext, candidateContext, stegoImg,
                    encodedKeys, qf, geometry);
            if (prepared == null) continue;
            for (int stcHeight : heights) {
                StcPipelineIntegration stc =
                        new StcPipelineIntegration(stcHeight);
                RecoveredPayload recovered = tryRecoverPayloadAtPreparedQf(
                        stegoImg, encodedKeys, stc, stcHeight, qf,
                        geometry, prepared);
                if (recovered != null) return recovered;
            }
        }

        throw new PayloadRecoveryRejectedException("Could not recover and authenticate a payload for QF range "
                + qfRange.minQf() + "-" + qfRange.maxQf()
                + " and STC heights " + Arrays.toString(heights));
    }

    private static int[] normalizedStcHeights(int[] requested) {
        int[] source = requested == null || requested.length == 0
                ? AUTO_EXTRACTION_HEIGHTS : requested;
        int[] temporary = new int[source.length];
        int count = 0;
        for (int height : source) {
            StcCarrierPlan.validateSupportedHeight(height);
            boolean duplicate = false;
            for (int i = 0; i < count; i++) {
                if (temporary[i] == height) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) temporary[count++] = height;
        }
        return Arrays.copyOf(temporary, count);
    }

    private record RecoveryQfContext(
            List<Candidate> keyedCandidates,
            byte[] headerCarrierBits
    ) {
        RecoveryQfContext {
            keyedCandidates = List.copyOf(keyedCandidates);
            headerCarrierBits = Arrays.copyOf(
                    headerCarrierBits, headerCarrierBits.length);
        }

        @Override public byte[] headerCarrierBits() {
            return Arrays.copyOf(headerCarrierBits, headerCarrierBits.length);
        }
    }

    /** QF-dependent preparation shared by every STC-height trial. */
    private static RecoveryQfContext prepareRecoveryAtQf(
            SourceImageContext sourceContext,
            CandidateGenerator.Context candidateContext,
            JpegImage stegoImg,
            String encodedKeys,
            int qf,
            SamplingGeometry geometry
    ) throws Exception {
        // Every canonical reference carries exactly the declared tables for
        // its QF, and carrierLayoutsMatch() below rejects any other QF -- but
        // only after the whole reference and SI-UNIWARD candidate set have
        // been built.  Comparing the stego tables first returns the identical
        // null for a wrong QF without that work.
        if (!quantTablesMatchCanonical(stegoImg, qf)) return null;
        int headerCarrierCount =
                HEADER_BITS + StcCarrierPlan.DEFAULT_HEADER_RESERVE_BITS;
        Path ref = Files.createTempFile(
                "rap-extract-" + geometry.outputTag()
                        + "-qf" + qf + "-",
                ".jpg");
        try {
            final JpegImage refImg;
            final CandidateSet candidateSet;
            try {
                if (TrueParallelStateCache.enabled()) {
                    TrueParallelStateCache.State preparedState =
                            TrueParallelStateCache.prepare(
                                    sourceContext.sourcePath(), sourceContext,
                                    candidateContext, qf, geometry);
                    preparedState.materializeReference(ref);
                    refImg = preparedState.freshReferenceImage();
                    candidateSet = preparedState.candidateSet();
                } else {
                    refImg = CanonicalReferenceJpeg.build(
                            sourceContext, ref, qf, geometry);
                    candidateSet = CandidateGenerator.generateSet(
                            candidateContext, refImg);
                }
            } catch (UnencodableReferenceException rejected) {
                return null;
            }
            if (!carrierLayoutsMatch(refImg, stegoImg)) return null;
            List<Candidate> candidates =
                    new ArrayList<>(candidateSet.dryBaseCostSorted());
            StcKeyedOrder.apply(candidates, encodedKeys, qf, geometry);
            if (candidates.size() < headerCarrierCount) return null;
            byte[] headerCarrierBits = coverBits(
                    stegoImg, candidates.subList(0, headerCarrierCount));
            return new RecoveryQfContext(candidates, headerCarrierBits);
        } finally {
            Files.deleteIfExists(ref);
        }
    }

    /**
     * Algorithm 6: one fixed-QF/fixed-height extraction trial using a QF context
     * already proven to match the stego layout. Public header fields only bound
     * the trial; acceptance still requires AES-GCM/AAD authentication and full
     * payload-format validation.
     */
    private static RecoveredPayload tryRecoverPayloadAtPreparedQf(
            JpegImage stegoImg,
            String encodedKeys,
            StcPipelineIntegration stc,
            int stcHeight,
            int qf,
            SamplingGeometry geometry,
            RecoveryQfContext prepared
    ) {
        List<Candidate> candidates = prepared.keyedCandidates();
        try {
            byte[] header = stc.extractPayload(
                    prepared.headerCarrierBits(), PayloadPackage.headerBytes());
            int cipherLen = PayloadPackage.ciphertextLengthFromHeader(header);
            if (cipherLen < PayloadPackage.PREPROCESSING_METADATA_LEN) return null;
            int bodyBits = Math.multiplyExact(cipherLen, 8);
            StcCarrierPlan carrierPlan =
                    StcCarrierPlan.createForHeight(
                            candidates.size(), HEADER_BITS,
                            bodyBits, stcHeight);
            int bodyEnd = Math.addExact(
                    carrierPlan.headerCarrierCount(),
                    carrierPlan.bodyCarrierCount());
            byte[] bodyCarrierBits = coverBits(
                    stegoImg,
                    candidates.subList(
                            carrierPlan.headerCarrierCount(), bodyEnd));
            byte[] body = stc.extractPayload(bodyCarrierBits, cipherLen);
            byte[] packed = new byte[Math.addExact(
                    PayloadPackage.headerBytes(), cipherLen)];
            System.arraycopy(
                    header, 0, packed, 0, PayloadPackage.headerBytes());
            System.arraycopy(
                    body, 0, packed, PayloadPackage.headerBytes(), cipherLen);
            byte[] plain = PayloadPackage.unpack(packed, encodedKeys, qf, geometry);
            return new RecoveredPayload(
                    qf, stcHeight, geometry, packed.length, plain);
        } catch (java.security.GeneralSecurityException | IllegalArgumentException
                | ArithmeticException rejectedTrial) {
            return null;
        }
    }

    /**
     * True iff the stego component tables equal the canonical tables that
     * CanonicalReferenceJpeg installs at {@code qf}: luminance for component
     * 0, chrominance for Cb and Cr.  A necessary condition for
     * carrierLayoutsMatch(), so it can only skip QFs that would be rejected.
     */
    static boolean quantTablesMatchCanonical(JpegImage stego, int qf) {
        int[] luminance = StandardJpegQuantization.luminance(qf);
        int[] chrominance = StandardJpegQuantization.chrominance(qf);
        for (int i = 0; i < stego.components.length; i++) {
            int[] expected = i == 0 ? luminance : chrominance;
            if (!Arrays.equals(stego.component(i).quantTable, expected)) {
                return false;
            }
        }
        return true;
    }

    private static boolean carrierLayoutsMatch(
            JpegImage reference, JpegImage stego
    ) {
        if (reference.width != stego.width
                || reference.height != stego.height
                || reference.components.length != stego.components.length) {
            return false;
        }
        for (int i = 0; i < reference.components.length; i++) {
            JpegComponent a = reference.component(i);
            JpegComponent b = stego.component(i);
            if (a.componentIndex != b.componentIndex
                    || a.widthBlocks != b.widthBlocks
                    || a.heightBlocks != b.heightBlocks
                    || a.hSamplingFactor != b.hSamplingFactor
                    || a.vSamplingFactor != b.vSamplingFactor
                    || a.quantTableIndex != b.quantTableIndex
                    || !Arrays.equals(a.quantTable, b.quantTable)
                    || a.coefficients.length != b.coefficients.length) {
                return false;
            }
        }
        return true;
    }

    private static SamplingGeometry geometryForStego(
            JpegImage stego,
            GeometrySelection selection
    ) {
        SamplingGeometry detected = SamplingGeometry.detect(stego);
        if (!selection.contains(detected)) {
            throw new IllegalArgumentException(
                    "Stego sampling geometry " + detected.label()
                            + " is not permitted by --geometryMode "
                            + selection.token());
        }
        return detected;
    }

    private static void batchEmbed(Map<String, String> options) throws Exception {
        final long batchStartedNanos = System.nanoTime();
        thesis.rap.alg.SiUniwardCostEngine.resetBackendPerformance();
        Path pngDir = Path.of(req(options, "--pngDir"));
        Path msgDir = Path.of(req(options, "--msgDir"));
        Path outDir = Path.of(req(options, "--outDir"));
        if (!Files.isDirectory(pngDir)) {
            throw new IllegalArgumentException(
                    "PNG directory not found: " + pngDir.toAbsolutePath());
        }
        if (!Files.isDirectory(msgDir)) {
            throw new IllegalArgumentException(
                    "Message directory not found: " + msgDir.toAbsolutePath());
        }
        requireDisjointBatchRoots(pngDir, msgDir, outDir);
        Files.createDirectories(outDir);

        boolean recursive = strictBoolean(options, "--recursive", true);
        boolean resume = strictBoolean(options, "--resume", false);
        boolean continueOnError =
                strictBoolean(options, "--continueOnError", true);
        boolean deleteJpegsAfterEvaluation =
                strictBoolean(options, "--deleteJpegsAfterEvaluation", false);
        if (deleteJpegsAfterEvaluation && resume) {
            throw new IllegalArgumentException(
                    "--resume=true is incompatible with "
                            + "--deleteJpegsAfterEvaluation=true because "
                            + "verified JPEG outputs are intentionally absent.");
        }
        long startIndex = Long.parseLong(
                options.getOrDefault("--startIndex", "1"));
        if (startIndex < 1) {
            throw new IllegalArgumentException("--startIndex must be >= 1");
        }

        List<Path> images = discoverFiles(pngDir, recursive, ".png");
        String messageExtension =
                options.getOrDefault("--messageExtension", ".msg");
        if (messageExtension.isBlank()
                || !messageExtension.startsWith(".")) {
            throw new IllegalArgumentException(
                    "--messageExtension must be a nonblank extension beginning with '.'");
        }
        List<Path> messages =
                discoverFiles(msgDir, recursive, messageExtension);
        String onlyMessageName =
                options.getOrDefault("--onlyMessageName", "");
        if (!onlyMessageName.isBlank()) {
            messages = messages.stream()
                    .filter(path -> path.getFileName().toString()
                            .equals(onlyMessageName))
                    .toList();
        }
        if (images.isEmpty()) {
            throw new IllegalArgumentException(
                    "No PNG files found under " + pngDir.toAbsolutePath());
        }
        if (messages.isEmpty()) {
            throw new IllegalArgumentException(
                    "No message files found under " + msgDir.toAbsolutePath());
        }
        if (!onlyMessageName.isBlank() && messages.size() != 1) {
            throw new IllegalArgumentException(
                    "--onlyMessageName must select exactly one discovered message: "
                            + onlyMessageName);
        }
        requireMessagesWithinProtocolLimit(messages);
        GeometrySelection geometrySelection = geometrySelection(options);
        requireGroupAlignedStartIndex(geometrySelection, startIndex);
        validateDistinctBatchOutputs(
                pngDir, images, msgDir, messages,
                geometrySelection.geometries());

        long totalPairs = Math.multiplyExact(
                Math.multiplyExact(
                        (long) images.size(), (long) messages.size()),
                geometrySelection.geometries().size());
        String batchId = UUID.randomUUID().toString();
        Path manifest = Path.of(options.getOrDefault(
                "--manifest",
                thesis.rap.runtime.RuntimePaths.results("batch_manifests",
                        "batch_" + batchId + ".csv").toString()));
        String encodedKeys = keyMaterial(options);
        QfRange range = qfRange(options);
        AdaptiveQFSelector.SelectionConstraints constraints =
                selectionConstraints(options);
        AuditMode auditMode = auditMode(options);
        AuditProfile auditProfile = auditProfile(options);
        boolean auditEnabled = auditEnabled(options);
        int stcHeight = embeddingStcHeight(options);
        BatchExecutorSupport.Config threading = batchThreading(options);
        String declaredCpuAffinity = options.getOrDefault(
                "--cpuAffinity", "UNDECLARED");
        BatchExecutorSupport.CpuSlotGate cpuSlotGate =
                new BatchExecutorSupport.CpuSlotGate(
                        threading.cpuSlots());
        System.setProperty(
                ThreeStateEmbeddingEngine.VERBOSE_ATTEMPTS_PROPERTY,
                Boolean.toString(auditEnabled
                        && auditMode.writesQfForensics()
                        && auditProfile == AuditProfile.FULL_AUDIT));

        int pairModulo = Integer.parseInt(options.getOrDefault("--pairModulo", "1"));
        int pairRemainder = Integer.parseInt(options.getOrDefault("--pairRemainder", "0"));
        if (pairModulo < 1) {
            throw new IllegalArgumentException("--pairModulo must be >= 1");
        }
        if (pairRemainder < 0 || pairRemainder >= pairModulo) {
            throw new IllegalArgumentException(
                    "--pairRemainder must be in 0..pairModulo-1");
        }
        String shardUnit = options.getOrDefault("--shardUnit", "PAIR")
                .trim().toUpperCase(java.util.Locale.ROOT);
        if (!shardUnit.equals("PAIR") && !shardUnit.equals("IMAGE")) {
            throw new IllegalArgumentException(
                    "--shardUnit must be PAIR or IMAGE: " + shardUnit);
        }
        long pairsPerImage = Math.multiplyExact(
                (long) messages.size(),
                geometrySelection.geometries().size());

        List<String> header = List.of(
                "batch_id", "message_group", "message_number", "message_name",
                "pair_index", "total_pairs", "pair_modulo", "pair_remainder",
                "execution_backend", "gpu_physical_id", "gpu_logical_device",
                "geometry_mode", "sampling_geometry", "selected_qf", "image", "message", "output",
                "status", "detail");
        List<BatchPair> pairs = new ArrayList<>();
        long pairIndex = 0;
        for (Path image : images) {
            for (Path message : messages) {
                for (SamplingGeometry geometry
                        : geometrySelection.geometries()) {
                    pairIndex++;
                    if (pairIndex < startIndex) continue;
                    boolean owned = shardUnit.equals("IMAGE")
                            ? imageBelongsToShard(pairIndex, pairsPerImage,
                                    pairModulo, pairRemainder)
                            : pairBelongsToShard(pairIndex, pairModulo, pairRemainder);
                    if (!owned) continue;
                    Path output = outDir.resolve(batchOutputName(
                            pngDir, image, msgDir, message,
                            geometry));
                    pairs.add(new BatchPair(
                            pairIndex, image, message, output,
                            geometry));
                }
            }
        }
        boolean concurrentExecution = !pairs.isEmpty()
                && continueOnError
                && threading.usesMultipleTasks(pairs.size());
        int initialExecutorSubmissions = concurrentExecution
                ? threading.submissionLimit(pairs.size()) : 0;

        System.out.println("===========================================================");
        System.out.println(" RAP-UNIWARD THREE-STATE BATCH START");
        System.out.println("===========================================================");
        System.out.printf("Batch ID: %s%n", batchId);
        System.out.println("Start UTC: " + Instant.now());
        System.out.printf("Execution backend: %s | physical_gpu=%s | logical_cuda_device=%s%n",
                ExecutionBackend.description(),
                System.getenv().getOrDefault("RAP_PHYSICAL_GPU_ID", "CPU_OR_UNDECLARED"),
                System.getenv().getOrDefault("RAP_CUDA_DEVICE", "0"));
        System.out.printf(
                "Discovered %d PNG images, %d message files, and %d "
                        + "selected geometries = %d deterministic "
                        + "image-message-geometry trials.%n",
                images.size(), messages.size(),
                geometrySelection.geometries().size(), totalPairs);
        System.out.printf(
                "Geometry mode: %s | profiles: %s%n",
                geometrySelection.token(),
                geometrySelection.geometries().stream()
                        .map(g -> g.label() + "[0x" + g.signatureHex() + "]")
                        .toList());
        if (messages.size() == 1) {
            var group = thesis.rap.analysis.MessageGroup.from(
                    messages.get(0).getFileName().toString());
            System.out.printf(
                    "Message group: %s | message_number=%d | message_name=%s%n",
                    group.label(), group.number(), group.name());
        }
        System.out.printf(
                "Start index: %d | shard=%d/%d unit=%s | selected_pairs=%d | resume=%s | recursive=%s%n",
                startIndex, pairRemainder, pairModulo, shardUnit,
                pairs.size(), resume, recursive);
        System.out.printf(
                "Thread mode: %s | pair task policy=%s | "
                        + "discovered executable pairs=%d | "
                        + "initial executor submissions=%d | "
                        + "carrier parallelism=%d | CPU-slot permits=%d%n",
                threading.mode().name(), threading.taskPolicy(),
                pairs.size(), initialExecutorSubmissions,
                threading.carrierParallelism(), threading.cpuSlots());
        System.out.printf(
                "CPU affinity: requested=%s | observed=%s%n",
                declaredCpuAffinity,
                BatchExecutorSupport.observedCpuAffinity());
        if (concurrentExecution
                && threading.allDiscoveredPairTasks()) {
            System.out.printf(
                    "Concurrency topology: all %d discovered pair tasks "
                            + "submitted once -> fair Semaphore(%d) -> at most "
                            + "%d admitted complete-pair kernels.%n",
                    pairs.size(), threading.cpuSlots(),
                    threading.cpuSlots());
        } else if (concurrentExecution
                && threading.boundedVirtualTasks()) {
            System.out.printf(
                    "Concurrency topology: virtual in-flight window=%d -> "
                            + "fair Semaphore(%d) -> at most %d admitted "
                            + "complete-pair kernels.%n",
                    initialExecutorSubmissions, threading.cpuSlots(),
                    threading.cpuSlots());
        } else if (concurrentExecution) {
            System.out.printf(
                    "Concurrency topology: platform window=%d -> fair "
                            + "Semaphore(%d) -> at most %d admitted "
                            + "complete-pair kernels.%n",
                    initialExecutorSubmissions, threading.cpuSlots(),
                    threading.cpuSlots());
        } else {
            System.out.println(
                    "Concurrency topology: sequential execution; executor "
                            + "submissions=0.");
        }
        System.out.printf(
                "Audit enabled: %s | mode: %s | detail scope: %s | "
                        + "operational storage gate: none%n",
                auditEnabled,
                auditEnabled ? auditMode.name() : "DISABLED",
                !auditEnabled ? "NOT_APPLICABLE"
                        : auditMode == AuditMode.RESULT_ONLY
                        ? "FINAL_QF_RUN_AND_CB_CR_CHANNEL_AGGREGATES"
                        : "ALL_QFS_" + auditProfile.name());
        System.out.println(
                "Delete reference/stego JPEGs after committed processing: "
                        + deleteJpegsAfterEvaluation);
        System.out.printf(
                "STC height: %d | body width cap: %d | profile: %s%n",
                stcHeight, StcCarrierPlan.bodyWidthCap(stcHeight),
                stcHeight == StcCarrierPlan.DEFAULT_STC_HEIGHT
                        ? "BASELINE" : "RESEARCH");
        System.out.println("Batch manifest: " + manifest.toAbsolutePath());
        System.out.println(
                "Parallelism boundary: independent image-message-geometry "
                        + "populations may execute concurrently; CUDA "
                        + "stages may also overlap through device stream pools. "
                        + "Each population retains the ordered QF 75-100 search "
                        + "and the same six-algorithm acceptance sequence.");
        if (!continueOnError
                && threading.usesMultipleTasks(pairs.size())) {
            System.out.println(
                    "Strict fail-fast requested; effective workers=1 so no later pair "
                            + "can commit after the first failed pair.");
        } else if (threading.usesMultipleTasks(pairs.size())) {
            System.out.println(
                    "Every Java console line is emitted atomically with its "
                            + "pair_id, image_id, and msg_id correlation prefix.");
        }

        BatchExecutionContext context = new BatchExecutionContext(
                outDir, encodedKeys, range, constraints, auditMode,
                auditProfile,
                auditEnabled, stcHeight, resume, continueOnError,
                deleteJpegsAfterEvaluation, manifest, header, batchId,
                totalPairs, pairModulo, pairRemainder, cpuSlotGate, geometrySelection);
        BatchCounters counters = new BatchCounters();
        if (concurrentExecution) {
            runConcurrentBatch(pairs, threading, context, counters);
        } else {
            runSequentialBatch(pairs, context, counters);
        }

        System.out.printf(
                "Batch invocation complete: handled=%d, completed=%d, "
                        + "skipped_existing=%d, no_output=%d, failed=%d%n",
                counters.handled, counters.completed,
                counters.skippedExisting, counters.noOutput, counters.failed);
        double batchSeconds = Math.max(1e-9,
                (System.nanoTime() - batchStartedNanos) / 1_000_000_000.0);
        var backendPerf = thesis.rap.alg.SiUniwardCostEngine.backendPerformance();
        System.out.printf(java.util.Locale.ROOT,
                "Measured batch throughput: %.3f terminal pairs/s over %.3f s.%n",
                counters.handled / batchSeconds, batchSeconds);
        System.out.printf(java.util.Locale.ROOT,
                "SI backend counters: CPU batches=%d candidates=%d ms=%.3f | "
                        + "HIP batches=%d candidates=%d ms=%.3f | "
                        + "CUDA batches=%d candidates=%d ms=%.3f | "
                        + "verify=%d fallbacks=%d%n",
                backendPerf.cpuBatches(), backendPerf.cpuCandidates(),
                backendPerf.cpuMilliseconds(),
                backendPerf.hipBatches(), backendPerf.hipCandidates(),
                backendPerf.hipMilliseconds(),
                backendPerf.cudaBatches(), backendPerf.cudaCandidates(),
                backendPerf.cudaMilliseconds(),
                backendPerf.verifyBatches(), backendPerf.autoFallbacks());
        // What actually executed, per stage; the manifest's execution_backend
        // column records only the configured policy.
        System.out.println("GPU_STAGE_EXECUTION " + GpuStageStats.summary()
                + " total_cpu_fallbacks=" + GpuStageStats.totalFallbacks());
        if (counters.rebuiltAfterQuarantine > 0) {
            // A resumed run found previously verified outputs that no longer
            // authenticate here.  After a GPU->CPU failover retry this is the
            // signature of a cross-backend parity break: the other shards'
            // GPU outputs may be unrecoverable on CPU as well.
            System.err.println("WARNING CROSS_BACKEND_OR_ARTIFACT_INTEGRITY: "
                    + counters.rebuiltAfterQuarantine
                    + " existing verified output(s) failed re-verification and were"
                    + " quarantined and re-embedded (status REBUILT_AFTER_QUARANTINE)."
                    + " If this run is a failover retry on a different backend,"
                    + " re-extract a sample of every other shard on CPU before"
                    + " trusting any GPU-produced output.");
        }
        if (auditEnabled && auditMode == AuditMode.RESULT_ONLY) {
            System.out.println(
                    "Authoritative final pair outcomes are in "
                            + ResultAuditWriter.DEFAULT_CSV
                            + "; each row contains the selected/last QF, "
                            + "terminal candidate/capacity/balance/STC state, "
                            + "and successful final-QF aggregate evaluation. "
                            + "No intermediate-QF history was written.");
        } else if (auditEnabled) {
            System.out.println(
                    "Authoritative per-run/skip/failure details remain in the "
                            + "RAP-UNIWARD evaluation CSVs, per-run JSON, audit bundles, "
                            + "and ordered batch manifest.");
        } else {
            System.out.println(
                    "Audit/evaluation artifacts were disabled; the ordered batch "
                            + "manifest remains the operational pair-status record.");
        }
        System.out.println("===========================================================");
        System.out.println(" RAP-UNIWARD THREE-STATE BATCH END");
        System.out.println("End UTC: " + Instant.now());
        System.out.printf(
                "Final status: handled=%d success=%d verified_existing=%d "
                        + "no_output=%d failed=%d%n",
                counters.handled, counters.completed,
                counters.skippedExisting, counters.noOutput, counters.failed);
        System.out.println("===========================================================");
        if (counters.failed > 0) {
            throw new IllegalStateException(
                    "Batch terminal result set contains unexpected failures="
                            + counters.failed + "; scientifically valid no_output="
                            + counters.noOutput + ".");
        }
    }

    
    static void requireMessagesWithinProtocolLimit(List<Path> messages)
            throws java.io.IOException {
        List<String> rejected = new ArrayList<>();
        for (Path message : messages) {
            byte[] bytes = Files.readAllBytes(message);
            try {
                PayloadPackage.prepareMessage(bytes);
            } catch (IllegalArgumentException | ArithmeticException tooLarge) {
                rejected.add(message.getFileName() + "=" + bytes.length + "B");
            }
        }
        if (!rejected.isEmpty()) {
            throw new IllegalArgumentException(
                    "RAP-UNIWARD compact protocol cannot carry " + rejected.size()
                            + " message file(s); no population was started. "
                            + "The prepared payload must be "
                            + PayloadPackage.PREPROCESSING_METADATA_LEN + ".."
                            + PayloadPackage.MAX_CIPHERTEXT_LEN
                            + " bytes (a raw message of at most "
                            + (PayloadPackage.MAX_CIPHERTEXT_LEN
                            - PayloadPackage.PREPROCESSING_METADATA_LEN)
                            + " bytes unless Huffman coding shrinks it): "
                            + rejected);
        }
    }

    private static void runSequentialBatch(
            List<BatchPair> pairs,
            BatchExecutionContext context,
            BatchCounters counters
    ) throws Exception {
        for (BatchPair pair : pairs) {
            BatchPairResult result = processBatchPair(pair, context);
            commitBatchResult(result, context, counters);
            if (result.failure() != null && !context.continueOnError()) {
                throw result.failure();
            }
        }
    }

    private static void runConcurrentBatch(
            List<BatchPair> pairs,
            BatchExecutorSupport.Config threading,
            BatchExecutionContext context,
            BatchCounters counters
    ) throws Exception {
        ExecutorService executor = BatchExecutorSupport.create(threading);
        CompletionService<BatchPairResult> completion =
                new ExecutorCompletionService<>(executor);
        java.util.Iterator<BatchPair> remaining = pairs.iterator();
        int inFlight = 0;
        int initial = threading.submissionLimit(pairs.size());
        TreeMap<Long, BatchPairResult> orderedBuffer = new TreeMap<>();
        java.util.ArrayDeque<Long> expectedManifestOrder = new java.util.ArrayDeque<>();
        for (BatchPair pair : pairs) expectedManifestOrder.addLast(pair.pairIndex());
        boolean completedNormally = false;
        Throwable primaryFailure = null;
        try {
            for (int i = 0; i < initial; i++) {
                BatchPair pair = remaining.next();
                completion.submit(() -> processBatchPair(pair, context));
                inFlight++;
            }
            while (inFlight > 0) {
                Future<BatchPairResult> future;
                try {
                    future = completion.take();
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(
                            "Interrupted while waiting for a batch pair",
                            interrupted);
                }
                inFlight--;
                BatchPairResult result = completedPairResult(future);
                BatchPairResult previous = orderedBuffer.put(
                        result.pair().pairIndex(), result);
                if (previous != null) {
                    throw new IllegalStateException(
                            "Duplicate completed pair index: "
                                    + result.pair().pairIndex());
                }

                while (!expectedManifestOrder.isEmpty()
                        && orderedBuffer.containsKey(expectedManifestOrder.peekFirst())) {
                    long expectedIndex = expectedManifestOrder.removeFirst();
                    BatchPairResult ordered = orderedBuffer.remove(expectedIndex);
                    commitBatchResult(ordered, context, counters);
                }

                if (result.failure() != null && !context.continueOnError()) {
                    throw result.failure();
                }

                if (remaining.hasNext()) {
                    BatchPair pair = remaining.next();
                    completion.submit(() -> processBatchPair(pair, context));
                    inFlight++;
                }
            }
            if (!orderedBuffer.isEmpty() || !expectedManifestOrder.isEmpty()) {
                throw new IllegalStateException(
                        "Completed-pair buffer does not match the deterministic shard order");
            }
            completedNormally = true;
        } catch (Exception | Error failure) {
            primaryFailure = failure;
            throw failure;
        } finally {
            BatchExecutorSupport.closePreservingFailure(
                    executor, completedNormally, primaryFailure);
        }
    }

    private static BatchPairResult completedPairResult(
            Future<BatchPairResult> future
    ) throws Exception {
        try {
            return future.get();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(
                    "Interrupted while collecting a batch pair", interrupted);
        } catch (ExecutionException execution) {
            Throwable cause = execution.getCause();
            if (cause instanceof Exception exception) throw exception;
            if (cause instanceof Error error) throw error;
            throw new IllegalStateException(
                    "Batch worker failed with a non-Exception throwable", cause);
        }
    }

    @SuppressWarnings("try")
    private static BatchPairResult processBatchPair(
            BatchPair pair,
            BatchExecutionContext context
    ) {
        Path image = pair.image();
        Path message = pair.message();
        Path output = pair.output();
        try (PairConsoleContext.Scope pairScope =
                     PairConsoleContext.batchPair(
                             pair.pairIndex(), context.totalPairs(),
                             image, message)) {
            String worker = BatchExecutorSupport.currentWorkerLabel();
            BatchExecutorSupport.CpuSlotGate gate =
                    context.cpuSlotGate();
            System.out.printf(
                    "PAIR_TASK_QUEUED pair_index=%d total_pairs=%d "
                            + "worker=%s geometry=%s image_file=%s "
                            + "msg_file=%s%n",
                    pair.pairIndex(), context.totalPairs(), worker,
                    pair.geometry().label(),
                    image.getFileName(), message.getFileName());
            System.out.printf(
                    "CPU_SLOT_WAIT limit=%d active=%d available=%d "
                            + "queued_waiters=%d worker=%s%n",
                    gate.limit(), gate.active(), gate.availablePermits(),
                    gate.queuedThreads(), worker);

            final BatchExecutorSupport.CpuPermit permit;
            try {
                permit = gate.acquire();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                System.err.printf(
                        "CPU_SLOT_INTERRUPTED limit=%d worker=%s%n",
                        gate.limit(), worker);
                return new BatchPairResult(
                        pair,
                        "FAILED_EXCEPTION",
                        "Interrupted while waiting for one of "
                                + gate.limit() + " CPU-slot permits.",
                        null,
                        worker,
                        interrupted);
            }

            try {
                System.out.printf(
                        "CPU_SLOT_ACQUIRED limit=%d active=%d available=%d "
                                + "worker=%s%n",
                        gate.limit(), gate.active(),
                        gate.availablePermits(), worker);
                BatchPairResult result = processBatchPairInsideCpuSlot(
                        pair, context, worker);
                System.out.printf(
                        "PAIR_TASK_COMPLETE pair_index=%d status=%s "
                                + "worker=%s%n",
                        pair.pairIndex(), result.status(), worker);
                return result;
            } finally {
                permit.close();
                System.out.printf(
                        "CPU_SLOT_RELEASED limit=%d active=%d available=%d "
                                + "peak_active=%d worker=%s%n",
                        gate.limit(), gate.active(),
                        gate.availablePermits(), gate.peakActive(), worker);
            }
        }
    }

    private static BatchPairResult processBatchPairInsideCpuSlot(
            BatchPair pair,
            BatchExecutionContext context,
            String worker
    ) {
        Path image = pair.image();
        Path message = pair.message();
        Path output = pair.output();
        System.out.printf(
                "PAIR_TASK_START pair_index=%d total_pairs=%d worker=%s "
                        + "backend=%s physical_gpu=%s logical_cuda_device=%s "
                        + "geometry=%s image_file=%s msg_file=%s%n",
                pair.pairIndex(), context.totalPairs(), worker,
                ExecutionBackend.mode().name(),
                System.getenv().getOrDefault("RAP_PHYSICAL_GPU_ID", "CPU_OR_UNDECLARED"),
                System.getenv().getOrDefault("RAP_CUDA_DEVICE", "0"),
                pair.geometry().label(),
                image.getFileName(), message.getFileName());
        String status = "FAILED_EXCEPTION";
        String detail = "";
        Integer selectedQf = null;
        Exception failure = null;
        try {
            ensureParentDirectory(output);
            if (context.resume() && outputArtifactsExist(output)) {
                Path sidecar = VerifiedOutputManifest.sidecarFor(output);
                boolean usableOutput = Files.isRegularFile(
                        output, LinkOption.NOFOLLOW_LINKS)
                        && Files.size(output) > 0;
                boolean usableSidecar = Files.isRegularFile(
                        sidecar, LinkOption.NOFOLLOW_LINKS);
                boolean manifestMatches = false;
                boolean authenticatedMatch = false;
                String validationDetail = "";
                if (usableOutput && usableSidecar) {
                    byte[] expectedMessage = Files.readAllBytes(message);
                    manifestMatches = VerifiedOutputManifest.matches(
                            output, image, message,
                            context.range().minQf(), context.range().maxQf(),
                            APPLICATION_VERSION, context.stcHeight(),
                            context.geometrySelection().singleModeFor(pair.geometry()), pair.geometry());
                    if (!manifestMatches) {
                        validationDetail =
                                "version/hash sidecar is mismatched";
                    }
                    if (manifestMatches) {
                        try {
                            RecoveredPayload recovered = recoverPayload(
                                    image, output, context.encodedKeys(),
                                    context.range(),
                                    new int[]{context.stcHeight()},
                                    pair.geometry());
                            selectedQf = recovered.qf();
                            authenticatedMatch = Arrays.equals(
                                    expectedMessage, recovered.plaintext());
                            if (!authenticatedMatch) {
                                validationDetail =
                                        "authenticated payload did not match the paired message";
                            }
                        } catch (Exception validationFailure) {
                            validationDetail =
                                    validationFailure.getClass().getSimpleName()
                                            + ": "
                                            + safeMessage(validationFailure);
                        }
                    }
                } else if (!usableOutput && !usableSidecar) {
                    validationDetail =
                            "output artifacts exist but are not regular files, "
                                    + "or the output is zero-length";
                } else if (!usableOutput) {
                    validationDetail =
                            "orphan sidecar or zero-length/non-regular output";
                } else {
                    validationDetail = "output has no regular verification sidecar";
                }
                if (manifestMatches && authenticatedMatch) {
                    status = "VERIFIED_EXISTING";
                    detail = "Existing output passed version/hash, extraction, "
                            + "AES-GCM/AAD, and message-equality checks.";
                } else {
                    quarantineExistingOutput(
                            output, context.outDir(), validationDetail);
                    selectedQf = embed(
                            image, message, output, context.encodedKeys(),
                            context.range(), context.constraints(),
                            context.stcHeight(),
                            context.auditMode(),
                            context.auditProfile(), context.auditEnabled(),
                            context.geometrySelection().singleModeFor(pair.geometry()), pair.geometry());
                    if (Files.isRegularFile(output)
                            && Files.size(output) > 0) {
                        status = "REBUILT_AFTER_QUARANTINE";
                        detail = validationDetail;
                    } else {
                        status = noOutputStatus(
                                context.auditEnabled(),
                                context.auditMode());
                        detail = "Existing output was quarantined; no replacement "
                                + "QF produced a fully verified profile-conforming output.";
                    }
                }
            } else {
                selectedQf = embed(
                        image, message, output, context.encodedKeys(),
                        context.range(), context.constraints(),
                        context.stcHeight(),
                        context.auditMode(),
                        context.auditProfile(), context.auditEnabled(),
                        context.geometrySelection().singleModeFor(pair.geometry()), pair.geometry());
                if (Files.isRegularFile(output) && Files.size(output) > 0) {
                    status = "COMPLETED";
                } else {
                    status = noOutputStatus(
                            context.auditEnabled(), context.auditMode());
                    detail = "Embedding returned without a stego output because no "
                            + "QF passed capacity plus the complete profile, write-back, "
                            + "authentication, and byte-equality kernel.";
                }
            }
        } catch (Exception exception) {
            FailureDiagnostics.log(exception);
            failure = exception;
            status = "FAILED_EXCEPTION";
            detail = exception.getClass().getSimpleName()
                    + ": " + safeMessage(exception);
        }

        if (context.deleteJpegsAfterEvaluation()
                && cleanupEligibleStatus(status)) {
            try {
                deletePairJpegsIfAccepted(output, status);
                String deletion;
                if (!context.auditEnabled()) {
                    deletion = "Reference and stego JPEGs deleted after committed "
                            + "embedding and authenticated verification; audit was disabled.";
                } else if (context.auditMode() == AuditMode.RESULT_ONLY) {
                    deletion = "Reference and stego JPEGs deleted after committed "
                            + "embedding, authenticated extraction, hashes, and "
                            + "detailed final-QF result-ledger publication.";
                } else {
                    deletion = "Reference and stego JPEGs deleted after committed "
                            + "embedding, authenticated extraction, evaluation, "
                            + "hashes, and QF forensic audit publication.";
                }
                detail = detail.isBlank()
                        ? deletion : detail + " " + deletion;
            } catch (Exception deletionFailure) {
                System.err.println("FAILED_STEGO_CLEANUP verified_result_preserved=true");
                FailureDiagnostics.log(deletionFailure);
                if (failure == null) {
                    failure = deletionFailure;
                    status = "FAILED_EXCEPTION";
                    detail = deletionFailure.getClass().getSimpleName()
                            + ": " + safeMessage(deletionFailure);
                } else {
                    failure.addSuppressed(deletionFailure);
                    detail = detail + " JPEG cleanup also failed: "
                            + safeMessage(deletionFailure);
                }
            }
        } else if (context.deleteJpegsAfterEvaluation()) {
            String skipped = "JPEG cleanup skipped because the pair did not "
                    + "reach an accepted verified-output status; pre-existing "
                    + "or failed artifacts were preserved.";
            detail = detail.isBlank() ? skipped : detail + " " + skipped;
        }
        return new BatchPairResult(
                pair, status, detail, selectedQf, worker, failure);
    }

    @SuppressWarnings("try")
    private static void commitBatchResult(
            BatchPairResult result,
            BatchExecutionContext context,
            BatchCounters counters
    ) throws Exception {
        BatchPair pair = result.pair();
        try (PairConsoleContext.Scope pairScope =
                     PairConsoleContext.batchPair(
                             pair.pairIndex(), context.totalPairs(),
                             pair.image(), pair.message())) {
            appendBatchManifest(
                    context.manifest(), context.header(), context.batchId(),
                    pair.pairIndex(), context.totalPairs(),
                    context.pairModulo(), context.pairRemainder(),
                    context.geometrySelection(), pair.geometry(), pair.image(),
                    result.selectedQf(), pair.message(), pair.output(),
                    result.status(), result.detail());
            counters.handled++;
            switch (result.status()) {
                case "COMPLETED" -> counters.completed++;
                case "REBUILT_AFTER_QUARANTINE" -> {
                    counters.completed++;
                    counters.rebuiltAfterQuarantine++;
                }
                case "VERIFIED_EXISTING" -> counters.skippedExisting++;
                case "NO_OUTPUT_CHECK_ATTEMPT_LEDGER",
                     "NO_OUTPUT_CHECK_RESULT_LEDGER",
                     "NO_OUTPUT_VERIFIED_KERNEL_REJECTED" -> counters.noOutput++;
                case "FAILED_EXCEPTION" -> counters.failed++;
                default -> throw new IllegalStateException(
                        "Unknown batch result status: " + result.status());
            }
            System.out.printf(
                    "PAIR %d of %d | %s | worker=%s image_file=%s "
                            + "msg_file=%s geometry=%s selected_qf=%s "
                            + "output=%s%n",
                    pair.pairIndex(), context.totalPairs(), result.status(),
                    result.worker(), pair.image().getFileName(),
                    pair.message().getFileName(), pair.geometry().label(),
                    result.selectedQf() == null
                            ? "none" : result.selectedQf(),
                    pair.output().getFileName());
        }
    }

    private record BatchPair(
            long pairIndex,
            Path image,
            Path message,
            Path output,
            SamplingGeometry geometry
    ) {}

    private static String noOutputStatus(
            boolean auditEnabled, AuditMode auditMode
    ) {
        if (!auditEnabled) return "NO_OUTPUT_VERIFIED_KERNEL_REJECTED";
        return auditMode == AuditMode.RESULT_ONLY
                ? "NO_OUTPUT_CHECK_RESULT_LEDGER"
                : "NO_OUTPUT_CHECK_ATTEMPT_LEDGER";
    }

    private record BatchPairResult(
            BatchPair pair,
            String status,
            String detail,
            Integer selectedQf,
            String worker,
            Exception failure
    ) {}

    private record BatchExecutionContext(
            Path outDir,
            String encodedKeys,
            QfRange range,
            AdaptiveQFSelector.SelectionConstraints constraints,
            AuditMode auditMode,
            AuditProfile auditProfile,
            boolean auditEnabled,
            int stcHeight,
            boolean resume,
            boolean continueOnError,
            boolean deleteJpegsAfterEvaluation,
            Path manifest,
            List<String> header,
            String batchId,
            long totalPairs,
            int pairModulo,
            int pairRemainder,
            BatchExecutorSupport.CpuSlotGate cpuSlotGate,
            GeometrySelection geometrySelection
    ) {}

    private static final class BatchCounters {
        private long handled;
        private long completed;
        private long skippedExisting;
        private long noOutput;
        private long failed;
        private long rebuiltAfterQuarantine;
    }

    static boolean cleanupEligibleStatus(String status) {
        return "COMPLETED".equals(status)
                || "REBUILT_AFTER_QUARANTINE".equals(status)
                || "VERIFIED_EXISTING".equals(status);
    }

    static void requireGroupAlignedStartIndex(
            GeometrySelection selection, long startIndex
    ) {
        if (selection == null || startIndex < 1) {
            throw new IllegalArgumentException(
                    "Geometry selection is required and --startIndex must be >= 1");
        }
        int geometries = selection.geometries().size();
        if (selection.isMultiple()
                && (startIndex - 1L) % geometries != 0L) {
            throw new IllegalArgumentException(
                    "Multiple-geometry --startIndex must begin at the first "
                            + "geometry of an image-message group (group size=" + geometries + "). "
                            + "Use --pairModulo/--pairRemainder for deterministic multi-process sharding.");
        }
    }


    static void requireGroupAlignedStartIndex(
            SamplingGeometryMode mode, long startIndex
    ) {
        if (mode == null) throw new IllegalArgumentException("Geometry mode is required");
        GeometrySelection selection = mode.includesAll()
                ? GeometrySelection.all()
                : GeometrySelection.of(mode.geometries().get(0));
        requireGroupAlignedStartIndex(selection, startIndex);
    }

    static boolean pairBelongsToShard(long pairIndex, int modulo, int remainder) {
        if (pairIndex < 1) throw new IllegalArgumentException("pairIndex must be >= 1");
        if (modulo < 1) throw new IllegalArgumentException("modulo must be >= 1");
        if (remainder < 0 || remainder >= modulo) {
            throw new IllegalArgumentException("remainder must be in 0..modulo-1");
        }
        return Math.floorMod(pairIndex - 1L, modulo) == remainder;
    }

    /**
     * Image-granular sharding.  Pair indexes run image-major (then message,
     * then geometry), so {@code (pairIndex-1)/pairsPerImage} is the image
     * ordinal.  Keeping all messages x geometries of one image in one process
     * lets the prepared (image, geometry, QF) cache serve all of them.  PAIR
     * sharding with 7 geometries and 2, 3, 4, 5, 6 or 8 shards scatters the
     * 15 messages of every (image, geometry) across all shards, so each
     * process rebuilds the same canonical references and SI-UNIWARD candidate
     * sets.  Both units are disjoint and complete; only locality differs.
     */
    static boolean imageBelongsToShard(
            long pairIndex, long pairsPerImage, int modulo, int remainder
    ) {
        if (pairIndex < 1) throw new IllegalArgumentException("pairIndex must be >= 1");
        if (pairsPerImage < 1) {
            throw new IllegalArgumentException("pairsPerImage must be >= 1");
        }
        if (modulo < 1) throw new IllegalArgumentException("modulo must be >= 1");
        if (remainder < 0 || remainder >= modulo) {
            throw new IllegalArgumentException("remainder must be in 0..modulo-1");
        }
        long imageOrdinal = (pairIndex - 1L) / pairsPerImage;
        return Math.floorMod(imageOrdinal, modulo) == remainder;
    }

    static boolean deletePairJpegsIfAccepted(
            Path stego, String status
    ) throws Exception {
        if (!cleanupEligibleStatus(status)) return false;
        VerifiedOutputManifest.delete(stego);
        Files.deleteIfExists(java.util.Objects.requireNonNull(stego, "stegoOutputPath"));
        if (Files.exists(stego)
                || Files.exists(VerifiedOutputManifest.sidecarFor(stego))) {
            throw new java.io.IOException(
                    "Could not remove verified temporary stego artifacts: "
                            + stego.toAbsolutePath());
        }
        return true;
    }

    static boolean outputArtifactsExist(Path output) {
        return Files.exists(output, LinkOption.NOFOLLOW_LINKS)
                || Files.exists(
                        VerifiedOutputManifest.sidecarFor(output),
                        LinkOption.NOFOLLOW_LINKS);
    }

    static void quarantineExistingOutput(
            Path output, Path outDir, String reason
    ) throws Exception {
        Path sidecar = VerifiedOutputManifest.sidecarFor(output);
        List<Path> present = new ArrayList<>();
        for (Path artifact : List.of(output, sidecar)) {
            if (!Files.exists(artifact, LinkOption.NOFOLLOW_LINKS)) continue;
            if (!Files.isRegularFile(
                    artifact, LinkOption.NOFOLLOW_LINKS)) {
                throw new java.io.IOException(
                        "Refusing to quarantine a non-regular output "
                                + "artifact: " + artifact.toAbsolutePath());
            }
            present.add(artifact);
        }
        if (present.isEmpty()) {
            throw new java.io.IOException(
                    "No stale output artifact exists to quarantine: "
                            + output.toAbsolutePath());
        }
        Path quarantine = outDir.resolve("quarantine");
        Files.createDirectories(quarantine);
        String token = UUID.randomUUID().toString();
        List<Path> moved = new ArrayList<>();
        for (Path artifact : present) {
            Path destination = quarantine.resolve(
                    artifact.getFileName() + "." + token + ".stale");
            Files.move(artifact, destination);
            moved.add(destination);
        }
        System.err.println("Quarantined unverified output artifacts: " + moved
                + " | reason=" + reason);
    }

    private static List<Path> discoverFiles(Path root, boolean recursive, String extension) throws Exception {
        String normalizedExtension = extension == null ? "" : extension.trim().toLowerCase(java.util.Locale.ROOT);
        try (java.util.stream.Stream<Path> stream = recursive ? Files.walk(root) : Files.list(root)) {
            return stream.filter(Files::isRegularFile)
                    .filter(p -> normalizedExtension.equals("*") || normalizedExtension.isBlank()
                            || p.getFileName().toString().toLowerCase(java.util.Locale.ROOT).endsWith(normalizedExtension))
                    .sorted(java.util.Comparator.comparing(p -> root.relativize(p).toString()))
                    .toList();
        }
    }

    private static String relativeKey(Path root, Path file) {
        String relative = root.toAbsolutePath().normalize().relativize(file.toAbsolutePath().normalize()).toString();
        int dot = relative.lastIndexOf('.');
        if (dot > 0) relative = relative.substring(0, dot);
        return relative.replaceAll("[^A-Za-z0-9._-]+", "__");
    }

    static String batchOutputName(
            Path pngRoot,
            Path image,
            Path messageRoot,
            Path message
    ) {
        return batchOutputName(
                pngRoot, image, messageRoot, message,
                SamplingGeometry.JPEG_420);
    }

    static String batchOutputName(
            Path pngRoot,
            Path image,
            Path messageRoot,
            Path message,
            SamplingGeometry geometry
    ) {
        String imageKey = relativeKey(pngRoot, image);
        String messageKey = relativeKey(messageRoot, message);
        return "i" + imageKey.length() + "_" + imageKey
                + "__m" + messageKey.length() + "_" + messageKey
                + "__" + geometry.outputTag() + "_stego.jpg";
    }

    static Path outputForGeometry(
            Path requested,
            GeometrySelection selection,
            SamplingGeometry geometry
    ) {
        if (requested == null || selection == null || geometry == null) {
            throw new IllegalArgumentException(
                    "output, geometry selection, and geometry cannot be null");
        }
        if (!selection.contains(geometry)) {
            throw new IllegalArgumentException(
                    "Geometry " + geometry.label()
                            + " is not selected by " + selection.token());
        }
        if (!selection.isMultiple()) return requested;
        Path namePath = requested.getFileName();
        if (namePath == null) {
            throw new IllegalArgumentException(
                    "Requested output must have a filename");
        }
        String name = namePath.toString();
        String lower = name.toLowerCase(java.util.Locale.ROOT);
        int extensionStart = lower.endsWith(".jpeg")
                ? name.length() - 5
                : lower.endsWith(".jpg")
                ? name.length() - 4 : name.length();
        String geometryName = name.substring(0, extensionStart)
                + "_" + geometry.outputTag()
                + name.substring(extensionStart);
        return requested.resolveSibling(geometryName);
    }


    static Path outputForGeometry(
            Path requested,
            SamplingGeometryMode mode,
            SamplingGeometry geometry
    ) {
        GeometrySelection selection = mode.includesAll()
                ? GeometrySelection.all()
                : GeometrySelection.of(mode.geometries().get(0));
        return outputForGeometry(requested, selection, geometry);
    }
    private static void validateDistinctBatchOutputs(
            Path pngRoot,
            List<Path> images,
            Path messageRoot,
            List<Path> messages,
            List<SamplingGeometry> geometries
    ) {
        Map<String, String> owners = new LinkedHashMap<>();
        for (Path image : images) {
            for (Path message : messages) {
                for (SamplingGeometry geometry : geometries) {
                    String outputName = batchOutputName(
                            pngRoot, image, messageRoot, message,
                            geometry);
                    String owner = image.toAbsolutePath().normalize()
                            + " | " + message.toAbsolutePath().normalize()
                            + " | " + geometry.label();
                    String previous = owners.putIfAbsent(
                            outputName, owner);
                    if (previous != null && !previous.equals(owner)) {
                        throw new IllegalArgumentException(
                                "Sanitized batch paths collide at output "
                                        + outputName + ": " + previous
                                        + " versus " + owner);
                    }
                }
            }
        }
    }

    private static void appendBatchManifest(Path manifest, List<String> header, String batchId, long pairIndex,
                                            long totalPairs, int pairModulo, int pairRemainder,
                                            GeometrySelection geometrySelection,
                                            SamplingGeometry geometry,
                                            Path image, Integer selectedQf,
                                            Path message, Path output,
                                            String status, String detail) throws Exception {
        var group = thesis.rap.analysis.MessageGroup.from(
                message.getFileName().toString());
        CsvUtil.append(manifest, header, List.of(
                batchId, group.label(), String.valueOf(group.number()),
                group.name(), String.valueOf(pairIndex), String.valueOf(totalPairs),
                String.valueOf(pairModulo), String.valueOf(pairRemainder),
                ExecutionBackend.mode().name(),
                System.getenv().getOrDefault("RAP_PHYSICAL_GPU_ID", "CPU_OR_UNDECLARED"),
                System.getenv().getOrDefault("RAP_CUDA_DEVICE", "0"),
                geometrySelection.token(), geometry.label(),
                selectedQf == null ? "" : String.valueOf(selectedQf),
                image.toAbsolutePath().toString(), message.toAbsolutePath().toString(), output.toAbsolutePath().toString(),
                status, detail == null ? "" : detail));
    }

    private static byte[] coverBits(JpegImage img, List<Candidate> candidates) {
        byte[] bits = new byte[candidates.size()];

        for (int i = 0; i < candidates.size(); i++) {
            Candidate c = candidates.get(i);
            int q = img.component(c.component).getCoeff(c.blockIndex, c.k);
            bits[i] = (byte) Math.floorMod(q, 2);
        }

        return bits;
    }

    private static void validateInMemoryCandidatePostconditions(List<Candidate> candidates, JpegImage intendedStego) {
        for (Candidate c : candidates) {
            int actual = intendedStego.component(c.component).getCoeff(c.blockIndex, c.k);
            int delta = actual - c.qRef;
            if (delta != 0 && delta != 1 && delta != -1) {
                throw new IllegalStateException("Candidate changed by non-unit delta=" + delta + ": " + c.describe());
            }
            if ((delta == 1 && JpegMath.isWetCost(c.rhoPlus))
                    || (delta == -1 && JpegMath.isWetCost(c.rhoMinus))) {
                throw new IllegalStateException("Wet directional change survived postcondition checks: " + c.describe());
            }
        }
    }

    private static TransitionCounts countTransitions(List<Candidate> candidates, JpegImage stego) {
        long zp = 0, zn = 0, pz = 0, nz = 0, op = 0, om = 0;
        for (Candidate c : candidates) {
            int q = stego.component(c.component).getCoeff(c.blockIndex, c.k);
            int d = q - c.qRef;
            if (c.qRef == 0 && q == 1) zp++;
            else if (c.qRef == 0 && q == -1) zn++;
            else if (c.qRef == 1 && q == 0) pz++;
            else if (c.qRef == -1 && q == 0) nz++;
            else if (d == 1) op++;
            else if (d == -1) om++;
        }
        return new TransitionCounts(zp, zn, pz, nz, op, om);
    }

    private record TransitionCounts(long zeroToPlus, long zeroToMinus, long plusToZero,
                                    long minusToZero, long otherPlus, long otherMinus) {
        long activations() { return zeroToPlus + zeroToMinus; }
        long deactivations() { return plusToZero + minusToZero; }
        long zeroBinDelta() { return deactivations() - activations(); }
    }

    private static JpegImage validateWrittenCoefficientState(
            Path output, List<Candidate> candidates, JpegImage intendedStego) {
        JpegImage written = JpegCore.readCoefficients(output.toString());
        if (written.width != intendedStego.width
                || written.height != intendedStego.height
                || written.components.length != intendedStego.components.length) {
            throw new IllegalStateException(
                    "Post-write JPEG layout differs from the intended stego image");
        }
        for (int component = 0;
             component < intendedStego.components.length;
             component++) {
            JpegComponent expected = intendedStego.component(component);
            JpegComponent actualComponent = written.component(component);
            if (expected.widthBlocks != actualComponent.widthBlocks
                    || expected.heightBlocks != actualComponent.heightBlocks
                    || expected.hSamplingFactor
                    != actualComponent.hSamplingFactor
                    || expected.vSamplingFactor
                    != actualComponent.vSamplingFactor
                    || expected.quantTableIndex
                    != actualComponent.quantTableIndex
                    || !Arrays.equals(
                    expected.quantTable, actualComponent.quantTable)
                    || !Arrays.equals(
                    expected.coefficients,
                    actualComponent.coefficients)) {
                throw new IllegalStateException(
                        "Post-write component/table/coefficient mismatch "
                                + "for component " + component);
            }
        }
        for (Candidate c : candidates) {
            int intended = intendedStego.component(c.component).getCoeff(c.blockIndex, c.k);
            int actual = written.component(c.component).getCoeff(c.blockIndex, c.k);
            if (actual != intended) {
                throw new IllegalStateException("Post-write coefficient mismatch for " + c.describe()
                        + ": intended=" + intended + ", written=" + actual);
            }
        }
        validateInMemoryCandidatePostconditions(candidates, written);
        return written;
    }

    private static VerificationResult verifyEmbeddedPayload(
            SourceImageContext sourceContext,
            CandidateGenerator.Context candidateContext,
            JpegImage writtenStego,
            String encodedKeys,
            int selectedQF,
            byte[] originalMessage,
            boolean qfForensicAudit,
            int stcHeight,
            SamplingGeometry geometry
    ) throws Exception {
        try {
            RecoveredPayload recovered = recoverPayload(
                    sourceContext, candidateContext, writtenStego, encodedKeys,
                    new QfRange(selectedQF, selectedQF),
                    new int[]{stcHeight}, geometry);
            boolean messageMatches = Arrays.equals(originalMessage, recovered.plaintext());
            return new VerificationResult(true, true, messageMatches, messageMatches ? "OK" : "message mismatch");
        } catch (PayloadRecoveryRejectedException e) {
            if (qfForensicAudit) {
                System.err.println(
                        "Round-trip verification warning: "
                                + e.getClass().getSimpleName() + ": "
                                + e.getMessage());
            }
            return new VerificationResult(false, false, false, e.toString());
        }
    }

    private static QfSelectionMetrics qfSelectionMetrics(
            AdaptiveQFSelector.QFDecision decision,
            int payloadBits,
            int selectedCandidateCount,
            int engineeringCapacity
    ) {
        int tested = decision.qfBandRows().size();
        int failedBefore = Math.max(0, tested - 1);
        Integer previousQf = failedBefore > 0 ? decision.qfBandRows().get(tested - 2).qf : null;
        int candidateHeadroom = selectedCandidateCount - payloadBits;
        int capacityHeadroom = engineeringCapacity - payloadBits;
        double candidatePct = payloadBits == 0 ? Double.NaN : 100.0 * candidateHeadroom / payloadBits;
        double capacityPct = payloadBits == 0 ? Double.NaN : 100.0 * capacityHeadroom / payloadBits;
        return new QfSelectionMetrics(tested, failedBefore, previousQf, selectedCandidateCount,
                candidateHeadroom, engineeringCapacity, capacityHeadroom, candidatePct, capacityPct);
    }

    private static void printPayloadBreakdown(PayloadPackage.PayloadLayout p) {
        System.out.println("========== PAYLOAD / CRYPTOGRAPHIC BREAKDOWN ==========");
        System.out.printf("Raw message: %d bytes%n", p.rawMessageBytes());
        System.out.printf("Huffman candidate size: %d bytes | used=%s%n", p.huffmanCandidateBytes(), p.huffmanUsed());
        System.out.printf("Selected content: %d bytes | preprocessing metadata: %d bytes%n",
                p.contentBytes(), p.preprocessingMetadataBytes());
        System.out.printf("Prepared plaintext / AES-GCM ciphertext: %d / %d bytes%n",
                p.preparedPlaintextBytes(), p.ciphertextBytes());
        System.out.printf("Header: public-marker=%d, embedded-QF=%d, length-field=%d, nonce=%d, tag=%d, total=%d bytes%n",
                p.protocolMarkerBytes(), p.qfFieldBytes(), p.ciphertextLengthFieldBytes(), p.nonceBytes(),
                p.authenticationTagBytes(), p.fixedHeaderBytes());
        System.out.printf(
                "AES-GCM AAD: geometry-domain || QF || geometry-signature || "
                        + "ciphertext-length-u32 || SHA-256(XOR-key commitment) = %d authenticated "
                        + "bytes (not embedded bytes)%n",
                PayloadPackage.GEOMETRY_AAD_LEN);
        System.out.printf(
                "Final embedded payload: %d bytes = %d bits | "
                        + "fixed framing=%d bytes | net expansion=%+d bytes%n",
                p.totalPayloadBytes(), p.totalPayloadBits(),
                p.fixedFramingOverheadBytes(),
                p.netPayloadExpansionBytes());
        System.out.printf("Payload expansion ratio vs raw message: %.9f | content compression ratio: %.9f%n",
                p.expansionRatio(), p.compressionRatio());
        System.out.println("=======================================================");
    }

    private static void printJpegMetadata(JpegMetadata m) {
        System.out.println("========== JPEG / CODEC REPRODUCIBILITY METADATA ==========");
        System.out.printf("Dimensions: %dx%d | components=%d | subsampling=%s%n",
                m.width(), m.height(), m.componentCount(), m.subsampling());
        System.out.printf("Sampling factors: %s | MCU grid=%dx%d (%d MCUs)%n",
                m.samplingFactors(), m.mcuColumns(), m.mcuRows(), m.mcuCount());
        System.out.printf("Blocks Y/Cb/Cr: %d / %d / %d%n", m.yBlockCount(), m.cbBlockCount(), m.crBlockCount());
        System.out.printf("Quantization-table indexes: %s%n", m.quantTableIndexes());
        System.out.printf("Quantization-table SHA-256: %s%n", m.quantizationTableSha256());
        System.out.println("===========================================================");
    }

    private static void printQfSelectionMargin(QfSelectionMetrics q) {
        System.out.println("========== ADAPTIVE QF SELECTION MARGIN ==========");
        System.out.printf("QFs tested: %d | failed before selection: %d | previous tested QF: %s%n",
                q.qfsTested(), q.failedQfsBeforeSelection(), q.previousTestedQf() == null ? "none" : q.previousTestedQf());
        System.out.printf("Selected candidates: %d | candidate headroom: %+d bits (%.6f%% of payload)%n",
                q.selectedCandidateCount(), q.candidateHeadroomBits(), q.candidateHeadroomPercentOfPayload());
        System.out.printf("RAP engineering capacity C_eng: %d | capacity headroom: %+d bits (%.6f%% of payload)%n",
                q.engineeringCapacityBits(), q.capacityHeadroomBits(), q.capacityHeadroomPercentOfPayload());
        System.out.println("==================================================");
    }

    private static long nanosToMillis(long nanos) {
        return Math.max(0L, nanos / 1_000_000L);
    }

    private record VerificationResult(
            boolean extractionSuccess,
            boolean aesGcmVerified,
            boolean messageMatches,
            String detail
    ) {}

    private record ProductionInputEvidence(
            long imageBytes,
            int width,
            int height,
            String imageSha256,
            long messageBytes,
            String messageSha256,
            int libjpegVersion
    ) {}

    /** Verified production-input metadata plus the already decoded source context. */
    private record VerifiedProductionInput(
            ProductionInputEvidence evidence,
            SourceImageContext sourceContext
    ) {
        VerifiedProductionInput {
            java.util.Objects.requireNonNull(evidence, "evidence");
            java.util.Objects.requireNonNull(sourceContext, "sourceContext");
        }
    }

    private record ProductionOutputEvidence(
            long stegoBytes,
            int width,
            int height,
            int components,
            String stegoSha256
    ) {}

    private record RecoveredPayload(
            int qf,
            int stcHeight,
            SamplingGeometry geometry,
            int packedBytes,
            byte[] plaintext
    ) {
        private RecoveredPayload {
            plaintext = Arrays.copyOf(plaintext, plaintext.length);
        }
        @Override public byte[] plaintext() { return Arrays.copyOf(plaintext, plaintext.length); }
    }

    private static final class PayloadRecoveryRejectedException extends IllegalStateException {
        private static final long serialVersionUID = 1L;
        PayloadRecoveryRejectedException(String message) { super(message); }
    }

    private static final class VerificationFailure extends Exception {
        private static final long serialVersionUID = 1L;
        VerificationFailure(String message) { super(message); }
    }

    private static Path writeAdaptiveQfBandCsv(
            Path png,
            Path msg,
            String runId,
            List<QfBandCoefficientReport.Row> rows
    ) throws Exception {
        Path outDir = thesis.rap.runtime.RuntimePaths.results("qf_band_reports");
        Files.createDirectories(outDir);

        String imageStem = fileStem(png.getFileName().toString());
        String msgStem = fileStem(msg.getFileName().toString());
        Path csv = outDir.resolve(imageStem + "_" + msgStem + "_" + runId + "_qf_band_report.csv");

        QfBandCoefficientReport.writeCsv(csv, rows);
        return csv;
    }

    private static String fileStem(String name) {
        if (name == null || name.isBlank()) return "unknown";
        int dot = name.lastIndexOf('.');
        String stem = dot > 0 ? name.substring(0, dot) : name;
        return stem.replaceAll("[^A-Za-z0-9._-]", "_");
    }

    private static String safeFileName(Path path) {
        if (path == null) return "";
        Path name = path.getFileName();
        return name == null ? path.toString() : name.toString();
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.isBlank() ? error.toString() : message.replaceAll("[\r\n]+", " ");
    }

    private static String inputImageHash(EmbedRunState state) {
        return state.inputEvidence == null
                ? "" : state.inputEvidence.imageSha256();
    }

    private static String inputMessageHash(EmbedRunState state) {
        return state.inputEvidence == null
                ? "" : state.inputEvidence.messageSha256();
    }

    private static void appendResultAudit(
            String runId,
            Path png,
            Path msg,
            Path out,
            QfRange qfRange,
            AdaptiveQFSelector.SelectionConstraints selectionConstraints,
            EmbedRunState state,
            String status,
            String failureReasonCode,
            String detail,
            EvaluationSummary evaluation,
            String evaluationUnavailableReason
    ) throws Exception {
        if (state.embeddingElapsedMillis == null
                && state.embeddingStartNanos != 0L) {
            state.embeddingElapsedMillis = nanosToMillis(
                    System.nanoTime() - state.embeddingStartNanos);
        }
        ResultAuditWriter.Builder builder =
                ResultAuditWriter.Entry.builder()
                        .identity(
                                state.startedUtc, Instant.now().toString(),
                                runId, APPLICATION_VERSION,
                                "EMBED_AND_AUTHENTICATED_EXTRACT")
                        .paths(png, msg, out)
                        .geometry(
                                state.geometryMode,
                                state.geometry)
                        .qfSearch(
                                qfRange.minQf(), qfRange.maxQf(),
                                selectionConstraints.policyName())
                        .outcome(
                                status,
                                state.embeddingSucceeded
                                        && state.outputEvidence != null,
                                state.finalQf.extractionVerified(),
                                state.finalQf.writebackVerified(),
                                state.finalQf.nativeStcVerified(),
                                state.finalQf.aesGcmVerified(),
                                state.finalQf.exactMessageVerified())
                        .hashes(
                                inputImageHash(state),
                                inputMessageHash(state),
                                state.outputEvidence == null
                                        ? ""
                                        : state.outputEvidence.stegoSha256())
                        .failure(failureReasonCode, detail)
                        .finalQf(state.finalQf)
                        .payload(state.payloadLayout);
        if (state.inputEvidence != null) {
            builder.inputEvidence(
                    state.inputEvidence.imageBytes(),
                    state.inputEvidence.width(),
                    state.inputEvidence.height(),
                    state.inputEvidence.messageBytes(),
                    state.inputEvidence.libjpegVersion());
        }
        if (state.outputEvidence != null) {
            builder.outputEvidence(
                    state.outputEvidence.stegoBytes(),
                    state.outputEvidence.width(),
                    state.outputEvidence.height(),
                    state.outputEvidence.components());
        }
        if (evaluation != null) {
            builder.evaluation(
                    evaluation,
                    state.embeddingElapsedMillis == null
                            ? 0L : state.embeddingElapsedMillis,
                    state.evaluationElapsedMillis == null
                            ? 0L : state.evaluationElapsedMillis);
        } else {
            builder.evaluationUnavailable(
                    evaluationUnavailableReason,
                    state.embeddingElapsedMillis);
        }
        builder.exception(state.failure);
        state.operation = "AUDIT_WRITE";
        ResultAuditWriter.append(builder.build());
    }

    private static String failureReasonCode(Throwable error) {
        if (error instanceof
                DiskSpaceGuard.InsufficientStorageException) {
            return "STORAGE_RESERVE_REJECTED";
        }
        if (error instanceof VerificationFailure) {
            return "FAILED_VERIFICATION";
        }
        if (error instanceof java.security.GeneralSecurityException) {
            return "FAILED_CRYPTO";
        }
        if (error instanceof java.io.IOException) {
            return "FAILED_IO";
        }
        if (error instanceof ArithmeticException) {
            return "FAILED_ARITHMETIC";
        }
        if (error instanceof IllegalArgumentException) {
            return "FAILED_INPUT";
        }
        if (error instanceof LinkageError) {
            return "FAILED_NATIVE_DEPENDENCY";
        }
        return "FAILED_EXCEPTION";
    }

    private static void validateReadableFile(Path path, String label) {
        validateReadableFile(path, label, false);
    }

    private static void validateReadableFile(
            Path path, String label, boolean allowEmpty
    ) {
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException(label + " does not exist or is not a regular file: " + path.toAbsolutePath());
        }
        if (!Files.isReadable(path)) {
            throw new IllegalArgumentException(label + " is not readable: " + path.toAbsolutePath());
        }
        try {
            if (!allowEmpty && Files.size(path) == 0L) {
                throw new IllegalArgumentException(
                        label + " is empty: " + path.toAbsolutePath());
            }
        } catch (java.io.IOException failure) {
            throw new IllegalArgumentException(
                    "Cannot determine " + label + " size: "
                            + path.toAbsolutePath(), failure);
        }
    }

    private static VerifiedProductionInput verifyProductionInputs(
            Path png, Path message
    ) throws Exception {
        SourceImageContext sourceContext = SourceImageContext.load(png);
        var decoded = sourceContext.sourceImage();
        if (decoded.getWidth() < 1 || decoded.getHeight() < 1) {
            throw new IllegalArgumentException(
                    "Decoded PNG dimensions must be positive: "
                            + png.toAbsolutePath());
        }
        long imageBytes = sourceContext.sourceByteCount();
        long messageBytes = Files.size(message);
        if (messageBytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Message is too large for the exact in-memory payload "
                            + "pipeline: " + messageBytes + " bytes");
        }

        int libjpegVersion = JpegCore.libjpegVersion();
        if (libjpegVersion <= 0) {
            throw new IllegalStateException(
                    "Native libjpeg reported an invalid version: "
                            + libjpegVersion);
        }
        ProductionInputEvidence evidence = new ProductionInputEvidence(
                imageBytes, decoded.getWidth(), decoded.getHeight(),
                sourceContext.sourceSha256(), messageBytes,
                FileFingerprint.sha256(message), libjpegVersion);
        return new VerifiedProductionInput(evidence, sourceContext);
    }

    private static ProductionOutputEvidence
    verifyCommittedProductionOutput(
            Path stego,
            Path png,
            Path message,
            ProductionInputEvidence inputEvidence
    ) throws Exception {
        validateReadableFile(stego, "Committed stego JPEG");
        verifyProductionInputStability(png, message, inputEvidence);
        JpegImage actual = JpegCore.readCoefficients(stego.toString());
        if (actual.width != inputEvidence.width()
                || actual.height != inputEvidence.height()
                || actual.components.length != 3) {
            throw new VerificationFailure(
                    "Committed stego JPEG dimensions/components do not match "
                            + "the verified PNG input");
        }
        return new ProductionOutputEvidence(
                Files.size(stego), actual.width, actual.height,
                actual.components.length, FileFingerprint.sha256(stego));
    }

    private static void verifyProductionInputStability(
            Path png,
            Path message,
            ProductionInputEvidence inputEvidence
    ) throws Exception {
        String currentImageHash = FileFingerprint.sha256(png);
        String currentMessageHash = FileFingerprint.sha256(message);
        if (!currentImageHash.equals(inputEvidence.imageSha256())
                || !currentMessageHash.equals(
                inputEvidence.messageSha256())) {
            throw new VerificationFailure(
                    "An input file changed during processing; evidence/output "
                            + "cannot be attributed to one stable "
                            + "image/message pair");
        }
    }

    private static void ensureParentDirectory(Path path) throws Exception {
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
    }

    record QfRange(int minQf, int maxQf) {}

    private static AdaptiveQFSelector.SelectionConstraints selectionConstraints(Map<String, String> options) {
        String policy = options.getOrDefault("--selectionPolicy", "capacity").trim().toLowerCase(java.util.Locale.ROOT);
        if (!"capacity".equals(policy)
                || options.containsKey("--maxCandidateLoading")
                || options.containsKey("--maxCandidateBpnzAC")
                || options.containsKey("--maxStcUtilization")) {
            throw new IllegalArgumentException(
                    "The six-algorithm conformance profile fixes selectionPolicy=capacity and permits no additional QF gates");
        }
        return AdaptiveQFSelector.SelectionConstraints.capacityOnly();
    }

    private static AuditProfile auditProfile(
            Map<String, String> options
    ) {
        String value = options.get("--auditProfile");
        if (value == null || value.isBlank()) {
            value = System.getenv("RAP_AUDIT_PROFILE");
        }
        if (value == null || value.isBlank()) {
            value = AuditProfile.forensicDefault().name();
        }
        return AuditProfile.parse(value);
    }

    static AuditMode auditMode(
            Map<String, String> options
    ) {
        String value = options.get("--auditMode");
        if (value == null || value.isBlank()) {
            value = System.getenv("RAP_AUDIT_MODE");
        }
        if (value == null || value.isBlank()) {
            value = AuditMode.RESULT_ONLY.name();
        }
        return AuditMode.parse(value);
    }

    static boolean auditEnabled(
            Map<String, String> options
    ) {
        String raw = options.get("--auditEnabled");
        if (raw == null || raw.isBlank()) {
            raw = System.getenv("RAP_AUDIT_ENABLED");
        }
        if (raw == null || raw.isBlank()) return true;
        return parseStrictBoolean("--auditEnabled", raw);
    }

    static BatchExecutorSupport.Config batchThreading(
            Map<String, String> options
    ) {
        BatchExecutorSupport.ThreadMode mode =
                BatchExecutorSupport.ThreadMode.parse(
                        options.getOrDefault("--threadMode", "platform"));
        String workerOption = options.get("--workers");
        int workers;
        if (mode == BatchExecutorSupport.ThreadMode.VIRTUAL) {
            String requested = workerOption == null
                    ? "auto" : workerOption.trim();
            workers = requested.equalsIgnoreCase("auto")
                    ? BatchExecutorSupport.ALL_DISCOVERED_PAIR_TASKS
                    : positiveIntOption(options, "--workers", 1);
        } else {
            workers = positiveIntOption(options, "--workers", 1);
        }
        int defaultCarriers = Math.max(
                1, Runtime.getRuntime().availableProcessors());
        int carriers = positiveIntOption(
                options, "--carrierParallelism", defaultCarriers);
        int cpuSlots = positiveIntOption(
                options, "--cpuSlots",
                mode == BatchExecutorSupport.ThreadMode.VIRTUAL
                        ? carriers : Math.min(workers, carriers));
        if (carriers > BatchExecutorSupport.MAX_RESOURCE_PARALLELISM) {
            throw new IllegalArgumentException(
                    "--carrierParallelism cannot exceed "
                            + BatchExecutorSupport.MAX_RESOURCE_PARALLELISM);
        }
        if (cpuSlots > BatchExecutorSupport.MAX_RESOURCE_PARALLELISM) {
            throw new IllegalArgumentException(
                    "--cpuSlots cannot exceed "
                            + BatchExecutorSupport.MAX_RESOURCE_PARALLELISM);
        }
        if (mode == BatchExecutorSupport.ThreadMode.PLATFORM
                && cpuSlots > workers) {
            throw new IllegalArgumentException(
                    "Platform --cpuSlots cannot exceed --workers");
        }
        return new BatchExecutorSupport.Config(
                workers, carriers, cpuSlots, mode);
    }

    private static long nonnegativeLongOption(
            Map<String, String> options,
            String name,
            long defaultValue
    ) {
        String text = options.getOrDefault(
                name, Long.toString(defaultValue));
        try {
            long value = Long.parseLong(text);
            if (value < 0L) {
                throw new IllegalArgumentException(
                        name + " must be nonnegative");
            }
            return value;
        } catch (NumberFormatException invalid) {
            throw new IllegalArgumentException(
                    name + " must be an integer: " + text, invalid);
        }
    }

    private static int positiveIntOption(
            Map<String, String> options,
            String name,
            int defaultValue
    ) {
        String text = options.getOrDefault(
                name, Integer.toString(defaultValue));
        try {
            int value = Integer.parseInt(text);
            if (value < 1) {
                throw new IllegalArgumentException(
                        name + " must be at least one");
            }
            return value;
        } catch (NumberFormatException invalid) {
            throw new IllegalArgumentException(
                    name + " must be an integer: " + text, invalid);
        }
    }

    static int embeddingStcHeight(Map<String, String> options) {
        String value = options.get("--stcHeight");
        if (value == null || value.isBlank()) {
            value = System.getenv("RAP_STC_HEIGHT");
        }
        if (value == null || value.isBlank()) {
            return StcCarrierPlan.DEFAULT_STC_HEIGHT;
        }
        if ("auto".equalsIgnoreCase(value.trim())) {
            throw new IllegalArgumentException(
                    "--stcHeight=auto is extraction-only; embedding must "
                            + "select 10 or 12 explicitly");
        }
        try {
            int height = Integer.parseInt(value.trim());
            StcCarrierPlan.validateSupportedHeight(height);
            return height;
        } catch (NumberFormatException invalid) {
            throw new IllegalArgumentException(
                    "--stcHeight must be 10 or 12 for embedding: "
                            + value, invalid);
        }
    }

    static int[] extractionStcHeights(Map<String, String> options) {
        String value = options.get("--stcHeight");
        if (value == null || value.isBlank()) {
            value = System.getenv("RAP_STC_HEIGHT");
        }
        if (value == null || value.isBlank()
                || "auto".equalsIgnoreCase(value.trim())) {
            return Arrays.copyOf(
                    AUTO_EXTRACTION_HEIGHTS,
                    AUTO_EXTRACTION_HEIGHTS.length);
        }
        try {
            int height = Integer.parseInt(value.trim());
            StcCarrierPlan.validateSupportedHeight(height);
            return new int[]{height};
        } catch (NumberFormatException invalid) {
            throw new IllegalArgumentException(
                    "--stcHeight must be auto, 10, or 12 for extraction: "
                            + value, invalid);
        }
    }

    static QfRange qfRange(Map<String, String> options) {
        int minQf = Integer.parseInt(options.getOrDefault("--minQf", String.valueOf(DEFAULT_MIN_QF)));
        int maxQf = Integer.parseInt(options.getOrDefault("--maxQf", String.valueOf(DEFAULT_MAX_QF)));
        if (minQf != DEFAULT_MIN_QF || maxQf != DEFAULT_MAX_QF) {
            throw new IllegalArgumentException("Invalid conformance QF range: " + minQf + "-" + maxQf
                    + ". The adaptive algorithms require the complete inclusive range 75-100.");
        }
        return new QfRange(minQf, maxQf);
    }

    static GeometrySelection geometrySelection(
            Map<String, String> options
    ) {
        return GeometrySelection.parse(options.get("--geometryMode"));
    }

    
    static SamplingGeometryMode geometryMode(Map<String, String> options) {
        return SamplingGeometryMode.parse(options.get("--geometryMode"));
    }

    private static Map<String, String> parse(String[] a) {
        Map<String, String> m = new LinkedHashMap<>();

        for (int i = 0; i < a.length; i += 2) {
            if (i + 1 >= a.length) {
                throw new IllegalArgumentException("Missing value for " + a[i]);
            }
            if (!a[i].startsWith("--") || a[i].length() <= 2) {
                throw new IllegalArgumentException(
                        "Invalid option name: " + a[i]);
            }
            if (m.putIfAbsent(a[i], a[i + 1]) != null) {
                throw new IllegalArgumentException(
                        "Duplicate option is not permitted: " + a[i]);
            }
        }

        return m;
    }

    private static void requireOnlyOptions(
            Map<String, String> options, Set<String> allowed
    ) {
        for (String option : options.keySet()) {
            if (!allowed.contains(option)) {
                throw new IllegalArgumentException(
                        "Unknown or unsupported option for this mode: " + option);
            }
        }
    }

    private static void requireDisjointBatchRoots(
            Path pngDir, Path msgDir, Path outDir
    ) throws java.io.IOException {
        Path pngRoot = canonicalPlannedPath(pngDir);
        Path msgRoot = canonicalPlannedPath(msgDir);
        Path outRoot = canonicalPlannedPath(outDir);
        if (outRoot.startsWith(pngRoot)
                || outRoot.startsWith(msgRoot)
                || pngRoot.startsWith(outRoot)
                || msgRoot.startsWith(outRoot)) {
            throw new IllegalArgumentException(
                    "Batch output directory must not equal, contain, or be contained "
                            + "by either input directory.");
        }
    }

    private static Path canonicalPlannedPath(Path path)
            throws java.io.IOException {
        Path absolute = path.toAbsolutePath().normalize();
        Path existing = absolute;
        while (existing != null && !Files.exists(existing)) {
            existing = existing.getParent();
        }
        if (existing == null) {
            throw new java.io.IOException(
                    "Cannot resolve any existing ancestor for path: " + absolute);
        }
        Path realExisting = existing.toRealPath();
        return realExisting.resolve(existing.relativize(absolute)).normalize();
    }

    private static boolean strictBoolean(
            Map<String, String> options, String option, boolean defaultValue
    ) {
        String raw = options.get(option);
        if (raw == null) return defaultValue;
        return parseStrictBoolean(option, raw);
    }

    private static boolean parseStrictBoolean(
            String option, String raw
    ) {
        return switch (raw.trim().toLowerCase(java.util.Locale.ROOT)) {
            case "true" -> true;
            case "false" -> false;
            default -> throw new IllegalArgumentException(
                    option + " must be exactly true or false: " + raw);
        };
    }

    private static String keyMaterial(Map<String, String> options) {
        boolean directTriple = options.containsKey("--encKey") || options.containsKey("--xorKey") || options.containsKey("--stcKey");
        boolean envTriple = options.containsKey("--encKeyEnv") || options.containsKey("--xorKeyEnv") || options.containsKey("--stcKeyEnv");
        if (directTriple == envTriple) {
            throw new IllegalArgumentException(
                    "Provide exactly one complete three-key set: direct --encKey/--xorKey/--stcKey or the three *KeyEnv options");
        }
        if (directTriple) {
            return new KeyMaterial(req(options, "--encKey"), req(options, "--xorKey"), req(options, "--stcKey")).encode();
        }
        return new KeyMaterial(
                requiredEnvironment(req(options, "--encKeyEnv")),
                requiredEnvironment(req(options, "--xorKeyEnv")),
                requiredEnvironment(req(options, "--stcKeyEnv"))).encode();
    }

    private static String requiredEnvironment(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) throw new IllegalArgumentException("Environment variable is not set or blank: " + name);
        return value;
    }

    private static String req(Map<String, String> m, String k) {
        String v = m.get(k);

        if (v == null) {
            throw new IllegalArgumentException("Missing " + k);
        }

        return v;
    }

    private static void usage() {
        System.out.println("""
                Commands:
                embed --png input.png --msgFile msg.txt --out stego.jpg --encKeyEnv RAP_ENC_KEY --xorKeyEnv RAP_XOR_KEY --stcKeyEnv RAP_STC_KEY [--geometryMode ALL_GEOMETRY|444|422|420|440|411|441|410|comma-separated-subset] [--stcHeight 10|12] [--auditEnabled true|false --auditMode RESULT_ONLY|QF_FORENSIC --auditProfile THESIS|CHANGES|FULL_AUDIT]
                batch-embed --pngDir input --msgDir msg --outDir output --encKeyEnv RAP_ENC_KEY --xorKeyEnv RAP_XOR_KEY --stcKeyEnv RAP_STC_KEY [--geometryMode ALL_GEOMETRY|444|422|420|440|411|441|410|comma-separated-subset] [--stcHeight 10|12] [--workers auto|N --threadMode virtual --carrierParallelism N --cpuSlots N --cpuAffinity LIST] [--auditEnabled true|false --auditMode RESULT_ONLY|QF_FORENSIC --auditProfile THESIS|CHANGES|FULL_AUDIT --deleteJpegsAfterEvaluation true|false] [batch options]
                extract --png input.png --stego stego.jpg --outFile recovered.msg --encKeyEnv RAP_ENC_KEY --xorKeyEnv RAP_XOR_KEY --stcKeyEnv RAP_STC_KEY [--geometryMode ALL_GEOMETRY|444|422|420|440|411|441|410|comma-separated-subset] [--stcHeight auto|10|12]
                summarize [--runs results/evaluation_runs.csv --positions results/evaluation_zz_positions.csv --channels results/evaluation_channels.csv --channelPositions results/evaluation_channel_zz_positions.csv --attempts results/evaluation_attempts.csv --outDir results/experiment_summary]
                detector-summary --csv detector_scores.csv [--labelColumn label --scoreColumn score --out results/detector_summary.csv]
                """);
    }
}
