# Thesis and implementation alignment

**RAP-UNIWARD 1.0.0.1 — review dated 11 November 2026**

## Scope and conclusion

The uploaded methodology names the method **Reference-Assisted Probability-Aware
UNIWARD (RAP-UNIWARD)**. The standalone application, Java namespace, JNI symbols,
accelerator file/library names, environment settings, logs, commands and release
metadata now use that identity. The application version is `1.0.0.1`.

The review compared 17 LaTeX sources, the 68-entry bibliography, the implementation,
and 19 supplied CSV summaries. The core mathematical choices inspected in the
source agree with the declared thesis profile. The supplied aggregate results are
internally consistent with the headline thesis counts and compact framing.
This does not identify the exact executable that produced those results or
independently replay the complete experiment.

Thirty Java test programs passed on the packaging host. Three integration programs
were blocked by unavailable JPEG JNI build dependencies, and two recorded GPU
skips/partial coverage. Both native STC tests passed. Therefore complete executable
conformance, GPU parity and the full experimental population remain unverified here.

## Standalone repository changes

| Area | Current implementation |
| --- | --- |
| Research/application name | RAP-UNIWARD |
| Main class | `thesis.rap.app.RapUniward` |
| Java namespace and JNI class lookups | `thesis.rap` / `thesis/rap` |
| Application release | `1.0.0.1` in emitted run metadata and `VERSION` |
| Executable JAR | `build/rap-uniward-1.0.0.1.jar` |
| CUDA/HIP libraries | `librapcuda.so`, `libraphip.so` |
| Application configuration prefix | `RAP_`; Java engine properties use `rap.` |
| Build/run entry points | `build.sh`, `run.sh`, `test.sh` and three focused scripts |
| Runtime ledgers | Version-free filenames under `results/`, configurable with `RAP_REPORT_DIR` |
| Statistical column terminology | Smoothed/clipped-support diagnostic fields use `smoothed` terminology |
| Result/evaluation CSV schema | `1.0`; forensic bundle schema 1; resume format `RAP-VERIFIED-OUTPUT-1` |
| Git contents | Active source/tests/configuration/docs; no generated corpus, outputs, logs, binaries, nested release archives or patches |
| Documentation | One README, one runtime contract, one current alignment review |

Machine-specific absolute launch paths, production key literals and fixed heap
allocations are absent from the execution instructions. The supplied 28-core
configuration is expressed through editable parameters. The new background
controller checks process identity; multi-GPU execution isolates report paths
and validates a complete merged manifest before publishing it.

Changes to application identity, output paths and metadata do not redefine the
coefficient band, cost equations, STC recurrence, balance rules or payload bytes.
The thesis explicitly preserves its authentication/ordering domain strings; these
are the only source identifiers that intentionally retain their exact spelling.
They are protocol constants, not release-history documentation.

## Six-algorithm traceability

| Thesis algorithm | Implementation sites | Correlation |
| --- | --- | --- |
| 1. Payload preparation and layout | `payload/PayloadPackage.java`: `prepareMessage`, `inspectLayout`; `payload/HuffmanCodec.java` | Serializes raw/Huffman choice, checks full serialization size, includes nine-byte preprocessing metadata, rejects prepared ciphertext outside 9–65,535 bytes |
| 2. Canonical candidate construction | `image/CanonicalReferenceJpeg.java`; `image/SourceImageContext.java`; `alg/CandidateGenerator.java`; `alg/SiUniwardCostEngine.java`; `alg/StatisticalTransitionCostModel.java` | Builds separate geometry/QF reference state; constructs every Cb/Cr rank 16–47 address; prices both legal directions; performs joint risk normalization and base sorting |
| 3. Fixed-QF authenticated embedding | `app/RapUniward.java`: fixed-QF trial path; `app/FixedQfEmbeddingKernel.java`; `app/ThreeStateEmbeddingEngine.java` | Capacity prefilter, fresh authenticated frame, keyed carrier allocation, STC/balance attempts, confinement checks, write-back and authenticated message equality |
| 4. Adaptive-QF embedding | `app/RapUniward.java`: `embedSelectedGeometries`, embedding orchestration; `app/AdaptiveQfSearch.java`: `firstVerified` | Independent message-specific ascending 75–100 search per geometry; expected infeasibility advances; unexpected integrity/I/O failures abort that geometry |
| 5. Adaptive authenticated extraction | `app/RapUniward.java`: recovery loop and `prepareRecoveryAtQf` | Detects and checks geometry, scans QF, reconstructs reference/order once per QF, tries allowed heights in order |
| 6. Fixed-QF/height extraction | `app/RapUniward.java`: fixed-QF recovery; `app/StcCarrierPlan.java`; `payload/PayloadPackage.java`: geometry-bound `unpack` | Extracts compact header, bounds body length/allocation, authenticates context/ciphertext, strictly parses plaintext/Huffman content |

Paths above are relative to `src/main/java/thesis/rap/`. The fixed-QF kernel is
only part of Algorithm 3: JPEG serialization and final authenticated round-trip
acceptance are enforced by the application orchestration around it.

## Numerical and protocol correlation

| Declared property | Source location / checked behavior | Assessment |
| --- | --- | --- |
| QF 75–100 only | `RapUniward.qfRange` rejects altered endpoints | Matches |
| Seven geometries | `SamplingGeometry`, `GeometrySelection`; exact Y/Cb/Cr factors | Matches |
| Rank 16–47 inclusive | `JpegMath.TARGET_ZZ_START_RANK=16`, `TARGET_ZZ_END_RANK=47` | Matches; 32 AC modes |
| Value-independent universe | `CandidateGenerator` checks universe size against every Cb/Cr block ×32 | Matches; zero coefficients included |
| Y/DC/out-of-band protection | `FixedQfEmbeddingKernel` checks active membership and whole-image postconditions | Matches in source; full JPEG replay blocked locally |
| JPEG AC representability | Magnitude limit 1023 in `JpegMath`; reference validation in canonical generation | Matches |
| DCT, quantization and rounding | `JpegMath`, `StandardJpegQuantization`, `CanonicalReferenceJpeg` | Declared deterministic binary64 DCT/Annex-K scaling/half-away rounding retained |
| UNIWARD residuals | 16-tap db8 filters, symmetric extension, full unit impact, corrected PAD−8 displacement | Matches inspected source and cost-engine tests |
| SI modulation | Preferred `1−2|e|`, opposite 1; offset `1e−4` | Matches; not replaced with the squared-error alternative |
| Luma context | Canonical coefficient-derived luma; `1/(1e−6+P_Y)` | Matches |
| Channel prior | Cb and Cr start equally weighted; no fixed payload quota | Matches |
| Four marginal risks | RGB-MSE energy, local contrast-normalized energy, symmetric one-count KL, absolute one-count entropy change | Matches |
| Risk normalization | Joint legal Cb/Cr directions at one geometry/QF; nearest-rank P95; clip to [0,1] | Matches |
| Risk multiplier | Four weights 0.25; strength 1; multiplier [1,2] | Matches |
| Wet treatment | Sentinel `1e13`; legal costs capped at `1e13−1` | Matches |
| Carrier ordering | Base sort by cost/component/block/index, then 64-address HMAC windows | Matches |
| Independent geometry context | Packed six sampling factors; four-byte signature in AAD/order context | Matches |
| Payload format | 2-byte ciphertext length +12-byte nonce +12-byte GCM tag | Matches: 26-byte fixed header |
| Framing | Nine-byte preprocessing metadata plus wire header | Matches: raw frame length `8 × (message_bytes + 35)` bits |
| GCM context | Production geometry-bound AAD is 72 bytes, authenticated but not embedded | Matches |
| Header allocation | 208 bits +32 reserve carriers =240 | Matches |
| Body allocation | `min(dry−240,256×body_bits)` at height 10 or 12 | Matches |
| Balance | U unrestricted; B absolute net zero-flow ≤4; S=0; B/S at most 40 attempts | Matches |
| Retry step | 0.08 × active-set median /sqrt(t+1), bound 8×median; frozen order | Matches |
| Acceptance | Written coefficients, STC syndrome, GCM and exact message equality | Implemented; complete integration not replayed locally |
| Security interpretation | Fidelity and extraction are not detector resistance | Thesis and runtime statements agree |

The code normalizes readable PNGs to 8-bit RGB, including compositing alpha on
white. It does not require the original PNG file to have been encoded as 24-bit
true color. The thesis's normalized-raster interpretation is therefore important.
Independent implementations still need the same decoding, numerical and STC
column/tie conventions; matching a nominal QF or STC height alone is insufficient.

## Supplied log-summary reconciliation

These are checks of the supplied summaries, not a rerun of their underlying JPEG
conditions. The original CSVs were read without changing values or converting
image/message identifiers to numbers.

| Check | Result |
| --- | --- |
| Raw records | 438,850 |
| Retained conditions | 372,750 |
| Superseded records | 66,100; raw minus retained reconciles |
| Population grid | 3,550 images ×15 messages ×7 geometries =372,750 |
| Verified outputs | 368,056 |
| No accepted output | 4,694 |
| Completion fraction | 98.7407109323% |
| Per-geometry population | 53,250 in each of seven groups |
| Per-message population | 24,850 in each of 15 groups |
| Per-image population | 105 for each of 3,550 textual image IDs |
| Message identity | 15 distinct SHA-256 values |
| Compact framing | All 15 message rows satisfy `framed_bits=8×(raw_bytes+35)` |
| QF distribution | All supplied terminal QFs are 75–100; verified counts reconcile |
| Invariants | 37 reported categories; every applicable count equals passed count and every failed/missing count is zero |
| Bootstrap point estimates | All 18 supplied point estimates reconcile with success counts or numeric distributions |
| Bootstrap intervals | All 18 supplied intervals contain their point estimate |

| Geometry | Verified | No accepted output | Total |
| --- | ---: | ---: | ---: |
| 4:4:4 | 52,584 | 666 | 53,250 |
| 4:2:2 | 52,701 | 549 | 53,250 |
| 4:2:0 | 52,670 | 580 | 53,250 |
| 4:4:0 | 52,756 | 494 | 53,250 |
| 4:1:1 | 52,574 | 676 | 53,250 |
| 4:4:1 | 52,434 | 816 | 53,250 |
| 4:1:0 | 52,337 | 913 | 53,250 |
| **Total** | **368,056** | **4,694** | **372,750** |

The reported reference–stego mean PSNR is 56.648535951 dB and mean SSIM is
0.999989138714. They match the thesis's rounded headline values. These statistics
condition on accepted outputs. No classifier performance or causal benefit of
individual cost terms can be concluded from them.

## Thesis findings requiring attention

1. **Source-row provenance remains unresolved.** Four extreme records refer to
   rows beyond the 178,439 rows reported for archive `101(2).zip`. The affected
   pointers are 181400 (image `img0003457`, message `014`) and 187108, 187085,
   187090 (image `img0003512`, messages `005`, `002`, `003`). Their source values
   were not changed. A mapping to the actual ledger/export is needed; the mismatch
   alone does not establish fabricated data. The appendix already acknowledges it.

2. **Bootstrap settings are not in the supplied summaries.** The evaluation
   chapter's Data Preparation section asserts 1,000 resamples and says the seed
   is recorded with the summary. `cluster_bootstrap_intervals.csv` contains only
   metric, n, mean, interval bounds and population. The appendix correctly says
   the resample count and random state cannot be independently confirmed.
   Supply the generating script/configuration or qualify the evaluation claim.

3. **Execution-time comparisons remain in the evaluation chapter.** The passages
   around lines 924 and 934–935 of the uploaded evaluation source discuss elapsed
   times and a paired elapsed-time difference. These remain inconsistent with the
   instruction to remove execution-time discussion from the thesis. Operational
   telemetry in source logs need not become a methodological comparison. The
   chapter was reviewed, not silently rewritten in this source-code task.

4. **The exact executed program cannot be identified from these 19 summaries.**
   They do not carry an application version, source commit or source-tree digest.
   The reviewed implementation and the summaries agree on inspected contracts,
   but there is no evidence here that this exact standalone source generated the
   reported 372,750-condition population. Reusing those statistics as new release
   test results would be incorrect.

5. **Thesis statistical consolidation is separate from the built-in summary CLI.**
   `summarize` reads forensic evaluation tables and computes descriptive outputs.
   It does not implement the thesis's latest-terminal-record reconciliation,
   source-cluster bootstrap or all paired geometry analyses. Their generating
   analysis code is not supplied. `tools/check_thesis.py` verifies aggregate
   consistency; it does not claim to regenerate those analyses.

6. **Appendix package references need precise scope.** It refers to `evidence/`,
   `qa/evidence_sha256.csv` and `extreme_provenance_exceptions.csv`. The 19 evidence
   CSVs are now supplied in this session, but those latter two generated files
   and a complete thesis directory were not attached. The present repository is
   source-only; it does not pretend to contain the full research evidence bundle.

7. **No complete LaTeX build was verified.** Static reference/citation checks pass,
   but this host lacks `pdflatex.fmt`, so the compilation attempt stopped before
   reading the thesis. `BUPLOGO.png` is also not attached, and the flattened upload
   filenames must be placed under the master document's expected paths.

## LaTeX structure and citations

Static checks found **197 unique labels**, **137 referenced label keys**, and
**68 citation keys matching all 68 bibliography entries**. No undefined reference
keys, duplicate labels, missing bibliography keys or duplicate bibliography keys
were found. This verifies linkage, not the factual accuracy of each cited claim.

The master expects:

| Material | Expected path |
| --- | --- |
| Master | `main.tex` |
| Title, declaration, approval, acknowledgement, abstract | `frontmatter/<name>.tex` |
| Terminology and chapters | `chapters/<name>.tex` |
| Bibliography | `chapters/bib/references.bib` |
| Evidence appendix | `appendices/appendix_evidence.tex` |
| University logo | `BUPLOGO.png` |

Internal LaTeX labels such as `eq:ahp_*` and color macro names do not appear as the
research method's visible title. Renaming them is unnecessary for semantic
correlation and would require synchronized changes to every reference.

## Verification performed on this source

| Verification | Observed result |
| --- | --- |
| Production and test Java compilation | Pass with OpenJDK 21.0.13 using matching source/target/runtime APIs |
| CLI help/version | Pass; application identifies 1.0.0.1 |
| Java software checks | 35 programs: 30 pass, 2 skip/partial, 3 dependency-blocked |
| Blocked JPEG integrations | Concurrent batch, six-algorithm end-to-end, forensic hardening; `libjpegcore` could not be built without development headers |
| Native STC self-test | Pass |
| Native STC concurrency test | 64 concurrent round trips pass |
| Python workflow checks | Five checks pass: manifest merging, invalid device lists and stopping diagnostic shard processes |
| Renamed JNI declarations | All 30 generated declarations have native source definitions |
| Package paths and shell syntax | Checked by `tools/check_repository.py` |
| Full default Java 22 build | Not executed on this host |
| CUDA/HIP compilation and device execution | Not executed on this host |
| Full corpus rerun | Not executed; original inputs and complete terminal ledger are unavailable |
| LaTeX compilation | Environment-blocked before document processing |

The Java checks cover candidate costs, statistical normalization, compact framing,
invalid-input rejection, audit records, geometry selection, deterministic ordering,
threading contracts and backend selection. Native STC checks are real compiled C++
executions. Diagnostic test images/messages are software fixtures and contribute
no thesis result values.

To repeat the source checks on a configured host:

```bash
RAP_BUILD_BACKEND=CPU ./build.sh
./test.sh
```

To repeat aggregate/reference checks against your thesis and supplied summaries:

```bash
python3 tools/check_thesis.py \
  --thesis-dir /path/to/thesis \
  --evidence-dir /path/to/summary_csvs \
  --out results/thesis_check.json
```

The current validation statuses are also recorded in [validation.json](validation.json).
Keep actual experiment data, input hashes, complete terminal ledgers and generating
analysis scripts alongside the research submission. Source code alone is not the
complete experimental record.
