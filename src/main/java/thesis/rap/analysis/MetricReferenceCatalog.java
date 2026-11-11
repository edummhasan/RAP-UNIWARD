package thesis.rap.analysis;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Writes interpretation guidance while clearly separating heuristics from universal detector thresholds. */
public final class MetricReferenceCatalog {
    private MetricReferenceCatalog() {}

    public static void write(Path out) throws IOException {
        Path parent = out.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        List<List<String>> rows = List.of(
                List.of("parameter","direction_or_reference","interpretation","threshold_type"),
                List.of("PSNR","<40 dB review; >=40 dB good; >=50 dB excellent","Practical visual-quality screening only; not a security proof","heuristic"),
                List.of("SSIM","1.0 is the theoretical maximum; >=0.98 high; >=0.99 excellent","Practical structural-similarity screening only; not a security proof","heuristic"),
                List.of("AUC","0.5 = chance-level ranking; farther above 0.5 = greater detector separation","Primary detector-facing metric on held-out source-matched data","defined reference point"),
                List.of("P_E","0.5 = chance-level balanced detection error; lower values mean easier detection","Use P_E=(P_FA+P_MD)/2 at the chosen/optimal threshold","defined reference point"),
                List.of("DCT L1 / total variation","0 is identical; lower is better","No universal undetectability cutoff; compare with cover-cover baseline and detector results","no universal cutoff"),
                List.of("KL / symmetric KL / Jensen-Shannon / Hellinger","0 is identical; lower is better","No universal undetectability cutoff; inspect tails and per-ZZ hotspots","no universal cutoff"),
                List.of("Global bpnzAC","lower generally reduces global payload pressure","Must be interpreted together with local/candidate-normalized density","no universal cutoff"),
                List.of("Candidate bpnzAC","lower generally reduces local selection-channel pressure","High local/global concentration can remain detectable despite very low global bpnzAC","no universal cutoff"),
                List.of("Candidate modification rate","lower is generally preferable","Report together with active-carrier modification rate and per-ZZ distribution","no universal cutoff"),
                List.of("Zero/nonzero transitions","Report activations, deactivations, and signed delta h(0)","RAP is three-state: individual zero crossings are permitted; U/B/S constrains their net balance, not each count to zero","methodology constraint and diagnostic"),
                List.of("Correlation delta","0 indicates unchanged measured dependency","Large absolute per-ZZ horizontal/vertical deltas may expose dependency changes","no universal cutoff")
        );
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            for (List<String> row : rows) {
                bw.write(CsvUtil.encodeRow(row));
                bw.newLine();
            }
        }
    }
}
