package thesis.rap.analysis;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Confirms verified-row filtering and source-image-clustered intervals. */
public final class EvaluationSummaryFilteringSelfTest {
    private EvaluationSummaryFilteringSelfTest() {}

    public static void main(String[] args) throws Exception {
        testActiveBandBounds();
        Path dir = Files.createTempDirectory("rap-summary-filter-");
        Path runs = dir.resolve("evaluation_runs.csv");
        Path out = dir.resolve("summary");
        String hashA = "a".repeat(64);
        String hashB = "b".repeat(64);
        String hashC = "c".repeat(64);
        String hashD = "d".repeat(64);
        String header = "schema_version,run_id,image,input_image_sha256,message_sha256,reference_jpeg_sha256,stego_jpeg_sha256,message,raw_message_bytes,selected_qf,embedding_success,extraction_success,aes_gcm_verified,recovered_message_matches,test_metric\n";
        String rows = "6.0,r1,same-name," + hashA + "," + hashB + "," + hashC + "," + hashD + ",m1,16,75,true,true,true,true,1\n"
                + "6.0,r2,same-name," + hashA + "," + hashB + "," + hashC + "," + hashD + ",m2,32,80,true,true,true,true,3\n"
                + "6.0,r3,same-name," + hashB + "," + hashC + "," + hashD + "," + hashA + ",m1,16,90,true,true,true,true,5\n"
                + "6.0,r4,image-c," + hashC + ",,,,m1,16,,false,false,false,false,100\n";
        Files.writeString(runs, header + rows, StandardCharsets.UTF_8);
        ExperimentSummaryGenerator.generate(
                runs, dir.resolve("positions.csv"), dir.resolve("attempts.csv"), out);

        List<String> summary = Files.readAllLines(out.resolve("run_metric_summary.csv"), StandardCharsets.UTF_8);
        List<String> columns = CsvUtil.parseLine(summary.get(0));
        List<String> metric = summary.stream().skip(1).map(CsvUtil::parseLine)
                .filter(row -> row.get(0).equals("test_metric")).findFirst().orElseThrow();
        assert metric.get(columns.indexOf("row_count")).equals("3");
        assert Math.abs(Double.parseDouble(metric.get(columns.indexOf("mean"))) - 3.0) < 1e-12;
        assert metric.get(columns.indexOf("source_image_clusters")).equals("2");
        assert Math.abs(Double.parseDouble(metric.get(columns.indexOf("source_cluster_mean"))) - 3.5) < 1e-12;

        String json = Files.readString(out.resolve("experiment_summary.json"), StandardCharsets.UTF_8);
        assert json.contains("\"successful_runs\": 3");
        assert json.contains("\"excluded_unverified_run_rows\": 1");

        Path invalidRuns = dir.resolve("invalid_qf.csv");
        Files.writeString(
                invalidRuns,
                header + "6.0,bad,image," + hashA
                        + "," + hashB + "," + hashC + "," + hashD
                        + ",m1,16,not-a-qf,true,true,true,true,1\n",
                StandardCharsets.UTF_8);
        boolean rejected = false;
        try {
            ExperimentSummaryGenerator.generate(
                    invalidRuns, dir.resolve("missing-positions.csv"),
                    dir.resolve("missing-attempts.csv"),
                    dir.resolve("invalid-summary"));
        } catch (java.io.IOException expected) {
            rejected = true;
        }
        assert rejected : "invalid selected_qf must fail closed";
        System.out.println("EvaluationSummaryFilteringSelfTest: ALL TESTS PASSED");
    }

    private static void testActiveBandBounds() throws Exception {
        assert ExperimentSummaryGenerator.parseTargetRank("16") == 16;
        assert ExperimentSummaryGenerator.parseTargetRank("47") == 47;
        for (String outside : List.of("15", "48")) {
            boolean rejected = false;
            try {
                ExperimentSummaryGenerator.parseTargetRank(outside);
            } catch (java.io.IOException expected) {
                rejected = true;
            }
            assert rejected : "summary must reject a rank outside ZZ0(16:47)";
        }
    }
}
