package thesis.rap.app;

import thesis.rap.image.SamplingGeometry;
import thesis.rap.image.SamplingGeometryMode;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/** Regression proof for independent first-verified QF selection per geometry. */
public final class MultiGeometrySelectionSelfTest {
    private MultiGeometrySelectionSelfTest() {}

    public static void main(String[] args) throws Exception {
        Map<SamplingGeometry, Integer> firstPassingQf =
                new EnumMap<>(SamplingGeometry.class);
        firstPassingQf.put(SamplingGeometry.JPEG_444, 75);
        firstPassingQf.put(SamplingGeometry.JPEG_422, 76);
        firstPassingQf.put(SamplingGeometry.JPEG_420, 95);
        // New profiles use deterministic control-flow sentinels only. These
        // values are test fixtures, not experimental candidate-capacity data.
        firstPassingQf.put(SamplingGeometry.JPEG_440, 75);
        firstPassingQf.put(SamplingGeometry.JPEG_411, 75);
        firstPassingQf.put(SamplingGeometry.JPEG_441, 75);
        firstPassingQf.put(SamplingGeometry.JPEG_410, 75);

        Map<SamplingGeometry, Integer> selected =
                new EnumMap<>(SamplingGeometry.class);
        Map<SamplingGeometry, List<Integer>> attempted =
                new EnumMap<>(SamplingGeometry.class);
        for (SamplingGeometry geometry
                : SamplingGeometryMode.ALL_GEOMATRY.geometries()) {
            List<Integer> qfs = new ArrayList<>();
            AdaptiveQfSearch.Result<String> result =
                    AdaptiveQfSearch.firstVerified(75, 100, qf -> {
                        qfs.add(qf);
                        if (qf == firstPassingQf.get(geometry)) {
                            return AdaptiveQfSearch.Trial.verified(
                                    qf, geometry.label());
                        }
                        return AdaptiveQfSearch.Trial.trialRejected(
                                qf, "synthetic full-kernel rejection");
                    });
            selected.put(geometry, result.selectedQf());
            attempted.put(geometry, List.copyOf(qfs));
        }

        assert selected.get(SamplingGeometry.JPEG_444) == 75;
        assert selected.get(SamplingGeometry.JPEG_422) == 76;
        assert selected.get(SamplingGeometry.JPEG_420) == 95;
        assert attempted.get(SamplingGeometry.JPEG_444)
                .equals(List.of(75));
        assert attempted.get(SamplingGeometry.JPEG_422)
                .equals(List.of(75, 76));
        assert attempted.get(SamplingGeometry.JPEG_420).size() == 21;
        assert attempted.get(SamplingGeometry.JPEG_420).get(0) == 75;
        assert attempted.get(SamplingGeometry.JPEG_420).get(20) == 95;
        for (SamplingGeometry geometry : List.of(
                SamplingGeometry.JPEG_440, SamplingGeometry.JPEG_411,
                SamplingGeometry.JPEG_441, SamplingGeometry.JPEG_410)) {
            assert selected.get(geometry) == 75;
            assert attempted.get(geometry).equals(List.of(75));
        }

        // A terminal failure in the middle population must not suppress the
        // later population.  This is the core ALL_GEOMATRY continuation rule.
        Map<SamplingGeometry, Integer> mixedTerminal =
                new EnumMap<>(SamplingGeometry.class);
        Map<SamplingGeometry, Integer> mixedAttemptCount =
                new EnumMap<>(SamplingGeometry.class);
        for (SamplingGeometry geometry
                : SamplingGeometryMode.ALL_GEOMATRY.geometries()) {
            List<Integer> qfs = new ArrayList<>();
            AdaptiveQfSearch.Result<String> result =
                    AdaptiveQfSearch.firstVerified(75, 100, qf -> {
                        qfs.add(qf);
                        boolean passes = switch (geometry) {
                            case JPEG_444 -> qf == 75;
                            case JPEG_422 -> false;
                            case JPEG_420 -> qf == 95;
                            case JPEG_440, JPEG_411, JPEG_441, JPEG_410 -> qf == 75;
                        };
                        return passes
                                ? AdaptiveQfSearch.Trial.verified(
                                        qf, geometry.label())
                                : AdaptiveQfSearch.Trial.trialRejected(
                                        qf, "synthetic full-kernel rejection");
                    });
            mixedTerminal.put(geometry, result.selectedQf());
            mixedAttemptCount.put(geometry, qfs.size());
        }
        assert mixedTerminal.get(SamplingGeometry.JPEG_444) == 75;
        assert mixedTerminal.get(SamplingGeometry.JPEG_422) == null;
        assert mixedTerminal.get(SamplingGeometry.JPEG_420) == 95;
        assert mixedAttemptCount.get(SamplingGeometry.JPEG_444) == 1;
        assert mixedAttemptCount.get(SamplingGeometry.JPEG_422) == 26;
        assert mixedAttemptCount.get(SamplingGeometry.JPEG_420) == 21;
        for (SamplingGeometry geometry : List.of(
                SamplingGeometry.JPEG_440, SamplingGeometry.JPEG_411,
                SamplingGeometry.JPEG_441, SamplingGeometry.JPEG_410)) {
            assert mixedTerminal.get(geometry) == 75;
            assert mixedAttemptCount.get(geometry) == 1;
        }

        Path requested = Path.of("study.jpg");
        assert RapUniward.outputForGeometry(
                requested, SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_444).toString()
                .equals("study_g444.jpg");
        assert RapUniward.outputForGeometry(
                requested, SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_422).toString()
                .equals("study_g422.jpg");
        assert RapUniward.outputForGeometry(
                requested, SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_420).toString()
                .equals("study_g420.jpg");
        assert RapUniward.outputForGeometry(
                requested, SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_440).toString()
                .equals("study_g440.jpg");
        assert RapUniward.outputForGeometry(
                requested, SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_411).toString()
                .equals("study_g411.jpg");
        assert RapUniward.outputForGeometry(
                requested, SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_441).toString()
                .equals("study_g441.jpg");
        assert RapUniward.outputForGeometry(
                requested, SamplingGeometryMode.ALL_GEOMATRY,
                SamplingGeometry.JPEG_410).toString()
                .equals("study_g410.jpg");
        assert RapUniward.outputForGeometry(
                requested, SamplingGeometryMode.GEOMETRY_444,
                SamplingGeometry.JPEG_444).equals(requested);

        boolean rejected = false;
        try {
            SamplingGeometryMode.parse("444INVALID");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected : "Only declared public geometry tokens are valid";

        RapUniward.requireGroupAlignedStartIndex(
                SamplingGeometryMode.ALL_GEOMATRY, 1);
        RapUniward.requireGroupAlignedStartIndex(
                SamplingGeometryMode.ALL_GEOMATRY, 8);
        rejected = false;
        try {
            RapUniward.requireGroupAlignedStartIndex(
                    SamplingGeometryMode.ALL_GEOMATRY, 2);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assert rejected
                : "ALL_GEOMATRY must not resume in the middle of a seven-profile group";
        RapUniward.requireGroupAlignedStartIndex(
                SamplingGeometryMode.GEOMETRY_420, 2);

        System.out.println(
                "MultiGeometrySelectionSelfTest: ALL TESTS PASSED");
    }
}
