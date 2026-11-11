package thesis.rap.app;


public final class BatchShardSelectionSelfTest {
    private BatchShardSelectionSelfTest() {}
    public static void main(String[] args) {
        int shards = 4;
        for (long pair = 1; pair <= 100; pair++) {
            int owners = 0;
            for (int r = 0; r < shards; r++) {
                if (RapUniward.pairBelongsToShard(pair, shards, r)) owners++;
            }
            assert owners == 1 : "pair " + pair + " owners=" + owners;
        }
        assert RapUniward.pairBelongsToShard(1, 2, 0);
        assert RapUniward.pairBelongsToShard(2, 2, 1);
        assert RapUniward.pairBelongsToShard(3, 2, 0);
        System.out.println("BatchShardSelectionSelfTest: ALL TESTS PASSED");
    }
}
