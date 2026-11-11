package thesis.rap.payload;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.PriorityQueue;

public final class HuffmanCodec {
    private HuffmanCodec() {}

    private static final class Node implements Comparable<Node> {
        final int symbol;
        final int freq;
        final int minSymbol;
        final Node left;
        final Node right;

        Node(int symbol, int freq, Node left, Node right) {
            this.symbol = symbol;
            this.freq = freq;
            this.left = left;
            this.right = right;
            this.minSymbol = symbol >= 0 ? symbol : Math.min(left.minSymbol, right.minSymbol);
        }

        boolean leaf() {
            return left == null && right == null;
        }

        @Override
        public int compareTo(Node o) {
            int byFrequency = Integer.compare(this.freq, o.freq);
            return byFrequency != 0 ? byFrequency : Integer.compare(this.minSymbol, o.minSymbol);
        }
    }

    public static byte[] encode(byte[] input) {
        int[] freq = new int[256];

        for (byte b : input) {
            freq[b & 0xFF] = Math.incrementExact(
                    freq[b & 0xFF]);
        }

        Node root = buildTree(freq);
        String[] codes = new String[256];
        buildCodes(root, "", codes);

        ByteArrayOutputStream bitOut = new ByteArrayOutputStream();
        int current = 0;
        int bitCount = 0;
        int totalBits = 0;

        for (byte b : input) {
            String code = codes[b & 0xFF];

            for (int i = 0; i < code.length(); i++) {
                current <<= 1;

                if (code.charAt(i) == '1') {
                    current |= 1;
                }

                bitCount++;
                totalBits = Math.incrementExact(totalBits);

                if (bitCount == 8) {
                    bitOut.write(current);
                    current = 0;
                    bitCount = 0;
                }
            }
        }

        if (bitCount > 0) {
            current <<= (8 - bitCount);
            bitOut.write(current);
        }

        byte[] encodedBits = bitOut.toByteArray();

        ByteBuffer header = ByteBuffer.allocate(4 + 256 * 4 + 4)
                .order(ByteOrder.BIG_ENDIAN);

        header.putInt(input.length);

        for (int f : freq) {
            header.putInt(f);
        }

        header.putInt(totalBits);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes(header.array());
        out.writeBytes(encodedBits);

        return out.toByteArray();
    }

    public static byte[] decode(byte[] packed) {
        if (packed == null || packed.length < 4) {
            throw new IllegalStateException("Huffman payload is truncated");
        }
        ByteBuffer bb = ByteBuffer.wrap(packed).order(ByteOrder.BIG_ENDIAN);

        int originalLength = bb.getInt();

        if (originalLength < 0) throw new IllegalStateException("Negative Huffman original length");

        if (bb.remaining() < 256 * Integer.BYTES + Integer.BYTES) {
            throw new IllegalStateException("Huffman frequency header is truncated");
        }

        int[] freq = new int[256];
        long frequencyTotal = 0;

        for (int i = 0; i < 256; i++) {
            freq[i] = bb.getInt();
            if (freq[i] < 0) throw new IllegalStateException("Negative Huffman symbol frequency");
            frequencyTotal += freq[i];
        }
        if (frequencyTotal != originalLength) {
            throw new IllegalStateException("Huffman frequencies do not match declared message length");
        }

        int totalBits = bb.getInt();
        if (totalBits < 0 || totalBits > Math.multiplyExact(bb.remaining(), 8)) {
            throw new IllegalStateException("Invalid Huffman bit length");
        }
        int requiredBitBytes = Math.addExact(totalBits, 7) / 8;
        if (bb.remaining() != requiredBitBytes) {
            throw new IllegalStateException("Huffman payload contains truncated or trailing coded bytes");
        }

        byte[] bitData = new byte[bb.remaining()];
        bb.get(bitData);
        if (totalBits % 8 != 0 && bitData.length > 0) {
            int unusedBits = 8 - (totalBits % 8);
            int paddingMask = (1 << unusedBits) - 1;
            if ((bitData[bitData.length - 1] & paddingMask) != 0) {
                throw new IllegalStateException("Huffman payload has nonzero padding bits");
            }
        }

        Node root = buildTree(freq);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        int[] observedFrequency = new int[256];
        Node node = root;

        for (int i = 0; i < totalBits; i++) {
            int byteIndex = i / 8;
            int bitIndex = 7 - (i % 8);
            int bit = (bitData[byteIndex] >> bitIndex) & 1;

            node = bit == 0 ? node.left : node.right;
            if (node == null) throw new IllegalStateException("Invalid Huffman code path");

            if (node.leaf()) {
                if (freq[node.symbol] == 0) {
                    throw new IllegalStateException(
                            "Invalid Huffman code path");
                }
                if (++observedFrequency[node.symbol] > freq[node.symbol]) {
                    throw new IllegalStateException(
                            "Huffman decoded symbol frequencies exceed the declared table");
                }
                out.write(node.symbol);
                node = root;
                if (out.size() > originalLength) {
                    throw new IllegalStateException("Huffman payload decodes beyond the declared message length");
                }
            }
        }

        if (out.size() != originalLength || node != root) {
            throw new IllegalStateException("Huffman payload ended before the declared message length");
        }
        if (!Arrays.equals(observedFrequency, freq)) {
            throw new IllegalStateException(
                    "Huffman decoded symbol frequencies do not match the declared table");
        }
        return out.toByteArray();
    }

    private static Node buildTree(int[] freq) {
        PriorityQueue<Node> pq = new PriorityQueue<>();

        for (int i = 0; i < 256; i++) {
            if (freq[i] > 0) {
                pq.add(new Node(i, freq[i], null, null));
            }
        }

        if (pq.isEmpty()) {
            pq.add(new Node(0, 1, null, null));
        }

        if (pq.size() == 1) {
            Node only = pq.poll();
            int invalidSymbol = (only.symbol + 1) & 0xff;
            return new Node(
                    -1, only.freq, only,
                    new Node(invalidSymbol, 0, null, null));
        }

        while (pq.size() > 1) {
            Node a = pq.poll();
            Node b = pq.poll();
            pq.add(new Node(
                    -1, Math.addExact(a.freq, b.freq), a, b));
        }

        return pq.poll();
    }

    private static void buildCodes(Node node, String prefix, String[] codes) {
        if (node.leaf()) {
            codes[node.symbol] = prefix.isEmpty() ? "0" : prefix;
            return;
        }

        buildCodes(node.left, prefix + "0", codes);
        buildCodes(node.right, prefix + "1", codes);
    }
}
