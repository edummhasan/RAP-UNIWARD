package thesis.rap.analysis;

import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Stable message-derived grouping metadata for corpus evaluation ledgers. */
public record MessageGroup(int number, String label, String name) {
    private static final Pattern FIRST_NUMBER = Pattern.compile("(\\d+)");

    public static MessageGroup from(String messageName) {
        String name = Path.of(messageName).getFileName().toString();
        Matcher matcher = FIRST_NUMBER.matcher(name);
        int number = matcher.find() ? Integer.parseInt(matcher.group(1)) : 0;
        String label = number > 0 ? String.format("msg_%03d", number) : "msg_unnumbered";
        return new MessageGroup(number, label, name);
    }
}
