package thesis.rap.jni;

public final class JpegCore {
    static { System.loadLibrary("jpegcore"); }
    private JpegCore() {}
    public static native JpegImage readCoefficients(String jpegPath);
    public static native void writeCoefficients(String templateJpegPath, String outputJpegPath, JpegImage image);
    public static native int libjpegVersion();
}
