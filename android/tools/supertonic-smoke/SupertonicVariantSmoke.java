import android.content.Context;
import android.content.ContextWrapper;
import java.io.*;
import java.lang.reflect.*;
import java.nio.file.Files;
import java.util.Map;

/** Tests the app's Supertonic engine from a debug APK without installing it. */
public final class SupertonicVariantSmoke {
    public static void main(String[] args) throws Exception {
        File root = new File(args[0]);
        Class<?> modelClass = Class.forName("com.llmhub.llmhub.data.SupertonicModel");
        Object model = modelClass.getField("INSTANCE").get(null);
        String full = (String) modelClass.getField("NAME").get(null);
        String fp16 = (String) modelClass.getField("FP16_NAME").get(null);
        Context context = new ContextWrapper(null) {
            @Override public File getFilesDir() { return new File(root, "test-files"); }
        };
        Method directory = modelClass.getMethod("directory", Context.class, String.class);
        Method complete = modelClass.getMethod("isComplete", Context.class, String.class);
        for (String variant : new String[]{full, fp16}) {
            File dir = (File) directory.invoke(model, context, variant);
            dir.mkdirs();
            @SuppressWarnings("unchecked")
            Map<String, Long> sizes = (Map<String, Long>) modelClass
                    .getMethod("coreFilesFor", String.class).invoke(model, variant);
            for (Map.Entry<String, Long> entry : sizes.entrySet()) {
                try (RandomAccessFile file = new RandomAccessFile(new File(dir, entry.getKey()), "rw")) {
                    file.setLength(entry.getValue());
                }
            }
            if (!(Boolean) complete.invoke(model, context, variant)) throw new AssertionError("Complete bundle rejected");
            File weight = new File(dir, "vocoder.onnx");
            try (RandomAccessFile file = new RandomAccessFile(weight, "rw")) { file.setLength(weight.length() - 1); }
            if ((Boolean) complete.invoke(model, context, variant)) throw new AssertionError("Truncated bundle accepted");
            weight.delete();
            if ((Boolean) complete.invoke(model, context, variant)) throw new AssertionError("Missing weight accepted");
            for (File file : dir.listFiles()) file.delete();
        }
        if (directory.invoke(model, context, full).equals(directory.invoke(model, context, fp16))) {
            throw new AssertionError("Variants share weights directory");
        }
        System.out.println("FP32/FP16 independent bundle completeness checks PASSED");

        Class<?> engineClass = Class.forName("com.llmhub.llmhub.ui.components.SupertonicEngine");
        Object engine = engineClass.getConstructor(File.class, boolean.class).newInstance(root, true);
        Class<?> voiceClass = Class.forName("com.llmhub.llmhub.data.SupertonicModel$Voice");
        Object voice = modelClass.getMethod("parseVoice", String.class).invoke(model,
                new String(Files.readAllBytes(new File(root, "M1.json").toPath()), "UTF-8"));
        Class<?> f0 = Class.forName("kotlin.jvm.functions.Function0");
        Method synthesize = engineClass.getMethod("synthesize", String.class, String.class, voiceClass, float.class, f0);
        Object active = Proxy.newProxyInstance(f0.getClassLoader(), new Class[]{f0}, (p, m, a) -> false);
        try {
            int rate = (Integer) engineClass.getMethod("getSampleRate").invoke(engine);
            if (rate != 44100) throw new AssertionError("Incorrect sample rate: " + rate);
            String[][] phrases = {{"en", "Hello, this is a voice test."}, {"ja", "こんにちは。"}, {"fr", "Bonjour, ceci est un test."}};
            for (String[] phrase : phrases) {
                float[] samples = (float[]) synthesize.invoke(engine, phrase[1], phrase[0], voice, 1f, active);
                if (samples.length == 0) throw new AssertionError("Empty audio");
                double energy = 0;
                for (float sample : samples) {
                    if (!Float.isFinite(sample)) throw new AssertionError("Nonfinite audio");
                    energy += sample * sample;
                }
                if (energy / samples.length < 1e-8) throw new AssertionError("Silent audio");
                System.out.println(phrase[0] + " synthesis PASSED: " + samples.length + " samples at " + rate + " Hz");
            }
            Object cancelled = Proxy.newProxyInstance(f0.getClassLoader(), new Class[]{f0}, (p, m, a) -> true);
            try {
                synthesize.invoke(engine, "Hello.", "en", voice, 1f, cancelled);
                throw new AssertionError("Cancellation ignored");
            } catch (InvocationTargetException error) {
                if (!(error.getCause() instanceof java.util.concurrent.CancellationException)) throw error;
            }
            System.out.println("Cancellation PASSED");
        } finally { engineClass.getMethod("close").invoke(engine); }
        System.out.println("SUPERTONIC VARIANT SMOKE PASSED");
    }
}
