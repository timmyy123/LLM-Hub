import java.io.File;
import java.lang.reflect.*;

/** Exercises the actual R8-processed Kotlin callback and JNI from the release APK. */
public final class PocketReleaseCallbackSmoke {
    public static void main(String[] args) throws Exception {
        Class<?> cls = Class.forName("com.llmhub.llmhub.ui.components.PocketTtsEngine");
        Method synthesize = null;
        for (Method method : cls.getDeclaredMethods()) {
            if (method.getName().equals("synthesize")) synthesize = method;
        }
        if (synthesize == null) throw new AssertionError("Missing synthesis entry point");
        Class<?> callback = synthesize.getParameterTypes()[2];

        // Avoid model downloads: all cases exit before the native engine is accessed.
        Class<?> unsafeClass = Class.forName("sun.misc.Unsafe");
        Field singleton = unsafeClass.getDeclaredField("theUnsafe");
        singleton.setAccessible(true);
        Object unsafe = singleton.get(null);
        Object engine = unsafeClass.getMethod("allocateInstance", Class.class).invoke(unsafe, cls);
        Field handle = cls.getDeclaredField("handle");
        handle.setAccessible(true);
        handle.setLong(engine, 1L);
        File missing = new File(args[0], "nonexistent-reference.wav");
        if (missing.exists()) throw new AssertionError("Reference must not exist");
        try {
            int[] calls = {0};
            Object cancelled = Proxy.newProxyInstance(callback.getClassLoader(), new Class[]{callback},
                    (p, m, a) -> { calls[0]++; return true; });
            float[] samples = (float[]) synthesize.invoke(engine, "Hello.", missing, cancelled);
            if (samples.length != 0 || calls[0] != 1) throw new AssertionError("Cancellation failed");
            System.out.println("Release Kotlin callback -> JNI pre-cancellation PASSED");

            Object active = Proxy.newProxyInstance(callback.getClassLoader(), new Class[]{callback},
                    (p, m, a) -> false);
            expect(synthesize, engine, missing, active, IllegalStateException.class,
                    "Reference audio was deleted");
            System.out.println("Release Kotlin callback -> JNI missing-reference exception PASSED");

            Object throwing = Proxy.newProxyInstance(callback.getClassLoader(), new Class[]{callback},
                    (p, m, a) -> { throw new IllegalStateException("callback test failure"); });
            expect(synthesize, engine, missing, throwing, IllegalStateException.class,
                    "callback test failure");
            System.out.println("JNI propagates callback exception without aborting PASSED");
            System.out.println("POCKET RELEASE CALLBACK SMOKE PASSED");
        } finally {
            // This is a sentinel, not a native allocation; never pass it to nativeClose.
            handle.setLong(engine, 0L);
        }
    }

    static void expect(Method method, Object engine, File reference, Object callback,
                       Class<?> type, String message) throws Exception {
        try {
            method.invoke(engine, "Hello.", reference, callback);
            throw new AssertionError("Expected " + type.getName());
        } catch (InvocationTargetException error) {
            Throwable cause = error.getCause();
            if (!type.isInstance(cause) || !message.equals(cause.getMessage())) {
                throw new AssertionError("Wrong exception", cause);
            }
        }
    }
}
