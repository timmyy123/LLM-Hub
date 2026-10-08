package com.llmhub.llmhub.ui.components;
public class PocketTtsEngine {
  public interface CancellationProbe { boolean isCancelled(); }
  private native long nativeCreate(String models,String voices);
  private native void nativeEncode(long handle,String reference);
  private native float[] nativeSynthesize(long handle,String text,String reference,CancellationProbe probe);
  private native void nativeClose(long handle);
  public static void main(String[] args) throws Exception {
    Object javaEnvironment=null;
    try {
      System.load(args[0]+"/libonnxruntime4j_jni.so");
      javaEnvironment=Class.forName("ai.onnxruntime.OrtEnvironment").getMethod("getEnvironment").invoke(null);
      System.out.println("Java ONNX environment loaded alongside native Pocket TTS");
    } catch (ClassNotFoundException | UnsatisfiedLinkError optionalRuntimeAbsent) {
      System.out.println("Standalone JNI smoke: Java ONNX environment not included");
    }
    System.load(args[0]+"/libllmhub_pocket_tts.so");
    PocketTtsEngine engine=new PocketTtsEngine();
    long h=engine.nativeCreate(args[0]+"/models",args[0]+"/voices");
    long second=engine.nativeCreate(args[0]+"/models",args[0]+"/voices");
    if(second!=h)throw new AssertionError("sessions were not shared");
    engine.nativeClose(second);
    try {
      long start=System.nanoTime();
      engine.nativeEncode(h,args[0]+"/reference.wav");
      System.out.println("ENCODE seconds="+(System.nanoTime()-start)/1e9);
      float[] audio=engine.nativeSynthesize(h,"This voice was cloned entirely on this Android device.",args[0]+"/reference.wav",()->false);
      double power=0;for(float a:audio){if(!Float.isFinite(a))throw new AssertionError("nonfinite"); power+=a*a;}
      if(audio.length<24000||power<1e-8)throw new AssertionError("empty or silent");
      System.out.println("OUTPUT seconds="+audio.length/24000.0+" rms="+Math.sqrt(power/audio.length));
      float[] aborted=engine.nativeSynthesize(h,"This must be cancelled.",args[0]+"/reference.wav",()->true);
      if(aborted.length!=0)throw new AssertionError("cancellation");
      final int[] polls={0};
      engine.nativeSynthesize(h,"This speech should stop while generation is running.",args[0]+"/reference.wav",()->++polls[0]>=3);
      if(polls[0]!=3)throw new AssertionError("mid-stream cancellation");
      System.out.println("CANCEL passed (before and during synthesis)");
      for (int seconds : new int[]{2,3}) {
        java.nio.ByteBuffer wav=java.nio.ByteBuffer.allocate(44+seconds*24000*2).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int data=wav.capacity()-44;
        wav.put("RIFF".getBytes()).putInt(data+36).put("WAVEfmt ".getBytes()).putInt(16);
        wav.putShort((short)1).putShort((short)1).putInt(24000).putInt(48000).putShort((short)2).putShort((short)16);
        wav.put("data".getBytes()).putInt(data);
        if(seconds==2)while(wav.hasRemaining())wav.putShort((short)1000);
        String path=args[0]+"/invalid-"+seconds+".wav";
        java.nio.file.Files.write(java.nio.file.Paths.get(path),wav.array());
        try{engine.nativeEncode(h,path);throw new AssertionError("short/silent reference accepted");}
        catch(IllegalStateException expected){System.out.println("SHORT/SILENT reference rejected");}
      }
      try {engine.nativeEncode(h,args[0]+"/missing.wav");throw new AssertionError("missing audio accepted");}
      catch(IllegalStateException expected){System.out.println("BAD REFERENCE rejected");}
    } finally {engine.nativeClose(h);}
    if(javaEnvironment!=null)javaEnvironment.getClass().getMethod("close").invoke(javaEnvironment);
    System.out.println("POCKET ANDROID JNI SMOKE PASSED");
  }
}
