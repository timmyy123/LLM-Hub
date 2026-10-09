import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import java.io.*;
import java.lang.reflect.*;
import java.nio.*;

/** Run with the built app APK on the app_process classpath; never installs/replaces the app. */
public final class PocketVoiceImportSmoke {
    static Object helper;
    static Method prepare;
    static Object track, check;
    static File root;
    static File converted(File source) throws Exception {
        try { return (File)prepare.invoke(helper, source, root, track, check); }
        catch (InvocationTargetException error) { throw (Exception)error.getCause(); }
    }
    public static void main(String[] args) throws Exception {
        root = new File(args[0]);
        Class<?> cls=Class.forName("com.llmhub.llmhub.ui.components.PocketVoiceImport");
        helper=cls.getField("INSTANCE").get(null);
        for(Method method:cls.getDeclaredMethods()) if(method.getName().startsWith("prepare") && method.getParameterCount()==4) prepare=method;
        Object unit=Class.forName("kotlin.Unit").getField("INSTANCE").get(null);
        Class<?> f1=Class.forName("kotlin.jvm.functions.Function1"), f0=Class.forName("kotlin.jvm.functions.Function0");
        track=Proxy.newProxyInstance(f1.getClassLoader(),new Class[]{f1},(p,m,a)->unit);
        check=Proxy.newProxyInstance(f0.getClassLoader(),new Class[]{f0},(p,m,a)->unit);
        File m4a=new File(root,"phone-recording.m4a"); encode(m4a,4);
        File renamed=new File(root,"reference-with-no-extension");
        if(!m4a.renameTo(renamed))throw new AssertionError("rename failed");
        File wav=converted(renamed);
        inspect(wav,3,5);
        System.out.println("M4A AAC stereo with no extension -> mono WAV PASSED");
        File wrongName=new File(root,"misnamed.mp3"); copy(renamed,wrongName);
        inspect(converted(wrongName),3,5);
        System.out.println("M4A content named .mp3 -> mono WAV PASSED");
        File nativeInput=new File(root,"wav-with-no-extension"); copy(wav,nativeInput);
        File copied=converted(nativeInput);
        if(copied.length()!=wav.length())throw new AssertionError("WAV changed");
        System.out.println("WAV without extension PASSED");
        File longAudio=new File(root,"long.m4a"); encode(longAudio,90);
        File longWav=converted(longAudio);
        inspect(longWav,30,30);
        System.out.println("90-second reference capped to 30 seconds PASSED");
        File shortAudio=new File(root,"short.m4a"); encode(shortAudio,1);
        int before=root.list().length;
        try{converted(shortAudio);throw new AssertionError("Short recording accepted");}catch(IllegalArgumentException expected){}
        if(root.list().length!=before)throw new AssertionError("Failed import left output");
        System.out.println("Short reference rejected and cleaned PASSED");
        if(new File(root,"models/tokenizer.model").isFile()) {
            Class<?> engineClass=Class.forName("com.llmhub.llmhub.ui.components.PocketTtsEngine");
            File voices=new File(root,"voices");voices.mkdirs();
            Object engine=engineClass.getConstructor(File.class,File.class).newInstance(new File(root,"models"),voices);
            try {
                File speech=new File(root,"speech.m4a");
                File reference=speech.isFile()?converted(speech):wav;
                engineClass.getMethod("encode",File.class).invoke(engine,reference);
                engineClass.getMethod("encode",File.class).invoke(engine,longWav);
                String stem=longWav.getName().substring(0,longWav.getName().lastIndexOf('.'));
                byte[] embedding=java.nio.file.Files.readAllBytes(new File(voices,".cache/"+stem+".emb").toPath());
                ByteBuffer tensor=ByteBuffer.wrap(embedding).order(ByteOrder.LITTLE_ENDIAN);
                long voiceFrames=tensor.getLong(16);
                if(voiceFrames!=375)throw new AssertionError("Incorrect native capped reference: "+voiceFrames);
                System.out.println("30-second capped reference -> native encoder PASSED ("+voiceFrames+" frames)");
                Object cancelled=Proxy.newProxyInstance(f0.getClassLoader(),new Class[]{f0},(p,m,a)->false);
                float[] audio=(float[])engineClass.getMethod("synthesize",String.class,File.class,f0).invoke(engine,"Hello.",longWav,cancelled);
                if(audio.length==0)throw new AssertionError("Long-reference synthesis empty");
                double energy=0;
                for(float sample:audio){if(!Float.isFinite(sample))throw new AssertionError("Invalid synthesis");energy+=sample*sample;}
                if(energy/audio.length<1e-8)throw new AssertionError("Silent synthesis");
                System.out.println("Speech synthesis using capped reference PASSED");
                System.out.println("Decoded M4A -> actual Pocket voice encoder PASSED");
            } finally {engineClass.getMethod("close").invoke(engine);}
        }
        try(BufferedReader status=new BufferedReader(new FileReader("/proc/self/status"))) {
            for(String line;(line=status.readLine())!=null;)if(line.startsWith("VmHWM:")||line.startsWith("VmRSS:"))System.out.println(line);
        }
        System.out.println("POCKET AUDIO IMPORT SMOKE PASSED");
    }
    static void copy(File source,File target)throws IOException {
        try(InputStream in=new FileInputStream(source);OutputStream out=new FileOutputStream(target)){
            byte[] b=new byte[8192];for(int n;(n=in.read(b))>=0;)out.write(b,0,n);
        }
    }
    static void inspect(File wav,int minimum,int maximum)throws IOException {
        byte[] bytes=java.nio.file.Files.readAllBytes(wav.toPath());
        ByteBuffer b=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        boolean rf64=new String(bytes,0,4).equals("RF64");
        if(!rf64&&!new String(bytes,0,4).equals("RIFF"))throw new AssertionError("Invalid WAV");
        int fmt=rf64?48:12, start=rf64?80:44;
        if(b.getShort(fmt+10)!=1||b.getShort(fmt+22)!=16)throw new AssertionError("Invalid mono PCM16 WAV");
        int rate=b.getInt(fmt+12),frames=(bytes.length-start)/2;
        if(rf64 && b.getLong(28)!=bytes.length-start)throw new AssertionError("RF64 size mismatch");
        double seconds=frames/(double)rate;
        if(seconds<minimum||seconds>maximum)throw new AssertionError("Duration "+seconds);
        double energy=0;for(int i=start;i<bytes.length;i+=2){double v=b.getShort(i)/32768.0;energy+=v*v;}
        if(energy/frames<1e-5)throw new AssertionError("Decoded audio silent");
    }
    static void encode(File target,int seconds)throws Exception {
        MediaCodec codec=MediaCodec.createEncoderByType("audio/mp4a-latm");
        MediaMuxer muxer=new MediaMuxer(target.getPath(),MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
        boolean muxing=false,started=false;
        try{
            MediaFormat f=MediaFormat.createAudioFormat("audio/mp4a-latm",44100,2);
            f.setInteger(MediaFormat.KEY_AAC_PROFILE,MediaCodecInfo.CodecProfileLevel.AACObjectLC);
            f.setInteger(MediaFormat.KEY_BIT_RATE,96000);
            codec.configure(f,null,null,MediaCodec.CONFIGURE_FLAG_ENCODE);codec.start();started=true;
            MediaCodec.BufferInfo info=new MediaCodec.BufferInfo();int frame=0,total=seconds*44100,track=-1;
            boolean inputDone=false,outputDone=false;
            long deadline=android.os.SystemClock.elapsedRealtime()+60000;
            while(!outputDone){
                if(android.os.SystemClock.elapsedRealtime()>deadline)throw new AssertionError("Encoder timed out");
                if(!inputDone){int i=codec.dequeueInputBuffer(10000);if(i>=0){
                    ByteBuffer data=codec.getInputBuffer(i);data.clear();data.order(ByteOrder.LITTLE_ENDIAN);
                    int count=Math.min(data.remaining()/4,total-frame);
                    long pts=frame*1000000L/44100;
                    for(int n=0;n<count;n++,frame++){short v=(short)(Math.sin(frame*2*Math.PI*440/44100)*8000);data.putShort(v);data.putShort((short)(v/2));}
                    inputDone=count==0;
                    codec.queueInputBuffer(i,0,count*4,pts,inputDone?MediaCodec.BUFFER_FLAG_END_OF_STREAM:0);
                }}
                int i=codec.dequeueOutputBuffer(info,10000);
                if(i==MediaCodec.INFO_OUTPUT_FORMAT_CHANGED){track=muxer.addTrack(codec.getOutputFormat());muxer.start();muxing=true;}
                else if(i>=0){
                    if(info.size>0&&(info.flags&MediaCodec.BUFFER_FLAG_CODEC_CONFIG)==0){ByteBuffer data=codec.getOutputBuffer(i);data.position(info.offset);data.limit(info.offset+info.size);muxer.writeSampleData(track,data,info);}
                    outputDone=(info.flags&MediaCodec.BUFFER_FLAG_END_OF_STREAM)!=0;codec.releaseOutputBuffer(i,false);
                }
            }
        }finally{try{if(started)codec.stop();}finally{codec.release();try{if(muxing)muxer.stop();}finally{muxer.release();}}}
    }
}
