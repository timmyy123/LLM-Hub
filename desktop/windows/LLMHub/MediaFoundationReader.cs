using System.Runtime.InteropServices;

namespace LLMHub;

/// <summary>
/// Media Foundation's Source Reader (mfreadwrite.dll), called through its COM vtables: a video's frames as BGRA in
/// order (decoded and converted to RGB32 by the reader), or its sound as float PCM, from any file Windows can play.
/// Used by Studio › Search › Videos, where Windows.Media.Editing thumbnails are half a second each.
/// </summary>
internal sealed unsafe class MediaFoundationReader : IDisposable
{
    private const uint AllStreams = 0xFFFFFFFE;
    private const uint FirstVideoStream = 0xFFFFFFFC;
    private const uint FirstAudioStream = 0xFFFFFFFD;
    private const uint MediaSource = 0xFFFFFFFF;
    private const uint EndOfStream = 0x2;
    private const uint MediaTypeChanged = 0x20;

    private static readonly Guid EnableVideoProcessing = new("fb394f3d-ccf1-42ee-bbb3-f9b845d5681d");
    private static readonly Guid MajorType = new("48eba18e-f8c9-4687-bf11-0a74c9f96a8f");
    private static readonly Guid Subtype = new("f7e34c9a-42e8-4714-b74b-cb29d72c35e5");
    private static readonly Guid MediaTypeVideo = new("73646976-0000-0010-8000-00AA00389B71");
    private static readonly Guid MediaTypeAudio = new("73647561-0000-0010-8000-00AA00389B71");
    private static readonly Guid FormatRgb32 = new("00000016-0000-0010-8000-00AA00389B71");
    private static readonly Guid FormatFloat = new("00000003-0000-0010-8000-00AA00389B71");
    private static readonly Guid FrameSize = new("1652c33d-d6b2-4012-b834-72030849a37d");
    private static readonly Guid DefaultStride = new("644b4e48-1e02-4516-b0eb-c01ca9d49ac6");
    private static readonly Guid DisplayAperture = new("d7388766-18fe-48c6-a177-ee894867c8c4");
    private static readonly Guid AudioChannels = new("37e48bf5-645e-4c5b-89de-ada9e29b696a");
    private static readonly Guid AudioRate = new("5faeeae7-0290-4c31-9e8a-c534f68d9dba");
    private static readonly Guid Duration = new("6c990d33-bb8e-477a-8598-0d5d96fcd88a");
    private static readonly Guid Buffer2D = new("7DC9D5F9-9ED9-44ec-9BBF-0600BB589FBB");
    private static readonly Lazy<bool> Started = new(() => MFStartup(0x00020070, 0) >= 0);

    private IntPtr _reader;
    private readonly uint _stream;

    private MediaFoundationReader(IntPtr reader, uint stream)
    {
        _reader = reader;
        _stream = stream;
    }

    /// <summary>Visible frame size after the reader's conversion to RGB32.</summary>
    public int Width { get; private set; }
    public int Height { get; private set; }
    public int Channels { get; private set; }
    public int SampleRate { get; private set; }
    private int _stride;
    private int _frameHeight;
    private int _left;
    private int _top;

    /// <summary>The file's video (or audio) stream, ready to read; null when Windows cannot decode it.</summary>
    public static MediaFoundationReader? Open(string path, bool video)
    {
        if (!Started.Value)
            return null;
        if (MFCreateAttributes(out var attributes, 1) < 0)
            return null;
        IntPtr reader;
        try
        {
            var key = EnableVideoProcessing;
            Call(attributes, 21, &key, 1u);
            if (MFCreateSourceReaderFromURL(path, attributes, out reader) < 0)
                return null;
        }
        finally
        {
            Release(attributes);
        }
        var result = new MediaFoundationReader(reader, video ? FirstVideoStream : FirstAudioStream);
        if (result.Configure(video))
            return result;
        result.Dispose();
        return null;
    }

    private bool Configure(bool video)
    {
        // IMFSourceReader::SetStreamSelection: only the one stream is decoded.
        ((delegate* unmanaged[Stdcall]<IntPtr, uint, int, int>)Slot(_reader, 4))(_reader, AllStreams, 0);
        if (((delegate* unmanaged[Stdcall]<IntPtr, uint, int, int>)Slot(_reader, 4))(_reader, _stream, 1) < 0)
            return false;
        if (MFCreateMediaType(out var type) < 0)
            return false;
        try
        {
            Guid major = MajorType, majorValue = video ? MediaTypeVideo : MediaTypeAudio, sub = Subtype, subValue = video ? FormatRgb32 : FormatFloat;
            ((delegate* unmanaged[Stdcall]<IntPtr, Guid*, Guid*, int>)Slot(type, 24))(type, &major, &majorValue);
            ((delegate* unmanaged[Stdcall]<IntPtr, Guid*, Guid*, int>)Slot(type, 24))(type, &sub, &subValue);
            // IMFSourceReader::SetCurrentMediaType
            if (((delegate* unmanaged[Stdcall]<IntPtr, uint, IntPtr, IntPtr, int>)Slot(_reader, 7))(_reader, _stream, IntPtr.Zero, type) < 0)
                return false;
        }
        finally
        {
            Release(type);
        }
        return ReadFormat(video);
    }

    /// <summary>Reads the current output type: frame size, stride and visible area, or channels and rate.</summary>
    private bool ReadFormat(bool video)
    {
        IntPtr current = IntPtr.Zero;
        if (((delegate* unmanaged[Stdcall]<IntPtr, uint, IntPtr*, int>)Slot(_reader, 6))(_reader, _stream, &current) < 0)
            return false;
        try
        {
            if (!video)
            {
                Channels = (int)GetUInt32(current, AudioChannels, 0);
                SampleRate = (int)GetUInt32(current, AudioRate, 0);
                return Channels > 0 && SampleRate > 0;
            }
            var size = GetUInt64(current, FrameSize);
            int width = (int)(size >> 32), height = (int)(size & 0xFFFFFFFF);
            _stride = (int)GetUInt32(current, DefaultStride, (uint)(width * 4));
            _frameHeight = height;
            (_left, _top, Width, Height) = (0, 0, width, height);
            // MFVideoArea: OffsetX and OffsetY (fract, value), then cx and cy; H.264 often decodes 1088 lines for 1080.
            var area = stackalloc byte[16];
            var key = DisplayAperture;
            uint length = 0;
            if (((delegate* unmanaged[Stdcall]<IntPtr, Guid*, byte*, uint, uint*, int>)Slot(current, 15))(current, &key, area, 16, &length) >= 0 && length == 16)
            {
                int left = *(short*)(area + 2), top = *(short*)(area + 6), cx = *(int*)(area + 8), cy = *(int*)(area + 12);
                if (cx > 0 && cy > 0 && left + cx <= width && top + cy <= height)
                    (_left, _top, Width, Height) = (left, top, cx, cy);
            }
            return width > 0 && height > 0;
        }
        finally
        {
            Release(current);
        }
    }

    /// <summary>The length from the container, in milliseconds (0 when unknown).</summary>
    public int DurationMs
    {
        get
        {
            var value = stackalloc byte[24];
            var key = Duration;
            if (((delegate* unmanaged[Stdcall]<IntPtr, uint, Guid*, byte*, int>)Slot(_reader, 12))(_reader, MediaSource, &key, value) < 0)
                return 0;
            return *(ushort*)value == 21 ? (int)(*(ulong*)(value + 8) / 10_000) : 0; // VT_UI8, 100 ns units
        }
    }

    /// <summary>Jumps to the key frame at or before the time; the next reads continue from there.</summary>
    public void Seek(int timeMs)
    {
        var position = stackalloc byte[24];
        *(ushort*)position = 20; // VT_I8
        *(long*)(position + 8) = timeMs * 10_000L;
        var format = Guid.Empty;
        ((delegate* unmanaged[Stdcall]<IntPtr, Guid*, byte*, int>)Slot(_reader, 8))(_reader, &format, position);
    }

    /// <summary>The next decoded sample (AddRef'd; pass to <see cref="Release"/>) and its time, or null at the end.</summary>
    public (IntPtr Sample, long Time100ns)? Read()
    {
        while (true)
        {
            uint actual = 0, flags = 0;
            long time = 0;
            var sample = IntPtr.Zero;
            var hr = ((delegate* unmanaged[Stdcall]<IntPtr, uint, uint, uint*, uint*, long*, IntPtr*, int>)Slot(_reader, 9))(
                _reader, _stream, 0, &actual, &flags, &time, &sample);
            if (hr < 0 || (flags & EndOfStream) != 0)
            {
                Release(sample);
                return null;
            }
            if ((flags & MediaTypeChanged) != 0)
                ReadFormat(_stream == FirstVideoStream);
            if (sample != IntPtr.Zero)
                return (sample, time);
        }
    }

    /// <summary>A video sample's visible pixels as top-down BGRA (alpha unset).</summary>
    public byte[]? Pixels(IntPtr sample)
    {
        var buffer = IntPtr.Zero;
        if (((delegate* unmanaged[Stdcall]<IntPtr, IntPtr*, int>)Slot(sample, 41))(sample, &buffer) < 0)
            return null;
        try
        {
            var pixels = new byte[Width * Height * 4];
            // IMF2DBuffer::Lock2D gives the real first line and pitch (negative when bottom-up); else Lock and the type's stride.
            var iid = Buffer2D;
            var buffer2D = IntPtr.Zero;
            if (((delegate* unmanaged[Stdcall]<IntPtr, Guid*, IntPtr*, int>)Slot(buffer, 0))(buffer, &iid, &buffer2D) >= 0)
            {
                try
                {
                    byte* line = null;
                    var pitch = 0;
                    if (((delegate* unmanaged[Stdcall]<IntPtr, byte**, int*, int>)Slot(buffer2D, 3))(buffer2D, &line, &pitch) < 0)
                        return null;
                    Copy(line, pitch, pixels);
                    ((delegate* unmanaged[Stdcall]<IntPtr, int>)Slot(buffer2D, 4))(buffer2D);
                    return pixels;
                }
                finally
                {
                    Release(buffer2D);
                }
            }
            byte* data = null;
            uint max = 0, length = 0;
            if (((delegate* unmanaged[Stdcall]<IntPtr, byte**, uint*, uint*, int>)Slot(buffer, 3))(buffer, &data, &max, &length) < 0)
                return null;
            var first = _stride < 0 ? data + (long)(-_stride) * (_frameHeight - 1) : data;
            Copy(first, _stride, pixels);
            ((delegate* unmanaged[Stdcall]<IntPtr, int>)Slot(buffer, 4))(buffer);
            return pixels;
        }
        finally
        {
            Release(buffer);
        }
    }

    private void Copy(byte* firstLine, int pitch, byte[] pixels)
    {
        var row = Width * 4;
        fixed (byte* target = pixels)
        {
            for (var y = 0; y < Height; y++)
                System.Buffer.MemoryCopy(firstLine + (long)(y + _top) * pitch + _left * 4, target + (long)y * row, row, row);
        }
    }

    /// <summary>An audio sample's interleaved float PCM.</summary>
    public float[]? Floats(IntPtr sample)
    {
        var buffer = IntPtr.Zero;
        if (((delegate* unmanaged[Stdcall]<IntPtr, IntPtr*, int>)Slot(sample, 41))(sample, &buffer) < 0)
            return null;
        try
        {
            byte* data = null;
            uint max = 0, length = 0;
            if (((delegate* unmanaged[Stdcall]<IntPtr, byte**, uint*, uint*, int>)Slot(buffer, 3))(buffer, &data, &max, &length) < 0)
                return null;
            var floats = new ReadOnlySpan<float>(data, (int)(length / 4)).ToArray();
            ((delegate* unmanaged[Stdcall]<IntPtr, int>)Slot(buffer, 4))(buffer);
            return floats;
        }
        finally
        {
            Release(buffer);
        }
    }

    public void Dispose()
    {
        Release(_reader);
        _reader = IntPtr.Zero;
    }

    public static void Release(IntPtr unknown)
    {
        if (unknown != IntPtr.Zero)
            ((delegate* unmanaged[Stdcall]<IntPtr, uint>)Slot(unknown, 2))(unknown);
    }

    private static void* Slot(IntPtr unknown, int index) => (*(void***)unknown)[index];

    private static void Call(IntPtr attributes, int slot, Guid* key, uint value) =>
        ((delegate* unmanaged[Stdcall]<IntPtr, Guid*, uint, int>)Slot(attributes, slot))(attributes, key, value);

    private static uint GetUInt32(IntPtr attributes, Guid key, uint fallback)
    {
        uint value = 0;
        return ((delegate* unmanaged[Stdcall]<IntPtr, Guid*, uint*, int>)Slot(attributes, 7))(attributes, &key, &value) >= 0 ? value : fallback;
    }

    private static ulong GetUInt64(IntPtr attributes, Guid key)
    {
        ulong value = 0;
        return ((delegate* unmanaged[Stdcall]<IntPtr, Guid*, ulong*, int>)Slot(attributes, 8))(attributes, &key, &value) >= 0 ? value : 0;
    }

    [DllImport("mfplat.dll")]
    private static extern int MFStartup(uint version, uint flags);

    [DllImport("mfplat.dll")]
    private static extern int MFCreateAttributes(out IntPtr attributes, uint initialSize);

    [DllImport("mfplat.dll")]
    private static extern int MFCreateMediaType(out IntPtr type);

    [DllImport("mfreadwrite.dll", CharSet = CharSet.Unicode)]
    private static extern int MFCreateSourceReaderFromURL(string url, IntPtr attributes, out IntPtr reader);
}
