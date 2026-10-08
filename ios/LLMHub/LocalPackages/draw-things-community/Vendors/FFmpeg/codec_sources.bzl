"""Pinned upstream releases for the optional FFmpeg codec build."""

CODEC_SOURCES = {
    "svt-av1": {"version": "4.2.0", "url": "https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v4.2.0/SVT-AV1-v4.2.0.tar.bz2", "sha256": "512f2ea5649e3e76c2dddcc25c2556fb67a9582baaab207c9c96161c94659dad", "strip_prefix": "SVT-AV1-v4.2.0"},
    "dav1d": {"version": "1.5.4", "url": "https://code.videolan.org/videolan/dav1d/-/archive/1.5.4/dav1d-1.5.4.tar.bz2", "sha256": "2abfb0c89212e6e4733a54e0ae509ec00a5b845a6360946f918806e14aedb011", "strip_prefix": "dav1d-1.5.4"},
    "opus": {"version": "1.6.1", "url": "https://ftp.osuosl.org/pub/xiph/releases/opus/opus-1.6.1.tar.gz", "sha256": "6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1", "strip_prefix": "opus-1.6.1"},
    "lame": {"version": "4.0", "url": "https://downloads.sourceforge.net/project/lame/lame/4.0/lame-4.0.tar.gz", "sha256": "3df5124d5ad3a98312ffd7ba6a9b36230e4f8a3e66d3ce0f425e336c32d216eb", "strip_prefix": "lame-4.0"},
    "libvmaf": {"version": "3.2.1", "url": "https://github.com/Netflix/vmaf/archive/refs/tags/v3.2.1.tar.gz", "sha256": "5df7386911bc15fd1ca783132528748d219768ae4fc5f8e0b61184f041648092", "strip_prefix": "vmaf-3.2.1"},
    "x264": {"version": "r3222", "url": "https://code.videolan.org/videolan/x264/-/archive/b35605ace3ddf7c1a5d67a2eb553f034aef41d55/x264-b35605ace3ddf7c1a5d67a2eb553f034aef41d55.tar.bz2", "sha256": "6eeb82934e69fd51e043bd8c5b0d152839638d1ce7aa4eea65a3fedcf83ff224", "strip_prefix": "x264-b35605ace3ddf7c1a5d67a2eb553f034aef41d55"},
    "x265": {"version": "4.3", "url": "https://github.com/Multicorewareinc/x265/releases/download/4.3/x265_4.3.tar.gz", "sha256": "83c53e4c8bbb8f1e33ed59e10a7d621d1d7801ca853910c3eb41f038b8ffb121", "strip_prefix": "x265_4.3"},
    "libvpx": {"version": "1.17.0", "url": "https://github.com/webmproject/libvpx/archive/refs/tags/v1.17.0.tar.gz", "sha256": "1020f184046187baa2985dbde38e0691f49c44088bca7a1842b0236c6081dc0a", "strip_prefix": "libvpx-1.17.0"},
    "meson": {"version": "1.12.1", "url": "https://github.com/mesonbuild/meson/releases/download/1.12.1/meson-1.12.1.tar.gz", "sha256": "ab0a6ca09f8ef70c564c8241fb5a23957886a0b53fb58412b5e07eaf07dba743", "strip_prefix": "meson-1.12.1"},
    "ninja": {"version": "1.13.2", "url": "https://github.com/ninja-build/ninja/archive/refs/tags/v1.13.2.tar.gz", "sha256": "974d6b2f4eeefa25625d34da3cb36bdcebe7fbce40f4c16ac0835fd1c0cbae17", "strip_prefix": "ninja-1.13.2"},
    "pkgconf": {"version": "3.0.7", "url": "https://distfiles.ariadne.space/pkgconf/pkgconf-3.0.7.tar.xz", "sha256": "c926ff491cbd9a331a589160811bd97ab1749b4d5198a519338f2cdfabe6940a", "strip_prefix": "pkgconf-3.0.7"},
}
