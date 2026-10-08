load("@build_bazel_rules_apple//apple:macos.bzl", "macos_command_line_application")
load("@build_bazel_rules_apple//apple:resources.bzl", "apple_resource_group")
load(":archive.bzl", "ffmpeg_archive")
load(":codec_build.bzl", "ffmpeg_build_tools", "ffmpeg_codec")
load(":size.bzl", "ffmpeg_size")

package(default_visibility = ["//visibility:public"])

apple_resource_group(
    name = "notices",
    resources = [
        "COPYING.GPLv2",
        "COPYING.LGPLv2.1",
        "LICENSE.md",
        "README.md",
    ],
    structured_resources = glob(["Licenses/**"]),
)

filegroup(
    name = "support",
    srcs = [
        "FFmpegCommand.h",
        "codec_configure.c",
        "ffmpeg_host.c",
        "ffmpeg_host.h",
    ],
)

ffmpeg_archive(name = "archive")

cc_import(
    name = "private_archive",
    static_library = ":archive",
)

objc_library(
    name = "FFmpeg",
    hdrs = ["FFmpegCommand.h"],
    sdk_dylibs = [
        "c++",
        "z",
        "iconv",
        "bz2",
    ],
    sdk_frameworks = [
        "AudioToolbox",
        "CoreFoundation",
        "CoreMedia",
        "CoreVideo",
        "Security",
        "VideoToolbox",
    ],
    deps = [":private_archive"],
)

objc_library(
    name = "command",
    srcs = ["ios_system_ffmpeg.m"],
    deps = [
        ":FFmpeg",
        "//Vendors/ios_system:api",
    ],
)

cc_test(
    name = "FFmpegCommandTests",
    size = "medium",
    srcs = ["FFmpegCommandTests.c"],
    deps = [":FFmpeg"],
)

objc_library(
    name = "ios_system_validation_main",
    srcs = ["IOSSystemFFmpegTests.m"],
    sdk_frameworks = ["Foundation"],
    deps = ["//Vendors/ios_system"],
)

macos_command_line_application(
    name = "ios_system_validation",
    bundle_id = "com.liuliu.ffmpeg-validation",
    minimum_os_version = "13.5",
    deps = [
        ":ios_system_validation_main",
        "//Vendors/NodeJS:NodeMobile",
        "//Vendors/PythonIOS:PythonRuntime",
    ],
)

sh_test(
    name = "IOSSystemFFmpegTests",
    size = "medium",
    srcs = ["run-ios-system-tests.sh"],
    data = [":ios_system_validation"],
)

ffmpeg_build_tools(name = "build_tools")

ffmpeg_codec(
    name = "svt_av1",
    codec = "svt-av1",
    sources = "@ffmpeg_svt_av1//:sources",
)

ffmpeg_codec(
    name = "dav1d",
    codec = "dav1d",
    sources = "@ffmpeg_dav1d//:sources",
)

ffmpeg_codec(
    name = "opus",
    codec = "opus",
    sources = "@ffmpeg_opus//:sources",
)

ffmpeg_codec(
    name = "lame",
    codec = "lame",
    sources = "@ffmpeg_lame//:sources",
)

ffmpeg_codec(
    name = "libvmaf",
    codec = "libvmaf",
    sources = "@ffmpeg_libvmaf//:sources",
)

ffmpeg_codec(
    name = "x264",
    codec = "x264",
    sources = "@ffmpeg_x264//:sources",
)

ffmpeg_codec(
    name = "x265",
    codec = "x265",
    sources = "@ffmpeg_x265//:sources",
)

ffmpeg_codec(
    name = "libvpx",
    codec = "libvpx",
    sources = "@ffmpeg_libvpx//:sources",
)

# Manual targets isolate each library's linked contribution.
ffmpeg_archive(
    name = "archive_baseline",
    codecs = [],
    gpl = False,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_baseline",
    archive = ":archive_baseline",
    tags = ["manual"],
)

ffmpeg_archive(
    name = "archive_gpl",
    codecs = [],
    gpl = True,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_gpl",
    archive = ":archive_gpl",
    tags = ["manual"],
)

ffmpeg_archive(
    name = "archive_svt_av1",
    codecs = [":svt_av1"],
    gpl = True,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_svt_av1",
    archive = ":archive_svt_av1",
    tags = ["manual"],
)

ffmpeg_archive(
    name = "archive_dav1d",
    codecs = [":dav1d"],
    gpl = True,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_dav1d",
    archive = ":archive_dav1d",
    tags = ["manual"],
)

ffmpeg_archive(
    name = "archive_opus",
    codecs = [":opus"],
    gpl = True,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_opus",
    archive = ":archive_opus",
    tags = ["manual"],
)

ffmpeg_archive(
    name = "archive_lame",
    codecs = [":lame"],
    gpl = True,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_lame",
    archive = ":archive_lame",
    tags = ["manual"],
)

ffmpeg_archive(
    name = "archive_libvmaf",
    codecs = [":libvmaf"],
    gpl = True,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_libvmaf",
    archive = ":archive_libvmaf",
    tags = ["manual"],
)

ffmpeg_archive(
    name = "archive_x264",
    codecs = [":x264"],
    gpl = True,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_x264",
    archive = ":archive_x264",
    tags = ["manual"],
)

ffmpeg_archive(
    name = "archive_x265",
    codecs = [":x265"],
    gpl = True,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_x265",
    archive = ":archive_x265",
    tags = ["manual"],
)

ffmpeg_archive(
    name = "archive_libvpx",
    codecs = [":libvpx"],
    gpl = True,
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_libvpx",
    archive = ":archive_libvpx",
    tags = ["manual"],
)

ffmpeg_size(
    name = "size_all",
    archive = ":archive",
    tags = ["manual"],
)

filegroup(
    name = "size_report",
    srcs = [
        ":size_all",
        ":size_baseline",
        ":size_dav1d",
        ":size_gpl",
        ":size_lame",
        ":size_libvmaf",
        ":size_libvpx",
        ":size_opus",
        ":size_svt_av1",
        ":size_x264",
        ":size_x265",
    ],
    tags = ["manual"],
)
