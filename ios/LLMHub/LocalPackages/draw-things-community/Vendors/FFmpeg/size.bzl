"""Comparable linked-size fixtures; no app assets, signing or debug symbols."""
load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")

def _size_impl(ctx):
    platform = ctx.fragments.apple.single_arch_platform
    xcode = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
    sdk = {
        apple_common.platform.ios_device: "iphoneos",
        apple_common.platform.ios_simulator: "iphonesimulator",
        apple_common.platform.macos: "macosx",
    }[platform]
    binary = ctx.actions.declare_file(ctx.label.name + ".macho")
    report = ctx.actions.declare_file(ctx.label.name + ".json")
    apple_support.run(
        actions = ctx.actions,
        apple_fragment = ctx.fragments.apple,
        xcode_config = xcode,
        executable = ctx.file._python,
        arguments = [ctx.file._script.path, sdk, ctx.fragments.apple.single_arch_cpu,
                     str(xcode.minimum_os_for_platform_type(platform.platform_type)),
                     ctx.file.archive.path, ctx.file._source.path, ctx.file._header.path,
                     binary.path, report.path],
        inputs = [ctx.file.archive, ctx.file._script, ctx.file._source, ctx.file._header],
        outputs = [binary, report],
        mnemonic = "MeasureFFmpegSize",
    )
    return [DefaultInfo(files = depset([binary, report]))]

ffmpeg_size = rule(
    implementation = _size_impl,
    attrs = dict(apple_support.action_required_attrs(), **{
        "archive": attr.label(allow_single_file = True, mandatory = True),
        "_script": attr.label(default = ":measure_size.py", allow_single_file = True),
        "_python": attr.label(default = "@ffmpeg_host_tools//:python3", allow_single_file = True),
        "_source": attr.label(default = ":SizeProbe.c", allow_single_file = True),
        "_header": attr.label(default = ":FFmpegCommand.h", allow_single_file = True),
    }),
    fragments = ["apple"],
)
