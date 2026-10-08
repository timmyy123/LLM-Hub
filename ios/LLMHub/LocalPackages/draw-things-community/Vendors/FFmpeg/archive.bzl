"""Build FFmpeg with its own configure/Makefile, then bind its private symbols."""

load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")

def _ffmpeg_archive_impl(ctx):
    output = ctx.actions.declare_file("lib%s.a" % ctx.label.name)
    platform = ctx.fragments.apple.single_arch_platform
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
    sdk = {
        apple_common.platform.ios_device: "iphoneos",
        apple_common.platform.ios_simulator: "iphonesimulator",
        apple_common.platform.macos: "macosx",
    }.get(platform)
    if not sdk:
        fail("Unsupported FFmpeg platform: %s" % platform)
    configure = [f for f in ctx.files.sources if f.basename == "configure" and f.short_path.endswith("/configure")][0]
    args = ctx.actions.args()
    args.add(ctx.file._script)
    args.add(sdk)
    args.add(ctx.fragments.apple.single_arch_cpu)
    args.add(str(xcode_config.minimum_os_for_platform_type(platform.platform_type)))
    args.add(configure.dirname)
    args.add(ctx.file._patch.dirname)
    args.add(output)
    args.add("asan" if "asan" in ctx.features else "none")
    args.add("gpl" if ctx.attr.gpl else "lgpl")
    args.add(ctx.file._tools.path)
    for codec in ctx.attr.codecs:
        args.add(codec.label.name.replace("_", "-"))
        args.add(codec[DefaultInfo].files.to_list()[0].path)
    apple_support.run(
        actions = ctx.actions,
        apple_fragment = ctx.fragments.apple,
        xcode_config = xcode_config,
        executable = "/bin/bash",
        arguments = [args],
        inputs = ctx.files.sources + ctx.files._support + ctx.files.codecs + [ctx.file._script, ctx.file._patch, ctx.file._tools],
        outputs = [output],
        mnemonic = "BuildFFmpeg",
    )
    return [DefaultInfo(files = depset([output]))]

ffmpeg_archive = rule(
    implementation = _ffmpeg_archive_impl,
    attrs = dict(apple_support.action_required_attrs(), **{
        "sources": attr.label(default = "@ffmpeg//:sources"),
        "gpl": attr.bool(default = True),
        "codecs": attr.label_list(default = [":svt_av1", ":dav1d", ":opus", ":lame", ":libvmaf", ":x264", ":x265", ":libvpx"]),
        "_tools": attr.label(default = ":build_tools", allow_single_file = True),
        "_script": attr.label(default = ":build_ffmpeg.sh", allow_single_file = True),
        "_patch": attr.label(default = ":ffmpeg-ios.patch", allow_single_file = True),
        "_support": attr.label(default = ":support"),
    }),
    fragments = ["apple"],
)
