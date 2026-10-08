"""Static Apple builds of optional FFmpeg dependencies using upstream build systems."""

load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")

def _source_root(files):
    # Every source archive is a separate external repository.
    return files[0].owner.workspace_root

def _build_tools_impl(ctx):
    output = ctx.actions.declare_directory(ctx.label.name)
    args = [ctx.file._script.path, "tools", output.path, ctx.file._cmake.path]
    inputs = [ctx.file._script, ctx.file._cmake]
    for name in ["ninja", "pkgconf"]:
        files = getattr(ctx.files, name)
        args.extend([name, _source_root(files)])
        inputs.extend(files)
    apple_support.run(
        actions = ctx.actions,
        apple_fragment = ctx.fragments.apple,
        xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
        executable = ctx.file._python,
        arguments = args,
        inputs = inputs,
        outputs = [output],
        mnemonic = "FFmpegBuildTools",
    )
    return [DefaultInfo(files = depset([output]))]

ffmpeg_build_tools = rule(
    implementation = _build_tools_impl,
    attrs = dict(apple_support.action_required_attrs(), **{
        "ninja": attr.label(default = "@ffmpeg_ninja//:sources"),
        "pkgconf": attr.label(default = "@ffmpeg_pkgconf//:sources"),
        "_script": attr.label(default = ":build_codec.py", allow_single_file = True),
        "_python": attr.label(default = "@ffmpeg_host_tools//:python3", allow_single_file = True),
        "_cmake": attr.label(default = "@ffmpeg_host_tools//:cmake", allow_single_file = True),
    }),
    fragments = ["apple"],
)

def _codec_impl(ctx):
    output = ctx.actions.declare_directory(ctx.label.name)
    platform = ctx.fragments.apple.single_arch_platform
    xcode = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
    sdk = {
        apple_common.platform.ios_device: "iphoneos",
        apple_common.platform.ios_simulator: "iphonesimulator",
        apple_common.platform.macos: "macosx",
    }.get(platform)
    if not sdk:
        fail("Unsupported FFmpeg platform: %s" % platform)
    args = [
        ctx.file._script.path, ctx.attr.codec, output.path, ctx.file._cmake.path,
        sdk, ctx.fragments.apple.single_arch_cpu,
        str(xcode.minimum_os_for_platform_type(platform.platform_type)),
        _source_root(ctx.files.sources), ctx.file._tools.path,
        _source_root(ctx.files._meson), "asan" if "asan" in ctx.features else "none",
        ctx.file._host.path,
        ctx.file._configure.path,
    ]
    apple_support.run(
        actions = ctx.actions,
        apple_fragment = ctx.fragments.apple,
        xcode_config = xcode,
        executable = ctx.file._python,
        arguments = args,
        inputs = ctx.files.sources + ctx.files._meson + [ctx.file._script, ctx.file._cmake, ctx.file._tools, ctx.file._host, ctx.file._configure],
        outputs = [output],
        mnemonic = "BuildFFmpegCodec",
    )
    return [DefaultInfo(files = depset([output]))]

ffmpeg_codec = rule(
    implementation = _codec_impl,
    attrs = dict(apple_support.action_required_attrs(), **{
        "codec": attr.string(mandatory = True),
        "sources": attr.label(mandatory = True),
        "_script": attr.label(default = ":build_codec.py", allow_single_file = True),
        "_host": attr.label(default = ":codec_host.h", allow_single_file = True),
        "_configure": attr.label(default = ":codec_configure.c", allow_single_file = True),
        "_python": attr.label(default = "@ffmpeg_host_tools//:python3", allow_single_file = True),
        "_cmake": attr.label(default = "@ffmpeg_host_tools//:cmake", allow_single_file = True),
        "_meson": attr.label(default = "@ffmpeg_meson//:sources"),
        "_tools": attr.label(default = ":build_tools", allow_single_file = True),
    }),
    fragments = ["apple"],
)
