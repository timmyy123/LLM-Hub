"""Bind Lean's implementation and dependencies before exposing its C ABI."""

load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")

def _lean_archive_impl(ctx):
    archives = []
    for dep in ctx.attr.deps:
        for linker_input in dep[CcInfo].linking_context.linker_inputs.to_list():
            if linker_input.user_link_flags or linker_input.additional_inputs:
                fail("Lean's private archive must not discard transitive link options or inputs")
            for library in linker_input.libraries:
                archive = library.static_library or library.pic_static_library
                if not archive:
                    fail("Lean encapsulation requires static libraries")
                archives.append(archive)
    archives = depset(archives)
    output = ctx.actions.declare_file("lib%s.a" % ctx.label.name)
    platform = ctx.fragments.apple.single_arch_platform
    xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]
    sdk = {
        apple_common.platform.ios_device: "iphoneos",
        apple_common.platform.ios_simulator: "iphonesimulator",
        apple_common.platform.macos: "macosx",
    }.get(platform)
    if not sdk:
        fail("Unsupported Lean platform: %s" % platform)
    args = ctx.actions.args()
    args.add(ctx.file._script)
    args.add(sdk)
    args.add(ctx.fragments.apple.single_arch_cpu)
    args.add(str(xcode_config.minimum_os_for_platform_type(platform.platform_type)))
    args.add(ctx.file.symbols)
    args.add(ctx.file.roots)
    args.add(output)
    args.add_all(archives)
    apple_support.run(
        actions = ctx.actions,
        apple_fragment = ctx.fragments.apple,
        xcode_config = xcode_config,
        executable = "/bin/bash",
        arguments = [args],
        # Limit this host-tool override to encapsulation actions.
        env = {"LEAN_LD_CLASSIC": ctx.var.get("LEAN_LD_CLASSIC", "")},
        inputs = depset([ctx.file._script, ctx.file.symbols, ctx.file.roots], transitive = [archives]),
        outputs = [output],
        mnemonic = "EncapsulateLean",
    )

    # Deliberately do not propagate the raw dependencies' CcInfo. Consumers link
    # only the resulting cc_import, never the unencapsulated dependency archives.
    return [DefaultInfo(files = depset([output]))]

lean_encapsulated_archive = rule(
    implementation = _lean_archive_impl,
    attrs = dict(apple_support.action_required_attrs(), **{
        "deps": attr.label_list(providers = [CcInfo]),
        "symbols": attr.label(allow_single_file = True, mandatory = True),
        "roots": attr.label(allow_single_file = True, mandatory = True),
        "_script": attr.label(default = ":encapsulate_lean.sh", allow_single_file = True),
    }),
    fragments = ["apple"],
)
