def _resource_library_impl(ctx):
    c_arrays = []
    c_headers = []
    for src in ctx.files.srcs:
      src_base = ".".join(src.basename.split(".")[:-1])
      c_array = ctx.actions.declare_file(src_base + "_generated.c")
      c_header = ctx.actions.declare_file(src_base + "_generated.h")
      ctx.actions.run(
          inputs = [src],
          outputs = [c_array, c_header],
          arguments = [src.path, c_array.path, c_header.path],
          executable = ctx.executable._packager,
      )
      c_arrays = c_arrays + [c_array]
      c_headers = c_headers + [c_header]

    # Create a CcInfo provider to make this target usable by other C++ rules
    cc_toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    compilation_context, compilation_outputs = cc_common.compile(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        public_hdrs = c_headers,
        srcs = c_arrays,
    )

    linking_context, linking_outputs = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = compilation_outputs,
    )

    return [
        DefaultInfo(files = depset(c_arrays + c_headers)),
        CcInfo(
            compilation_context = compilation_context,
            linking_context = linking_context,
        ),
    ]

resource_library = rule(
    implementation = _resource_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
        "_packager": attr.label(
            executable = True,
            allow_files = True,
            cfg = "exec",
            default = Label("//Tools:Packager"),
        ),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    fragments = ["cpp"],
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)
