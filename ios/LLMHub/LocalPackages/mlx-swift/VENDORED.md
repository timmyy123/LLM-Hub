# MLX sources for LLM Hub

These are regular source files tracked by LLM Hub, with no Git submodules.

- MLX Swift 0.32.3: `19601207e9a0de51e03ee6ec0c3c5f3784275075`
- MLX core 0.32.4: `36969c094354086e15d2a327a9c50feb20b9cf11`
- MLX C bridge: `ebc88f10caa1b625e6b581437a8dea6df8a70085` (from MLX Swift 0.32.3)

Upstream repositories: https://github.com/ml-explore/mlx-swift,
https://github.com/ml-explore/mlx, https://github.com/ml-explore/mlx-c.

The Swift package's MLX_VERSION definition matches the bundled core (0.32.4).
The new CUDA sources `wddm.cpp` and `cusolver_utils.cpp` are excluded from
Apple/CPU builds with the other CUDA backend sources.
The Metal argument handling uses the unmodified MLX 0.32.4 implementation.
`mlx-c-core-0.32.4.patch` supplies the new optional `gather_qmm` global scale
as `std::nullopt`, preserving the C/Swift bridge signature and its existing default
behavior with MLX core 0.32.4.
Embedded Metal kernels, CPU preamble, CUDA source header, and Metal include files
are regenerated from core 0.32.4. To regenerate after future source updates, run
`python3 tools/regenerate-vendored-sources.py` from this package directory with
the Xcode Metal toolchain installed. Generated files are included in this repo.
All upstream license files are retained. No Xcode-cache edits are needed.
