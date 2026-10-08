# SwiftSoup

Upstream: https://github.com/scinfu/SwiftSoup

Vendored release: **2.13.9** (2026-08-27)

Upstream commit: `18b80329749eca5ea29fc50211dca5c7eff5bfec`

The upstream Swift sources, tests, resources, and profiling tools are included.
Swift files use this repository's `.swift-format.json`; `BUILD` supplies the local
Bazel integration. Upstream Xcode projects and workspace metadata are omitted.
The upstream podspec still declares 2.11.3; the tag and commit above identify the
actual vendored source version.

To update, copy those files from the tagged upstream archive, preserve `BUILD`,
format the Swift files, and update this provenance record. Validate with the
SwiftSoup SwiftPM test suite and `bazel test //Libraries/WebSearch:WebSearchTests`.
