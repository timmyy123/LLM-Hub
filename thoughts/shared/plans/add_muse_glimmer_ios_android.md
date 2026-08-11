# Add Muse Glimmer to iOS and Android

- Add the eight pinned Muse Glimmer GGUF variants below 16 GB to iOS with the BF16 vision projector as an additional file.
- Add matching Android catalog entries, fully commented out until the bundled GenieX runtime supports the architecture.
- Keep the KQuant projector and DFlash drafter out because neither app runtime exposes working DFlash VLM orchestration.
- Bump RunAnywhere's llama.cpp pin from `b10303` to `b10360`, rebuild the local Apple XCFrameworks, and verify Muse Glimmer symbols/metadata support in every LlamaCPP slice.

## Completion notes

- Added all eight qualifying main-model files from the pinned Hugging Face revision. Eligibility is based on each main GGUF being below 16,000,000,000 bytes, and each entry reports only its own file size.
- Listed the ordinary BF16 and Q8_0 vision projectors as independent catalog downloads, matching the existing GGUF vision-model pattern.
- Enabled the entries in the iOS catalog and enclosed the matching Android entries in one block comment.
- Left `dflash-kquant.gguf` and `mmproj-Muse-Glimmer-30B-kquant.gguf` out. DFlash is a block-diffusion speculative drafter, while the app runtimes do not expose compatible multimodal DFlash orchestration.
- Updated `LLAMACPP_VERSION` to `b10360` and completed the repository's full `build-core-xcframework.sh` workflow successfully. It rebuilt the device, simulator, and macOS slices, packaged the XCFrameworks, and synchronized the React Native and Flutter local binaries.
- Confirmed Muse Glimmer vision graph/preprocessor symbols in every generated `RABackendLLAMACPP` archive. `swiftc -parse` and `git diff --check` both pass.
