# Draw Things vendored source and upstream provenance

The SDK source trees in `LocalPackages/draw-things-community` and
`LocalPackages/media-generation-kit` are regular files owned and tracked by this
repository. This repository uses zero Git submodules. A normal checkout contains
the complete SDK sources, including all local fixes and upstream licenses.

`versions.json` records the upstream revisions used as the source baselines.
The engine uses the latest published release checked on 8 October 2026. The wrapper
uses its latest main revision; its published release is older than the previously
installed wrapper snapshot.

The patches preserve the app's local model storage, download behavior, SVD model
registrations, iOS generation isolation, ZIPFoundation integration, and local package
wiring. The wrapper patch also wires the upstream CLI's CLICloudAuth dependency.

The patch files document differences from the recorded upstream baselines.
Those changes are already included in the tracked SDK source; no patch application
or additional checkout step is needed to build the app. Future SDK updates must
update these regular source files directly and preserve the local fixes.

The app bundles its selected community model metadata and generation presets in
`Sources/LLMHub/models.json` and `Sources/LLMHub/configs.json`.

The engine retains the app's previously working SentencePiece Swift revision
`8d17bf2e017c97563e8805545d676be9739b6c0e`. Its public Swift API is identical to
the revision requested upstream. The newer revision switches the C++ backend to
SentencePiece 0.2.2, whose package currently omits CoreFoundation linkage when
Xcode builds it as a dynamic product for iOS. Retaining the prior revision avoids
that linker failure and keeps the music and generation runtimes on one shared
SentencePiece dependency. The engine and its compute backends remain upgraded.
