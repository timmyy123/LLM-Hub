// swift-tools-version: 5.9
// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import PackageDescription

let package = Package(
  name: "LiteRTLM",
  platforms: [
    .iOS(.v15),
    .macOS(.v12),
  ],
  products: [
    .library(
      name: "LiteRTLM",
      targets: ["LiteRTLM"]
    ),
    .library(
      name: "LiteRTLMFoundationModels",
      targets: ["LiteRTLMFoundationModels"]
    ),
    // LLM Hub: the raw C library, for magenta-runtime's MagentaLiteRTBridge (it calls LiteRT's C API, which the
    // framework also exports, to run Magenta RealTime's MusicCoCa encoder).
    .library(
      name: "CLiteRTLMRuntime",
      targets: ["CLiteRTLM"]
    ),
  ],
  targets: [
    // The Prebuilt Binary Target for iOS (Official Stable Release)
    // For nightly builds or pinning to specific nightly dates, switch to the 'nightly' branch or see:
    // https://github.com/google-ai-edge/LiteRT-LM/tree/main/samples/ios_and_mac#1-add-dependency
    // LLM Hub: the v0.19.0 tag's Package.swift still names the v0.18.1 zips; these are the v0.19.0 release assets
    // (checksums = their SHA-256 and GitHub's digests), which match the v0.19.0 Swift sources (C API 1.0.0).
    .binaryTarget(
      name: "CLiteRTLM",
      url:
        "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.19.0/CLiteRTLM.xcframework.zip",
      checksum: "b7b2450fc084058adc5185498f8b7810a9297e054a10cf55818b62e1d5f8d65d"
    ),
    // The Prebuilt Binary Target for Mac
    .binaryTarget(
      name: "CLiteRTLM_mac",
      url:
        "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.19.0/CLiteRTLM_mac.xcframework.zip",
      checksum: "1a9e283a99a119aefe21afda541a5308a7d88aa9cb12f58aa7974038e3369c7c"
    ),
    // The Swift Wrapper Target
    .target(
      name: "LiteRTLM",
      dependencies: [
        .target(name: "CLiteRTLM", condition: .when(platforms: [.iOS])),
        .target(name: "CLiteRTLM_mac", condition: .when(platforms: [.macOS])),
      ],
      path: "swift",
      exclude: [
        "apple_fm",
        "ModelInfoTests.swift",
        "EngineTests.swift",
        "EmbeddingEngineTests.swift",
        "ConversationTests.swift",
        "ToolTests.swift",
        "MessageTests.swift",
        "BUILD",
        "Info.plist",
      ],
    ),
    // Apple Foundation Models Adapter
    .target(
      name: "LiteRTLMFoundationModels",
      dependencies: ["LiteRTLM"],
      path: "swift/apple_fm",
      exclude: [
        "BUILD",
        "main.swift",
        "AdapterTests.swift",
      ]
    ),
    // Separate test targets for each file to avoid naming conflicts:
    .testTarget(
      name: "ModelInfoTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["ModelInfoTests.swift"]
    ),
    .testTarget(
      name: "ConversationTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["ConversationTests.swift"]
    ),
    .testTarget(
      name: "ToolTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["ToolTests.swift"]
    ),
    .testTarget(
      name: "EngineTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["EngineTests.swift"]
    ),
    .testTarget(
      name: "EmbeddingEngineTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["EmbeddingEngineTests.swift"]
    ),
    .testTarget(
      name: "MessageTests",
      dependencies: ["LiteRTLM"],
      path: "swift",
      sources: ["MessageTests.swift"]
    ),
  ]
)
