// LTX-Video engine: stable-diffusion.cpp with its own ggml build, packaged as libllmhub_sdcpp.so.
// Kept separate from the app's llama.cpp ggml so it can target armv8.2-a (dotprod + fp16), which is
// ~4.5x faster for the video VAE, while the LLM CPU runtime stays on baseline armv8-a.
plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.llmhub.sdcpp"
    compileSdk = 37
    ndkVersion = "29.0.13113456"

    defaultConfig {
        minSdk = 27
        ndk {
            abiFilters += setOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DANDROID_STL=c++_static"
                )
            }
        }
    }

    buildTypes {
        release {
            ndk {
                debugSymbolLevel = "NONE"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}
