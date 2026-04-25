# JNI bridge that exposes espeak-ng's text→IPA function to Kotlin.
#
# Layout assumed (populated by scripts/build_espeak_android.sh):
#   src/main/cpp/
#     espeak_jni.cpp          — this directory
#     espeak-ng/include/      — espeak-ng public headers (speak_lib.h)
#   src/main/jniLibs/<abi>/
#     libespeak-ng.so         — cross-compiled espeak-ng runtime per ABI
#   src/main/assets/espeak-ng-data/
#     ...                     — espeak-ng's data files (voices, phoneme tables)
#
# If espeak-ng/include/ doesn't exist (e.g. fresh checkout), this file does
# nothing — the Kotlin layer's `EspeakG2P.tryLoad()` will return null and the
# pipeline falls back to the bundled DictionaryG2P.

cmake_minimum_required(VERSION 3.18.1)
project(mimo_espeak_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(ESPEAK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/espeak-ng/include")

if(NOT EXISTS "${ESPEAK_INCLUDE_DIR}/speak_lib.h")
    message(WARNING "espeak-ng headers not found at ${ESPEAK_INCLUDE_DIR}. "
                    "Run scripts/build_espeak_android.sh to set them up. "
                    "Skipping espeak JNI build.")
    return()
endif()

# Locate the per-ABI prebuilt libespeak-ng.so that the build script dropped
# under src/main/jniLibs/<abi>/. Gradle automatically packages everything in
# jniLibs into the APK/AAB, so we just declare the import target here.
add_library(espeak-ng SHARED IMPORTED)
set_target_properties(espeak-ng PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libespeak-ng.so")

add_library(espeak-jni SHARED espeak_jni.cpp)
target_include_directories(espeak-jni PRIVATE "${ESPEAK_INCLUDE_DIR}")
target_link_libraries(espeak-jni PRIVATE espeak-ng log)
