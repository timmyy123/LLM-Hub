"""Write a self-contained AWS instrumentation spec; no model weights in the upload.

Usage: make_test_spec.py sdxl|sd3|embedgemma|embedgemma_fallback|litert_sd|litert_sd_app|soundgen|soundgen_long|soundgen_gpu|soundgen_app|llama_gpu OUTPUT.yml [--soc=REGEX] [--build=tensor_g5] [VK_ENV ...]

Device Farm wraps each command separately, so every command must be a single shell line.
"""
import base64
import json
from pathlib import Path
import sys

SUITES = {
    "sdxl": ("SdxlGenerationDeviceTest", "sdxlEncodersAndModifiersAffectGeneration",
             ["sdxl.safetensors", "clip-g.safetensors", "clip-g-zero-projection.safetensors", "lora.safetensors"]),
    "sd3": ("Sd3GenerationDeviceTest", "sd3TextEncodersAffectGeneration",
            ["sd3.gguf", "clip-l.safetensors", "clip-g.safetensors", "t5.gguf", "t5-negated-norm.gguf"]),
    "embedgemma": ("com.llmhub.llmhub.embedding.EmbeddingGemmaNpuDeviceTest", "npuEmbeddings",
                   ["embeddinggemma-2-text-270m_MediaTek_MT6991.litertlm", "embeddinggemma-2-740m_MediaTek_MT6991.litertlm"]),
    "embedgemma_fallback": ("com.llmhub.llmhub.mediasearch.MediaSearchNpuFallbackDeviceTest", "videoMomentFinderLoads",
                            ["embeddinggemma-2-740m_MediaTek_MT6991.litertlm", "embeddinggemma-2-740m.litertlm"]),
    # A per-SoC AOT build next to the uncompiled models, each with the CLIP tokenizer files.
    "litert_sd": ("LiteRtSdDeviceTest", "npuBuildGenerates",
                  [f"{d}/{f}" for d in ("aot", "source") for f in
                   ("diffusion.tflite", "decoder.tflite", "clip.tflite", "vocab.json", "merges.txt")]),
    "soundgen": ("com.llmhub.llmhub.soundgen.SoundGenDeviceTest", "everyModelGeneratesAudio",
                 ["dit_model.tflite", "autoencoder_model_undilated_1g.tflite", "conditioners_float32.tflite", "spiece.model",
                  "dit_L256_int8.tflite", "ae_dec_L256_int8.tflite", "t5gemma_enc_int8.tflite", "tokenizer.model",
                  "dit_L2048_int8.tflite", "ae_dec_L2048_int8.tflite"]),
    "soundgen_long": ("com.llmhub.llmhub.soundgen.SoundGenDeviceTest", "hdLongGeneratesAudio",
                      ["dit_L2048_int8.tflite", "ae_dec_L2048_int8.tflite", "t5gemma_enc_int8.tflite", "tokenizer.model"]),
    # A GGUF model on the GPU through the app's llama.cpp service; the test downloads it (--arg=downloadFixtures=1).
    "llama_gpu": ("com.llmhub.llmhub.inference.LlamaCppGpuDeviceTest", "ggufGeneratesOnGpu", []),
    # SoundGen Quick and HD through the app's backend; the test downloads them on the phone (--arg=downloadFixtures=1).
    "soundgen_app": ("com.llmhub.llmhub.soundgen.SoundGenDeviceTest", "quickAndHdGenerateAudio", []),
    # SoundGen Quick decoders on the GPU; the test downloads them on the phone (--arg=downloadFixtures=1).
    "soundgen_gpu": ("com.llmhub.llmhub.soundgen.SoundGenGpuOptionsDeviceTest", "quickDecoderGpuOptions", []),
    # The in-app path: one AOT build laid out as the downloader writes it.
    "litert_sd_app": ("LiteRtSdAppDeviceTest", "appGeneratesOrFailsWithoutCrashing",
                      [f"app/{f}" for f in ("diffusion.tflite", "decoder.tflite", "clip.tflite", "vocab.json", "merges.txt")]),
}
suite, output = sys.argv[1], Path(sys.argv[2])
# Optional ggml-vulkan switch sets, each probed in its own process before the full test, e.g. GGML_VK_DISABLE_DOT2=1
# --soc=REGEX limits the run to matching ro.soc.model values; the default covers Snapdragon 8 Gen 3 and newer.
soc = next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("--soc=")), "SM8650|SM8750|SM8850")
# --build=FOLDER picks the litert_sd AOT build; for embedgemma_fallback it is the NPU build's SoC tag.
build = next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("--build=")), "")
# --arg=KEY=VALUE passes an instrumentation argument (-e KEY VALUE) to the test.
extra = "".join(f" -e {a[6:].split('=', 1)[0]} {a[6:].split('=', 1)[1]}" for a in sys.argv[3:] if a.startswith("--arg="))
probes = [a for a in sys.argv[3:] if not a.startswith(("--soc=", "--build=", "--arg="))]
test_class, test_method, fixture_names = SUITES[suite]
if suite == "embedgemma_fallback" and build:
    fixture_names = [name.replace("MediaTek_MT6991", build) for name in fixture_names]
test_class = test_class if "." in test_class else f"com.llmhub.llmhub.imagegen.{test_class}"
script = base64.b64encode(Path(__file__).with_name("prepare_fixtures.py").read_bytes()).decode()
device = 'adb -s "$DEVICEFARM_DEVICE_UDID"'
package = "com.llmhub.llmhub"
host = f"/tmp/llmhub-{suite}-fixtures"
app = f"files/{suite}-test-fixtures"
log = '"$DEVICEFARM_LOG_DIR'
subdirs = sorted({name.rsplit("/", 1)[0] for name in fixture_names if "/" in name})
stage = [f'{device} shell run-as {package} mkdir -p {" ".join(f"{app}/{d}" for d in subdirs)}'] if subdirs else []
stage += [
    f'{device} push "{host}/{name}" "/data/local/tmp/llmhub-{name.replace("/", "_")}" && '
    f'{device} shell run-as {package} cp "/data/local/tmp/llmhub-{name.replace("/", "_")}" "{app}/{name}" && '
    f'{device} shell rm "/data/local/tmp/llmhub-{name.replace("/", "_")}"'
    for name in fixture_names
]
spec = {
    "version": 0.1,
    "android_test_host": "amazon_linux_2",
    "phases": {
        "pre_test": {"commands": [
            f'{device} shell getprop > {log}/device-properties.txt"',
            f'{device} shell cat /proc/meminfo > {log}/memory-before.txt"',
            f'{device} shell getprop ro.soc.model | tee {log}/soc.txt" | grep -E "{soc}"',
            f'df -h /tmp > {log}/host-disk.txt"',
            f'echo {script} | base64 -d > /tmp/llmhub_prepare_fixtures.py',
            f'python3 /tmp/llmhub_prepare_fixtures.py {suite} {host} {build} > {log}/fixtures.log" 2>&1',
            f'{device} shell run-as {package} mkdir -p {app}',
            *stage,
            f'cp {host}/manifest.json {log}/fixture-manifest.json"',
            f'{device} shell run-as {package} ls -l {app} > {log}/staged-fixtures.txt"',
            f'{device} logcat -c',
        ]},
        "test": {"commands": [
            *[f'{device} shell am instrument -r -w --no-window-animation -e class {test_class}#vulkanProbe '
              f'-e vkEnv {env} $DEVICEFARM_TEST_PACKAGE_NAME/$DEVICEFARM_TEST_PACKAGE_RUNNER 2>&1 | '
              f'sed "s/^/[vkEnv {env}] /" | tee -a {log}/probes.log"' for env in probes],
            f'{device} shell am instrument -r -w --no-window-animation{extra} -e class {test_class}#{test_method} '
            f'$DEVICEFARM_TEST_PACKAGE_NAME/$DEVICEFARM_TEST_PACKAGE_RUNNER 2>&1 | tee {log}/instrument.log"',
            f'grep -Eq "OK \\(1 test\\)" {log}/instrument.log"',
            f'! grep -Eq "FAILURES|INSTRUMENTATION_FAILED|Process crashed" {log}/instrument.log"',
        ]},
        "post_test": {"commands": [
            f'{device} logcat -d -v threadtime > {log}/logcat.txt"',
            f'{device} shell dumpsys meminfo {package} > {log}/memory-after.txt"',
            f'{device} exec-out run-as {package} tar -cf - files/{suite}-test-results > {log}/{suite}-results.tar"',
            f'{device} shell run-as {package} rm -rf {app} files/{suite}-test-results files/image_extras',
        ]},
    },
    "artifacts": ["$DEVICEFARM_LOG_DIR"],
}
output.write_text(json.dumps(spec, indent=2))  # JSON is valid YAML.
