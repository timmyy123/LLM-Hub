# Draw Things Community

This is a community repository that maintains public-facing code that runs [the Draw Things app](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820).

Currently, it contains the source code for our re-implementation of image generation models, samplers, data models, trainer within the app. Over time, as we move more core functionalities into separate libraries, this repository will grow.

# Contributions

While we expect the development to be mainly carried out by us, we welcome contributions from the community. We require a Contributor License Agreement to make this more manageable. If you prefer not to sign the CLA, you're still welcome to fork this repository and contribute in your own way.

# Roadmap and Repository Sync

The Draw Things app is managed through a private mono-repository. We do per-commit sync with the community repository in both ways. Thus, internal implemented features would be "leaked" to the community repository from time to time and it is expected.

That being said, we don't plan to publish roadmap for internal mono-repository. Whether we would like to publish roadmap for public community repository would require further deliberation and feedback from the community.

# Quick Start with Swift Package Manager on macOS

If you want to build the SwiftPM `gRPCServerCLI` target locally, you can build it with:

```bash
swift build --target gRPCServerCLI
```

To run the CLI with SwiftPM, use the executable name form (not `--target`):

```bash
swift run -c release gRPCServerCLI /path/to/models
```

You can also inspect CLI options with:

```bash
swift run -c release gRPCServerCLI --help
```

# Quick Start with draw-things-cli on macOS

If you want to build the local inference and LoRA training CLI from this repository, build the SwiftPM product with:

```bash
swift build --product draw-things-cli
```

To inspect the CLI surface from SwiftPM, run:

```bash
swift run -c release draw-things-cli --help
swift run -c release draw-things-cli generate --help
```

A minimal generation run looks like:

```bash
swift run -c release draw-things-cli generate --model flux_2_klein_4b_q6p.ckpt --prompt "a small red cube on a table"
```

On macOS, `draw-things-cli` defaults to the same model directory used by the sandboxed Draw Things app:

```text
~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models
```

You can override that with `--models-dir` or `DRAWTHINGS_MODELS_DIR`.

If you prefer Homebrew, install the latest CLI with:

```bash
brew install --HEAD drawthingsai/draw-things/draw-things-cli
```

Then verify the install with:

```bash
draw-things-cli --help
```

# Self-host gRPCServerCLI from Packaged Binaries

We provide pre-built self-hosted gRPCServerCLI binaries through this repository. Latest version should be available at [Releases](https://github.com/drawthingsai/draw-things-community/releases).

These pre-built binaries provide a quick way to host Draw Things gRPC Server on your Mac or Linux systems without download the Draw Things app. Draw Things app then can connect to these self-hosted servers through Server-Offload feature within your network.

## macOS

On macOS, simply download the gRPCServerCLI-macOS on your macOS systems. You can put it under `/usr/local/bin` or anywhere you feel comfortable, and launch it with:

```
gRPCServerCLI-macOS /the-path-to-host-the-models
```

If you have Draw Things app installed, you can simply refer the model path by doing:

```
gRPCServerCLI-macOS ~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models
```

## CUDA-capable Linux

We provides CUDA-accelerated gRPCServerCLI for Linux systems. The simplest way to start is to use [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

Note that as of writing, the gRPCServerCLI supports RTX 20xx / RTX 30xx / RTX 40xx and should support A100 up to H100 NVIDIA cards. Our Docker image uses CUDA 12.4.1 as the base, which requires NVIDIA graphics driver version 550.54.15 or above.

Once both are properly installed. You can pull the latest gRPCServerCLI Docker image and then verify that GPUs are accessible from the container:

```
docker run --gpus all drawthingsai/draw-things-grpc-server-cli:latest nvidia-smi
```

If it shows your GPUs, you are OK. A typical output looks like this:

```
liu@rtx6x4:~/workspace/draw-things-community$ docker run --gpus all drawthingsai/draw-things-grpc-server-cli:latest nvidia-smi                                        (main)
Unable to find image 'drawthingsai/draw-things-grpc-server-cli:latest' locally
latest: Pulling from drawthingsai/draw-things-grpc-server-cli
Digest: sha256:71133ce6e827c432de3f18052bee2c0a74b3236a7abe8c39584ad6397dd6c9b0
Status: Downloaded newer image for drawthingsai/draw-things-grpc-server-cli:latest

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

Fri Dec  6 18:50:28 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:21:00.0 Off |                  Off |
| 30%   42C    P8             28W /  300W |      12MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:22:00.0 Off |                  Off |
| 30%   50C    P8             31W /  300W |      12MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:41:00.0 Off |                  Off |
| 30%   48C    P8             33W /  300W |      89MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:43:00.0 Off |                  Off |
| 30%   47C    P8             28W /  300W |      12MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

The Docker image is maintained at [https://hub.docker.com/r/drawthingsai/draw-things-grpc-server-cli](https://hub.docker.com/r/drawthingsai/draw-things-grpc-server-cli).

To run the actual gRPCServerCLI from Docker, do:

```
docker run -v /[Your local path to store models]:/grpc-models -p 7859:7859 --gpus all drawthingsai/draw-things-grpc-server-cli:latest gRPCServerCLI /grpc-models
```

Note that the containerized process cannot advertise its addresses to your local network hence you need to manually type the IP address and port inside Draw Things app to successfully connect.

For RTX 20xx graphics cards, you need to disable FlashAttention:

```
docker run -v /[Your local path to store models]:/grpc-models -p 7859:7859 --gpus all drawthingsai/draw-things-grpc-server-cli:latest gRPCServerCLI /grpc-models --no-flash-attention
```

# Getting Started for Development

We use [Bazel](https://bazel.build/) as our main build system. You can build components either on Linux or on macOS. On macOS, you need to have Xcode installed. On Linux, depending on whether you have NVIDIA CUDA-compatible GPU, the setup can be different.

After neccessary dependencies installed, you can run:

```bash
./Scripts/install.sh
```

To setup the repo properly.

To verify the installation, run:

```bash
bazel run Apps:ModelConverter
```

# Self-host gRPCServerCLI from Scratch

gRPCServerCLI is fully open-source, meaning that on the same system, you can build byte-by-byte matching prebuilt binaries as our provided ones. This is relatively easy on macOS systems, and requires a bit more setup on others.

## macOS

First, you need to setup this repository on your macOS. To build anything with this repository, you need [Homebrew](https://brew.sh/) and [Xcode](https://developer.apple.com/xcode/).

To setup the repository, simply run:

```
./Scripts/install.sh
```

This should setup necessary tools, such as Bazel for you. To build gRPCServerCLI, you can run:

```
bazel build Apps:gRPCServerCLI-macOS
```

To build exactly the same as our prebuilt-binary:

```
bazel build Apps:gRPCServerCLI-macOS --nostamp --config=release --macos_cpus=arm64,x86_64
```

## CUDA-capable Linux

We will show how to do build gRPCServerCLI with Ubuntu 22.04. First, you need to install [CUDA](https://developer.nvidia.com/cuda-downloads), [Swift](https://www.swift.org/install/linux/#platforms), [Bazel](https://github.com/bazelbuild/bazelisk/releases) and [CUDNN](https://developer.nvidia.com/cudnn).

Afterwards, you can `sudo apt install` the rest:

```
sudo apt -y install libpng-dev libjpeg-dev libatlas-base-dev libblas-dev clang llvm
```

Make sure when running `./Scripts/install.sh`, the `clang.bazelrc` file successfully generated. This file should look like this:

```
# Generated file, do not edit. If you want to disable clang, just delete this file.
build:clang --action_env='PATH=/usr/local/bin:/home/liu/.deno/bin:/opt/cmake/bin:/opt/intel/bin:/opt/intel/vtune_amplifier_xe/bin64:/opt/go/bin:/opt/llvm/tools/clang/tools/scan-build:/usr/local/cuda/bin:/opt/llvm/tools/clang/tools/scan-view:/usr/local/bin:/usr/local/sbin/:/bin:/sbin:/usr/bin:/usr/sbin:/home/liu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/liu/.rvm/bin:/opt/go/bin:/home/liu/.local/bin:/opt/swift/usr/bin:/home/liu/.cargo/bin' --host_action_env='PATH=/usr/local/bin:/home/liu/.deno/bin:/opt/cmake/bin:/opt/intel/bin:/opt/intel/vtune_amplifier_xe/bin64:/opt/go/bin:/opt/llvm/tools/clang/tools/scan-build:/usr/local/cuda/bin:/opt/llvm/tools/clang/tools/scan-view:/usr/local/bin:/usr/local/sbin/:/bin:/sbin:/usr/bin:/usr/sbin:/home/liu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/liu/.rvm/bin:/opt/go/bin:/home/liu/.local/bin:/opt/swift/usr/bin:/home/liu/.cargo/bin'
build:clang --action_env=CC=clang --host_action_env=CC=clang
build:clang --action_env=CXX=clang++ --host_action_env=CXX=clang++
build:clang --action_env='LLVM_CONFIG=/usr/local/bin/llvm-config' --host_action_env='LLVM_CONFIG=/usr/local/bin/llvm-config'
build:clang --repo_env='LLVM_CONFIG=/usr/local/bin/llvm-config'
build:clang --linkopt='-L/usr/local/lib'
build:clang --linkopt='-Wl,-rpath,/usr/local/lib'

build:clang-asan --action_env=ENVOY_UBSAN_VPTR=1
build:clang-asan --copt=-fsanitize=vptr,function
build:clang-asan --linkopt=-fsanitize=vptr,function
build:clang-asan --linkopt='-L/usr/local/lib/clang/18.1.1/lib/x86_64-unknown-linux-gnu'
build:clang-asan --linkopt=-l:libclang_rt.ubsan_standalone.a
build:clang-asan --linkopt=-l:libclang_rt.ubsan_standalone_cxx.a
```

There should also be a `.bazelrc.local` file contains CUDA configuration:

```
build --action_env TF_NEED_CUDA="1"
build --action_env TF_NEED_OPENCL="1"
build --action_env TF_CUDA_CLANG="0"
build --action_env HOST_CXX_COMPILER="/usr/local/bin/clang"
build --action_env HOST_C_COMPILER="/usr/local/bin/clang"
build --action_env CLANG_CUDA_COMPILER_PATH="/usr/local/bin/clang"
build --action_env GCC_HOST_COMPILER_PATH="/usr/local/bin/clang"

build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda"
build --action_env TF_CUDA_VERSION="12.4"
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0,8.6"
build --action_env COMPUTECPP_TOOLKIT_PATH="/usr/local/computecpp"
build --action_env TMP="/tmp"
build --action_env TF_CUDNN_VERSION="9"
build --action_env CUDNN_INSTALL_PATH="/usr"
build --action_env TF_NCCL_VERSION="2"
build --action_env NCCL_INSTALL_PATH="/usr"

build --config=clang
build --config=cuda

build --linkopt="-z nostart-stop-gc"
build --host_linkopt="-z nostart-stop-gc"

build --define=enable_sm80=true
```

Pay attention to CUDA version and CUDNN version to match yours.

To build the exact binary as the ones we use for the Docker image, do the following:

```
bazel build Apps:gRPCServerCLI --keep_going --spawn_strategy=local --compilation_mode=opt
```

Note that you might encounter issues with cannot find `cudnn`. In that case, make sure it is properly named, I have to do:

```
sudo ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so
```

to make it discoverable.

# License

This repository is licensed under GNU General Public License version 3 (GPL-v3) just like [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) or [Fooocus](https://github.com/lllyasviel/Fooocus/).

## License Examples

To illustrate how GPL-v3 applies in various scenarios, consider the following examples of acceptable licensing practices under our policy:

### Standalone Applications

If you develop a new application using code from this repository, and you release it on any app distribution platform (e.g., App Store) as a standalone product (meaning it functions independently and is not merely an HTTP client for Draw Things), you are required to license the entire application under GPL-v3.

### Forking and Enhancements

Should you fork this repository and introduce enhancements or additional functionalities, any new code you contribute must also be licensed under GPL-v3. This ensures all derivative works remain open and accessible under the same terms.

### Server and Client Development

If you use this repository to build a server application, either through HTTP or Google Protobuf (not limited to either), and subsequently develop a client application that communicates with your server, the following rules apply:

 * The source code for both the Google Protobuf definitions and the server must be published under GPL-v3.
 * The client application can be licensed under a different license, provided you do not bundle the client and server into a single distribution package or facilitate the automatic download of the server from within the client (a practice often referred to as "deep integration"). This distinction ensures that while the server-side components remain open, developers have flexibility in licensing client-side applications.

These examples are meant to provide guidance on common use cases involving our codebase. By adhering to these practices, you help maintain the spirit of open collaboration and software freedom that GPL-v3 champions.

## Alternative Licenses

We can provide alternative licenses for this repo, but not any 3rd-party forks. In particular, we may provide [LGPL-v3](https://www.gnu.org/licenses/lgpl-3.0.en.html) license as an alternative to free software on case-by-case basis. If you want to acquire more liberal licenses for closed-source applications or other legal obligations, please contact us. 
