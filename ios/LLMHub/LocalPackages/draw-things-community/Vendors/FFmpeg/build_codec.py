#!/usr/bin/env python3
"""Cross-compile a pinned codec with its upstream build system, without host libraries."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def capture(*args):
    return subprocess.check_output(args, text=True).strip()


def run(args, cwd, env):
    result = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        print("Failed:", args, file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        if (Path(cwd) / "config.log").exists():
            print((Path(cwd) / "config.log").read_text(errors="replace")[-12000:], file=sys.stderr)
        raise SystemExit(result.returncode)


def main():
    name, destination, cmake = sys.argv[1:4]
    destination = Path(destination).absolute()
    cmake = str(Path(cmake).resolve())
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(LC_ALL="C", PYTHONDONTWRITEBYTECODE="1", ZERO_AR_DATE="1")
    host_sdk = capture("xcrun", "--sdk", "macosx", "--show-sdk-path")
    clang = capture("xcrun", "--find", "clang")
    clangxx = capture("xcrun", "--find", "clang++")
    with tempfile.TemporaryDirectory(prefix="ffmpeg-codec-") as temporary:
        work = Path(temporary)
        if name == "tools":
            env.update(SDKROOT=host_sdk, CC=clang, CXX=clangxx,
                       CFLAGS=f"-isysroot {host_sdk}", CXXFLAGS=f"-isysroot {host_sdk}",
                       LDFLAGS=f"-isysroot {host_sdk}")
            for tool, source in zip(sys.argv[4::2], sys.argv[5::2]):
                src = work / tool
                shutil.copytree(source, src, symlinks=False)
                if tool == "ninja":
                    run([sys.executable, "configure.py", "--bootstrap"], src, env)
                    (destination / "bin").mkdir(exist_ok=True)
                    shutil.copy2(src / "ninja", destination / "bin/ninja")
                else:
                    run(["./configure", "--disable-shared", "--disable-dependency-tracking",
                         f"--prefix={destination}"], src, env)
                    run(["make", "-j4"], src, env)
                    run(["make", "install"], src, env)
            return

        sdk, arch, minimum, source, tools, meson, sanitizer = sys.argv[4:11]
        tools = Path(tools).absolute()
        meson = Path(meson).absolute() / "meson.py"
        env["PATH"] = str(tools / "bin") + os.pathsep + env.get("PATH", "/usr/bin:/bin")
        sdkroot = capture("xcrun", "--sdk", sdk, "--show-sdk-path")
        target = f"{arch}-apple-" + (f"macos{minimum}" if sdk == "macosx" else f"ios{minimum}")
        if sdk == "iphonesimulator":
            target += "-simulator"
        flags = ["-target", target, "-isysroot", sdkroot, "-O2", "-fPIC", "-fvisibility=hidden"]
        configure_object = work / "codec_configure.o"
        run([clang] + flags + ["-c", str(Path(sys.argv[12]).absolute()), "-o", str(configure_object)], work, env)
        flags += ["-include", str(Path(sys.argv[11]).absolute())]
        if sanitizer == "asan":
            # CMake also passes CFLAGS at link time. SVT treats unused driver
            # arguments as errors in feature probes; this LLVM option is only
            # consumed when compiling, not when linking those probes.
            flags += ["-fsanitize=address", "-fno-omit-frame-pointer",
                      "--start-no-unused-arguments", "-mllvm", "-asan-globals-live-support=0",
                      "--end-no-unused-arguments"]
        env.update(SDKROOT=sdkroot, CC=clang, CXX=clangxx, CFLAGS=" ".join(flags),
                   CXXFLAGS=" ".join(flags), LDFLAGS=" ".join(flags + [str(configure_object)]), ASFLAGS=" ".join(flags),
                   PKG_CONFIG=str(tools / "bin/pkgconf"), PKG_CONFIG_LIBDIR=str(destination / "lib/pkgconfig"),
                   PKG_CONFIG_PATH="")
        src = work / "source"
        shutil.copytree(source, src, symlinks=False)
        build = work / "build"
        build.mkdir()
        cmake_args = [cmake, "-G", "Unix Makefiles", "-DCMAKE_BUILD_TYPE=Release",
                      "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", f"-DCMAKE_INSTALL_PREFIX={destination}",
                      "-DCMAKE_INSTALL_LIBDIR=lib", f"-DCMAKE_C_COMPILER={clang}",
                      f"-DCMAKE_CXX_COMPILER={clangxx}", f"-DCMAKE_OSX_SYSROOT={sdkroot}",
                      f"-DCMAKE_OSX_ARCHITECTURES={arch}", f"-DCMAKE_OSX_DEPLOYMENT_TARGET={minimum}",
                      "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", "-DBUILD_SHARED_LIBS=OFF",
                      f"-DCMAKE_ASM_FLAGS={' '.join(flags)}",
                      "-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG", "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG"]
        if sdk != "macosx":
            cmake_args += ["-DCMAKE_SYSTEM_NAME=iOS", "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"]

        if name in ("svt-av1", "opus", "x265"):
            if name == "svt-av1":
                extra = ["-DBUILD_APPS=OFF", "-DBUILD_TESTING=OFF", "-DSVT_AV1_LTO=OFF",
                         "-DEXCLUDE_HASH=ON", "-DENABLE_SVE=OFF", "-DENABLE_SVE2=OFF"]
            elif name == "opus":
                extra = ["-DOPUS_BUILD_TESTING=OFF", "-DOPUS_BUILD_PROGRAMS=OFF"]
            else:
                src = src / "source"
                extra = ["-DENABLE_SHARED=OFF", "-DENABLE_CLI=OFF", "-DENABLE_LIBNUMA=OFF",
                         "-DENABLE_PIC=ON", "-DENABLE_SVE=OFF", "-DENABLE_SVE2=OFF",
                         "-DENABLE_SVE2_BITPERM=OFF"]
                if arch == "arm64":
                    extra += [f"-DASM_FLAGS=-target;{target};-isysroot;{sdkroot}"]
                # Match Homebrew's 8/10/12-bit API in a single static archive.
                for bits in (10, 12):
                    high = work / str(bits)
                    run(cmake_args + ["-S", str(src), "-B", str(high)] + extra +
                        ["-DHIGH_BIT_DEPTH=ON", "-DEXPORT_C_API=OFF", f"-DMAIN12={'ON' if bits == 12 else 'OFF'}"], work, env)
                    run([cmake, "--build", str(high), "-j4"], work, env)
                    shutil.copy2(high / "libx265.a", build / f"libx265_main{bits}.a")
                extra += ["-DLINKED_10BIT=ON", "-DLINKED_12BIT=ON",
                          "-DEXTRA_LIB=x265_main10.a;x265_main12.a", f"-DEXTRA_LINK_FLAGS=-L{build}"]
            run(cmake_args + ["-S", str(src), "-B", str(build)] + extra, work, env)
            run([cmake, "--build", str(build), "-j4"], work, env)
            if name == "x265":
                shutil.move(build / "libx265.a", build / "libx265_main.a")
                run(["xcrun", "libtool", "-static", "-o", str(build / "libx265.a"),
                     str(build / "libx265_main.a"), str(build / "libx265_main10.a"),
                     str(build / "libx265_main12.a")], work, env)
            run([cmake, "--install", str(build)], work, env)
        elif name in ("dav1d", "libvmaf"):
            cross = work / "cross.ini"
            cross.write_text("[binaries]\nc = " + repr(clang) + "\ncpp = " + repr(clangxx) +
                             "\nar = 'ar'\nstrip = 'strip'\npkg-config = " + repr(str(tools / "bin/pkgconf")) +
                             "\n[host_machine]\nsystem = 'darwin'\ncpu_family = " + repr("aarch64" if arch == "arm64" else arch) +
                             "\ncpu = " + repr(arch) + "\nendian = 'little'\n[built-in options]\nc_args = " + repr(flags) +
                             "\nc_link_args = " + repr(flags + [str(configure_object)]) + "\n[properties]\nneeds_exe_wrapper = true\n")
            meson_source = src if name == "dav1d" else src / "libvmaf"
            run([sys.executable, str(meson), "setup", str(build), str(meson_source),
                 "--cross-file", str(cross), "--prefix", str(destination), "--libdir=lib",
                 "--default-library=static", "--buildtype=release", "-Doptimization=2",
                 "-Denable_tools=false", "-Denable_tests=false", "-Denable_docs=false"], work, env)
            run([str(tools / "bin/ninja"), "-C", str(build), "-j4", "install"], work, env)
        elif name == "lame":
            host_arch = "aarch64" if arch == "arm64" else arch
            # Autoconf's char function() probes conflict with declarations from
            # the forced host header. Apply redirection only to the real build.
            probe_flags = " ".join(flags[:flags.index("-include")])
            lame_env = dict(env, CFLAGS=probe_flags, CXXFLAGS=probe_flags, LDFLAGS=probe_flags)
            run(["./configure", f"--prefix={destination}", f"--host={host_arch}-apple-darwin",
                 "--disable-shared", "--enable-static", "--disable-frontend", "--disable-decoder",
                 "--disable-dependency-tracking"], src, lame_env)
            run(["make", "-j4", "CFLAGS=" + " ".join(flags)], src, env)
            run(["make", "install"], src, env)
        elif name == "x264":
            # x264 appends --extra-ldflags to the environment's LDFLAGS.
            x264_env = dict(env, LDFLAGS="")
            run(["./configure", f"--prefix={destination}", f"--host={arch}-apple-darwin",
                 "--enable-static", "--enable-pic", "--disable-cli", "--disable-opencl",
                 f"--extra-cflags={' '.join(flags)}", f"--extra-ldflags={' '.join(flags + [str(configure_object)])}"], src, x264_env)
            run(["make", "-j4"], src, x264_env)
            run(["make", "install-lib-static"], src, x264_env)
        elif name == "libvpx":
            vpx_target = f"{arch}-" + ("darwin-gcc" if sdk == "iphoneos" else
                          "iphonesimulator-gcc" if sdk == "iphonesimulator" else "darwin25-gcc")
            run(["./configure", f"--prefix={destination}", f"--target={vpx_target}",
                 "--disable-shared", "--enable-static", "--enable-pic", "--disable-examples",
                 "--disable-tools", "--disable-docs", "--disable-unit-tests", "--enable-vp9-highbitdepth"], src, env)
            run(["make", "-j4"], src, env)
            run(["make", "install"], src, env)
        else:
            raise ValueError(name)

        # Prefixes change between Bazel execroots. Resolve pkg-config paths at
        # consumption time using pcfiledir, never temporary absolute paths.
        for pc in destination.glob("lib/pkgconfig/*.pc"):
            pc.write_text(pc.read_text().replace(str(destination), "${pcfiledir}/../.."))
        notices = destination / "licenses"
        notices.mkdir(exist_ok=True)
        for file in src.iterdir():
            if file.is_file() and file.name.upper().startswith(("LICENSE", "COPYING", "PATENTS")):
                shutil.copy2(file, notices / file.name)
        (destination / "build.json").write_text(json.dumps(dict(codec=name, sdk=sdk, arch=arch,
            minimum_os=minimum, sanitizer=sanitizer), indent=2) + "\n")


if __name__ == "__main__":
    main()
