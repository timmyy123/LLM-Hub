<#
Builds sd-cli.exe (stable-diffusion.cpp with LTX-Video on the Hexagon NPU) for Windows on Arm, cross-compiled on an
x64 Windows PC with llvm-mingw (https://github.com/mstorsjo/llvm-mingw, ucrt x86_64 host build) and Ninja.

  .\build.ps1 -Toolchain C:\tools\llvm-mingw [-Out C:\out\sdcpp-arm64]

The result is one static executable; Studio runs it next to the app's QnnHtp.dll / QnnSystem.dll.
#>
param(
    [Parameter(Mandatory = $true)][string]$Toolchain,
    [string]$Out = (Join-Path $PSScriptRoot "out"),
    [string]$Ninja = ""
)
$ErrorActionPreference = "Stop"
$build = Join-Path $Out "build"
New-Item -ItemType Directory -Force $build | Out-Null
if (-not $Ninja) {
    $Ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
    if (-not $Ninja) {
        $Ninja = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    }
}
$bin = (Join-Path $Toolchain "bin").Replace('\', '/')
$Ninja = $Ninja.Replace('\', '/')
$env:PATH = "$bin;$env:PATH"
$ErrorActionPreference = "Continue" # cmake reports progress on stderr
cmake -S $PSScriptRoot -B $build -G Ninja "-DCMAKE_MAKE_PROGRAM=$Ninja" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_SYSTEM_PROCESSOR=ARM64 `
    "-DCMAKE_C_COMPILER=$bin/aarch64-w64-mingw32-clang.exe" `
    "-DCMAKE_CXX_COMPILER=$bin/aarch64-w64-mingw32-clang++.exe" `
    "-DCMAKE_RC_COMPILER=$bin/aarch64-w64-mingw32-windres.exe"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --build $build --target sd-cli -j ([Environment]::ProcessorCount)
if ($LASTEXITCODE -ne 0) { throw "build failed" }
$exe = Get-ChildItem $build -Recurse -Filter sd-cli.exe | Select-Object -First 1
Copy-Item $exe.FullName (Join-Path $Out "sd-cli.exe") -Force
"built " + (Join-Path $Out "sd-cli.exe")
