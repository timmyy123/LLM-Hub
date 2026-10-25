<#
.SYNOPSIS
Builds the LLM Hub installers for Windows x64 and Arm64:
  LLMHub-<version>-win-<arch>-setup.exe   per-user installer (Inno Setup) for direct download, no admin rights needed
  LLMHub-<version>-win-<arch>.msix        Microsoft Store package (unsigned: Partner Center signs Store packages)
  LLMHub-<version>-win.msixbundle         both architectures in one Store upload

.EXAMPLE
desktop\tools\package\windows\package-windows.ps1
desktop\tools\package\windows\package-windows.ps1 -Arch arm64 -Kind setup -SdCli C:\out\sdcpp-arm64\sd-cli.exe
desktop\tools\package\windows\package-windows.ps1 -Kind msix -StoreIdentityName 12345Publisher.LLMHub -StorePublisher "CN=..." -StorePublisherDisplayName "..."

.NOTES
Needs the .NET 10 SDK, the Windows 11 SDK (makeappx, signtool), Visual Studio's C++ build tools for x64 and ARM64 (the
Visual C++ runtime files that ship next to LLMHub.exe come from its VC\Redist folder) and Inno Setup 6.3 or later
(winget install JRSoftware.InnoSetup). Runs on an x64 PC; Arm64 is cross-compiled.

Code signing (optional; unsigned installers work but SmartScreen warns until the publisher has reputation). Set one of:
  LLMHUB_SIGN_CERT (path to a .pfx) and LLMHUB_SIGN_PASSWORD
  LLMHUB_SIGN_THUMBPRINT (SHA-1 of a certificate in the current user's store, for example on a hardware token)
  LLMHUB_SIGN_COMMAND (any signing command, with {file} where the file goes, for example Azure Trusted Signing's
                       signtool /dlib command line)
LLMHUB_SIGN_TIMESTAMP overrides the RFC 3161 timestamp server. LLMHub.exe, the installer and its uninstaller are signed.
MSIX packages stay unsigned for the Store; -SignMsix signs them too (for sideloading: the certificate subject must then
equal the manifest Publisher, so pass -StorePublisher).
#>
param(
    [ValidateSet("x64", "arm64", "all")][string]$Arch = "all",
    [ValidateSet("setup", "msix", "all")][string]$Kind = "all",
    [string]$Out = "",
    [string]$Work = "",
    # Arm64: Studio's NPU video engine, built by desktop\windows\sdcpp\build.ps1. Without it Studio hides that workflow.
    [string]$SdCli = "",
    [string]$StoreIdentityName = "",
    [string]$StorePublisher = "",
    [string]$StorePublisherDisplayName = "",
    [switch]$SignMsix
)
$ErrorActionPreference = "Stop"

$Desktop = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Project = Join-Path $Desktop "windows\LLMHub\LLMHub.csproj"
if (-not $Out) { $Out = Join-Path $Desktop "tools\package\out" }
if (-not $Work) { $Work = Join-Path $Desktop "tools\package\work\windows" }
New-Item -ItemType Directory -Force $Out, $Work | Out-Null
$Out = (Resolve-Path $Out).Path
$Work = (Resolve-Path $Work).Path

$versionFile = Get-Content (Join-Path $Desktop "version.xcconfig") -Raw
$Version = [regex]::Match($versionFile, 'MARKETING_VERSION\s*=\s*(\d+\.\d+\.\d+)').Groups[1].Value
$Build = [regex]::Match($versionFile, 'CURRENT_PROJECT_VERSION\s*=\s*(\d+)').Groups[1].Value
if (-not $Version -or -not $Build) { throw "version.xcconfig needs MARKETING_VERSION and CURRENT_PROJECT_VERSION" }
$Arches = if ($Arch -eq "all") { @("x64", "arm64") } else { @($Arch) }
"LLM Hub $Version (build $Build): $($Arches -join ', '), $Kind"

# --- Tools -------------------------------------------------------------------------------------------------------------

function Find-SdkTool([string]$name) {
    $kits = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
    $tool = Get-ChildItem $kits -Directory -Filter "10.*" -ErrorAction SilentlyContinue | Sort-Object { [version]$_.Name } -Descending |
        ForEach-Object { Join-Path $_.FullName "x64\$name" } | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $tool) { throw "$name not found: install the Windows 11 SDK" }
    $tool
}

function Find-Iscc {
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"))
    $found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $found) { $found = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source }
    if (-not $found) { throw "Inno Setup 6 not found: winget install JRSoftware.InnoSetup" }
    $found
}

# Visual Studio's redistributable Visual C++ runtime for one architecture (VC\Redist\MSVC\<version>\<arch>\Microsoft.VC*.CRT).
function Find-VcRuntime([string]$arch) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $roots = @()
    if (Test-Path $vswhere) { $roots = & $vswhere -products * -property installationPath }
    foreach ($root in $roots) {
        $redist = Join-Path $root "VC\Redist\MSVC"
        $dir = Get-ChildItem $redist -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
            Sort-Object { [version]$_.Name } -Descending | ForEach-Object {
                Get-ChildItem (Join-Path $_.FullName $arch) -Directory -Filter "Microsoft.VC*.CRT" -ErrorAction SilentlyContinue
            } | Select-Object -First 1
        if ($dir) { return $dir.FullName }
    }
    throw "No Visual C++ runtime for $arch in Visual Studio's VC\Redist. Add 'MSVC $arch build tools' in the Visual Studio Installer."
}

# --- Signing hook ------------------------------------------------------------------------------------------------------

$Timestamp = if ($env:LLMHUB_SIGN_TIMESTAMP) { $env:LLMHUB_SIGN_TIMESTAMP } else { "http://timestamp.digicert.com" }
$SignTool = $null
$SignArgs = $null       # signtool arguments before the file name
$IsccSign = $null       # the same command for Inno Setup's /S option ($q is a quote, $f the quoted file name)
if ($env:LLMHUB_SIGN_COMMAND) {
    $IsccSign = $env:LLMHUB_SIGN_COMMAND.Replace('"{file}"', '$f').Replace('{file}', '$f').Replace('"', '$q')
} elseif ($env:LLMHUB_SIGN_CERT -or $env:LLMHUB_SIGN_THUMBPRINT) {
    $SignTool = Find-SdkTool "signtool.exe"
    $SignArgs = @("/fd", "SHA256", "/tr", $Timestamp, "/td", "SHA256")
    $SignArgs += if ($env:LLMHUB_SIGN_CERT) { @("/f", $env:LLMHUB_SIGN_CERT, "/p", $env:LLMHUB_SIGN_PASSWORD) } else { @("/sha1", $env:LLMHUB_SIGN_THUMBPRINT) }
    $IsccSign = ('$q' + $SignTool + '$q sign ' + (($SignArgs | ForEach-Object { '$q' + $_ + '$q' }) -join " ") + ' $f')
}
if ($IsccSign) { "Signing: on" } else { "Signing: off (set LLMHUB_SIGN_CERT, LLMHUB_SIGN_THUMBPRINT or LLMHUB_SIGN_COMMAND)" }

function Invoke-Sign([string[]]$files) {
    if (-not $IsccSign) { return }
    foreach ($file in $files) {
        # Timestamp servers fail now and then, so try a few times (Inno Setup retries its own signing too).
        for ($attempt = 1; $attempt -le 4; $attempt++) {
            if ($env:LLMHUB_SIGN_COMMAND) {
                $cmdFile = Join-Path $Work "sign.cmd"
                Set-Content $cmdFile ("@" + $env:LLMHUB_SIGN_COMMAND.Replace('"{file}"', '{file}').Replace("{file}", "`"$file`"")) -Encoding ASCII
                & cmd.exe /c $cmdFile
                Remove-Item $cmdFile
            } else {
                & $SignTool sign /q @SignArgs $file
            }
            if ($LASTEXITCODE -eq 0) { break }
            if ($attempt -eq 4) { throw "signing failed: $file" }
            Start-Sleep -Seconds (5 * $attempt)
        }
    }
}

# --- Build helpers -----------------------------------------------------------------------------------------------------

function Invoke-Logged([string]$log, [string]$exe, [string[]]$arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"   # dotnet and ISCC report progress on stderr
    & $exe @arguments *> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) {
        Get-Content $log | Select-String -Pattern "error|Error" | Select-Object -First 20 | ForEach-Object { $_.Line.Trim() }
        throw "$([IO.Path]::GetFileName($exe)) failed (exit $code), see $log"
    }
}

# A fresh obj/bin for each build: packaged and unpackaged builds of the same platform share these folders.
function Clear-ProjectOutput([string]$platform) {
    $projectDir = Split-Path $Project
    foreach ($dir in @("obj\$platform\Release", "bin\$platform\Release")) {
        $path = Join-Path $projectDir $dir
        if (Test-Path $path) { Remove-Item $path -Recurse -Force }
    }
}

$SdCliTarget = Join-Path $Desktop "windows\sdcpp\out\sd-cli.exe"
if ($SdCli -and (Resolve-Path $SdCli).Path -ne $SdCliTarget) {
    New-Item -ItemType Directory -Force (Split-Path $SdCliTarget) | Out-Null
    Copy-Item $SdCli $SdCliTarget -Force
}

$msixFiles = @()
foreach ($a in $Arches) {
    $platform = if ($a -eq "arm64") { "ARM64" } else { "x64" }
    $rid = "win-$a"
    $vcRuntime = Find-VcRuntime $a
    if ($a -eq "arm64" -and -not (Test-Path $SdCliTarget)) {
        "warning: no $SdCliTarget (desktop\windows\sdcpp\build.ps1 or -SdCli): Studio's NPU video will be hidden"
    }
    $common = @("-c", "Release", "-p:Platform=$platform", "-r", $rid, "--self-contained", "-p:LLMHubVcRuntimeDir=$vcRuntime", "-nologo", "-v:q")

    if ($Kind -ne "msix") {
        "[$a] publishing"
        $publish = Join-Path $Work "publish-$a"
        if (Test-Path $publish) { Remove-Item $publish -Recurse -Force }
        Clear-ProjectOutput $platform
        Invoke-Logged (Join-Path $Work "publish-$a.log") "dotnet" (@("publish", $Project, "-o", $publish) + $common)
        Invoke-Sign @((Join-Path $publish "LLMHub.exe"), (Join-Path $publish "LLMHub.dll"), (Join-Path $publish "LLMHub.Core.dll"))

        "[$a] building the installer"
        $iscc = Find-Iscc
        $isccArgs = @("/Q", "/DAppVersion=$Version", "/DFileVersion=$Version.$Build", "/DArch=$a", "/DSourceDir=$publish",
            "/DOutputDir=$Out", "/DDesktopDir=$Desktop")
        if ($IsccSign) { $isccArgs += @("/DSign", "/Sllmhub=$IsccSign") }
        $isccArgs += (Join-Path $PSScriptRoot "LLMHub.iss")
        Invoke-Logged (Join-Path $Work "iscc-$a.log") $iscc $isccArgs
        $setup = Join-Path $Out "LLMHub-$Version-win-$a-setup.exe"
        "  {0} ({1:N0} MB)" -f $setup, ((Get-Item $setup).Length / 1MB)
    }

    if ($Kind -ne "setup") {
        "[$a] building the MSIX"
        $packageDir = Join-Path $Work "msix-$a"
        if (Test-Path $packageDir) { Remove-Item $packageDir -Recurse -Force }
        Clear-ProjectOutput $platform
        $msixArgs = @("build", $Project, "-p:WindowsPackageType=MSIX", "-p:GenerateAppxPackageOnBuild=true",
            "-p:AppxPackageSigningEnabled=false", "-p:AppxBundle=Never", "-p:UapAppxPackageBuildMode=SideloadOnly",
            "-p:AppxPackageDir=$packageDir/")
        foreach ($pair in @(@("StoreIdentityName", $StoreIdentityName), @("StorePublisher", $StorePublisher), @("StorePublisherDisplayName", $StorePublisherDisplayName))) {
            # MSBuild splits -p values on ; and , so escape them.
            if ($pair[1]) { $msixArgs += "-p:$($pair[0])=$($pair[1].Replace('%', '%25').Replace(';', '%3B').Replace(',', '%2C'))" }
        }
        Invoke-Logged (Join-Path $Work "msix-$a.log") "dotnet" ($msixArgs + $common)
        $built = Get-ChildItem $packageDir -Recurse -Filter *.msix | Select-Object -First 1
        if (-not $built) { throw "no .msix in $packageDir" }
        $msix = Join-Path $Out "LLMHub-$Version-win-$a.msix"
        Copy-Item $built.FullName $msix -Force
        if ($SignMsix) { Invoke-Sign @($msix) }
        $msixFiles += $msix
        "  {0} ({1:N0} MB)" -f $msix, ((Get-Item $msix).Length / 1MB)
        # The next unpackaged build must not reuse packaged outputs.
        Clear-ProjectOutput $platform
    }
}

# One bundle with both architectures for the Store.
if ($msixFiles.Count -eq 2) {
    "bundling"
    $makeappx = Find-SdkTool "makeappx.exe"
    $bundleIn = Join-Path $Work "bundle"
    if (Test-Path $bundleIn) { Remove-Item $bundleIn -Recurse -Force }
    New-Item -ItemType Directory $bundleIn | Out-Null
    $msixFiles | ForEach-Object { Copy-Item $_ $bundleIn }
    $bundle = Join-Path $Out "LLMHub-$Version-win.msixbundle"
    Invoke-Logged (Join-Path $Work "bundle.log") $makeappx @("bundle", "/d", $bundleIn, "/p", $bundle, "/bv", "$Version.0", "/o")
    if ($SignMsix) { Invoke-Sign @($bundle) }
    "  {0} ({1:N0} MB)" -f $bundle, ((Get-Item $bundle).Length / 1MB)
}
"done: $Out"
