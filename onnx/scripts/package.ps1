<#
.SYNOPSIS
  Build a portable DCVC-SDK package on Windows (native MSVC).

.DESCRIPTION
  Configures (Visual Studio, x64), builds the libdcvc shared library + demo,
  stages headers/lib/ORT/CMake config/demo/examples/docs into a relocatable
  folder, discovers resolution model packs, and zips it.

  Output: out\packages\dcvc-sdk-<version>-windows-x64-msvc.zip

.PARAMETER ModelPack
  One or more name=directory model packs. When omitted, models_<resolution>
  directories such as models_720p and models_1080p are discovered.

.PARAMETER OutDir
  Output directory for the archive (default: out\packages).

.PARAMETER BuildDir
  Build directory (default: out\build\windows-x64-msvc).

.PARAMETER KeepStaging
  Keep the expanded staging tree after the zip is created.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\package.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\package.ps1 `
    -ModelPack "720p=C:\models\720p","1080p=C:\models\1080p"
#>
[CmdletBinding()]
param(
    [string[]]$ModelPack = @(),
    [switch]$NoModels,
    [switch]$KeepStaging,
    [string]$OutDir = "",
    [string]$BuildDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OnnxDir   = Split-Path -Parent $ScriptDir

$ModelPacks = @()
if (-not $NoModels) {
    if ($ModelPack.Count -eq 0) {
        Get-ChildItem -Path $OnnxDir -Directory -Filter "models_*" |
            Where-Object Name -Match '^models_[0-9]+p$' |
            Sort-Object { [int]($_.Name -replace '^models_([0-9]+)p$', '$1') } |
            ForEach-Object {
                $_.Name -match '^models_([0-9]+p)$' | Out-Null
                $ModelPacks += [PSCustomObject]@{ Name = $Matches[1]; Path = $_.FullName }
            }
    } else {
        foreach ($spec in $ModelPack) {
            $separator = $spec.IndexOf('=')
            if ($separator -le 0) { throw "invalid -ModelPack '$spec' (expected name=directory)" }
            $name = $spec.Substring(0, $separator)
            $path = $spec.Substring($separator + 1)
            if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "invalid model pack name '$name'" }
            $ModelPacks += [PSCustomObject]@{ Name = $name; Path = $path }
        }
    }
    if ($ModelPacks.Count -eq 0) {
        throw "no model packs found (expected models_<resolution>, or use -ModelPack name=directory)"
    }
    $duplicates = $ModelPacks | Group-Object Name | Where-Object Count -gt 1
    if ($duplicates) { throw "duplicate model pack name '$($duplicates[0].Name)'" }
    foreach ($pack in $ModelPacks) {
        if (-not (Test-Path $pack.Path -PathType Container)) { throw "model dir '$($pack.Path)' not found" }
        if (-not (Test-Path (Join-Path $pack.Path "intra_analysis_standard.onnx") -PathType Leaf)) {
            throw "model pack '$($pack.Name)' is missing intra_analysis_standard.onnx"
        }
    }
}

$Version   = "1.0.0"
$Platform  = if ($env:DCVC_ORT_GPU) { "windows-x64-msvc-gpu" } else { "windows-x64-msvc" }
$PkgName   = "dcvc-sdk-$Version-$Platform"
$OutputRoot = if ($env:DCVC_OUTPUT_ROOT) { $env:DCVC_OUTPUT_ROOT } else { Join-Path $OnnxDir "out" }
if ([string]::IsNullOrEmpty($BuildDir)) { $BuildDir = Join-Path $OutputRoot "build\$Platform" }
if ([string]::IsNullOrEmpty($OutDir))   { $OutDir   = Join-Path $OutputRoot "packages" }
$Staging   = Join-Path $OutputRoot "staging\$PkgName"

Write-Host "==========================================================="
Write-Host " DCVC-SDK package builder (Windows native, MSVC)"
Write-Host "   version : $Version"
Write-Host "   build   : $BuildDir"
Write-Host "   staging : $Staging"
$modelSummary = if ($NoModels) { "<disabled>" } else {
    (($ModelPacks | ForEach-Object { "$($_.Name)=$($_.Path)" }) -join "; ")
}
Write-Host "   models  : $modelSummary"
Write-Host "   out     : $OutDir\$PkgName.zip"
Write-Host "==========================================================="

# ---- configure ------------------------------------------------------------
$cmakeArgs = @(
    "-S", $OnnxDir, "-B", $BuildDir, "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DDCVC_BUILD_SDK=ON",
    "-DDCVC_FXP_CUDA=OFF",
    "-DDCVC_OUTPUT_ROOT=$OutputRoot",
    "-DDCVC_ARTIFACT_TAG=$Platform"
)
if ($env:DCVC_ORT_GPU)    { $cmakeArgs += "-DDCVC_ORT_GPU=ON" }
if ($env:DCVC_ORT_ARCHIVE){ $cmakeArgs += "-DDCVC_ORT_ARCHIVE=$env:DCVC_ORT_ARCHIVE" }
if ($env:DCVC_ORT_URL_BASE){ $cmakeArgs += "-DDCVC_ORT_URL_BASE=$env:DCVC_ORT_URL_BASE" }

Write-Host "== Configuring =="
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

# ---- build ----------------------------------------------------------------
Write-Host "== Building (dcvc + dcvc_demo) =="
& cmake --build $BuildDir --target dcvc dcvc_demo --config Release -- -m
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# ---- stage ----------------------------------------------------------------
Write-Host "== Staging SDK into $Staging =="
if (Test-Path $Staging) { Remove-Item -Recurse -Force $Staging }
& cmake --install $BuildDir --prefix $Staging --config Release | Out-Null
if ($LASTEXITCODE -ne 0) { throw "install failed" }

# ---- locate ORT runtime ---------------------------------------------------
$OrtRoot = Get-ChildItem -Path $BuildDir -Directory -Filter "onnxruntime-*" | Select-Object -First 1
if (-not $OrtRoot) { throw "ONNX Runtime not found under $BuildDir" }
$OrtLib = Join-Path $OrtRoot.FullName "lib"

$StageLib = Join-Path $Staging "lib"
$StageBin = Join-Path $Staging "bin"
New-Item -ItemType Directory -Force -Path $StageBin | Out-Null

# ---- post-install copies --------------------------------------------------
Write-Host "== Copying runtime + demo + examples + docs =="
Copy-Item -Force (Join-Path $BuildDir "Release\dcvc_demo.exe") $StageBin
Get-ChildItem -Path $OrtLib -Filter "onnxruntime*.dll" | ForEach-Object { Copy-Item -Force $_.FullName $StageBin }
$Shared = Join-Path $OrtLib "onnxruntime_providers_shared.dll"
if (Test-Path $Shared) { Copy-Item -Force $Shared $StageBin }

New-Item -ItemType Directory -Force -Path (Join-Path $Staging "examples") | Out-Null
Copy-Item -Force (Join-Path $OnnxDir "examples\dcvc_demo.c") (Join-Path $Staging "examples")
$Qs = Join-Path $OnnxDir "docs\sdk_quickstart.md"
if (Test-Path $Qs) { Copy-Item -Force $Qs $Staging }

# ---- models ---------------------------------------------------------------
if (-not $NoModels) {
    Write-Host "== Bundling $($ModelPacks.Count) model pack(s) =="
    foreach ($pack in $ModelPacks) {
        $dest = Join-Path $Staging "models\$($pack.Name)"
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Copy-Item -Path (Join-Path $pack.Path "*") -Destination $dest -Recurse -Force
    }
}

# ---- top-level README -----------------------------------------------------
$hasModels = Test-Path (Join-Path $Staging "models")
$modelNames = ($ModelPacks | ForEach-Object Name) -join ", "
$quickTests = @()
if ($ModelPacks.Name -contains "720p")  { $quickTests += "  bin\dcvc_demo.exe models\720p 1280 768 32 3" }
if ($ModelPacks.Name -contains "1080p") { $quickTests += "  bin\dcvc_demo.exe models\1080p 1920 1088 32 3" }
@"
DCVC-SDK $Version ($Platform)
============================

Portable neural video codec. Cross-platform bit-exact by construction
(FXP fixed-point nets + integer scale->CDF index + rANS).

Contents:
  include\dcvc\      public C headers
  bin\               executables + dcvc.dll + ONNX Runtime DLLs
  lib\               dcvc import library
  lib\cmake\dcvc\    find_package(dcvc) support
  bin\dcvc_demo.exe  example encoder/decoder
  examples\          dcvc_demo.c source
$(if ($hasModels) { "  models\<name>\     model packs: $modelNames`n" })
Quick test:
$($quickTests -join "`n")

Integrate (CMake):
  find_package(dcvc $Version CONFIG REQUIRED)
  target_link_libraries(myapp PRIVATE dcvc::dcvc)

See sdk_quickstart.md for the full walkthrough.
"@ | Set-Content -Encoding UTF8 (Join-Path $Staging "README.txt")

# ---- sanity check ---------------------------------------------------------
Write-Host "== Sanity check =="
$required = @(
    "include\dcvc\dcvc.h",
    "bin\dcvc.dll",
    "lib\dcvc.lib",
    "bin\onnxruntime.dll",
    "lib\cmake\dcvc\dcvcTargets.cmake",
    "bin\dcvc_demo.exe"
)
foreach ($f in $required) {
    $p = Join-Path $Staging $f
    if (-not (Test-Path $p)) { Write-Host "  MISSING: $f"; $script:bad = $true }
}
if (-not $NoModels) {
    foreach ($pack in $ModelPacks) {
        $f = "models\$($pack.Name)\intra_analysis_standard.onnx"
        if (-not (Test-Path (Join-Path $Staging $f))) { Write-Host "  MISSING: $f"; $script:bad = $true }
    }
}
if ($script:bad) { throw "package incomplete" }
Write-Host "  all expected files present"

# ---- archive --------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Archive = Join-Path $OutDir "$PkgName.zip"
Write-Host "== Archiving -> $Archive =="
if (Test-Path $Archive) { Remove-Item -Force $Archive }
Compress-Archive -Path $Staging -DestinationPath $Archive

$size = (Get-Item $Archive).Length / 1MB
Write-Host ""
Write-Host "Done."
Write-Host ("  package : $Archive")
Write-Host ("  size    : {0:N1} MB" -f $size)
if ($KeepStaging) {
    Write-Host "  staging : $Staging"
} else {
    Remove-Item -Recurse -Force $Staging
    Write-Host "  staging : removed (use -KeepStaging to retain it)"
}
