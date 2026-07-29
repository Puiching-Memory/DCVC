# Build the pure-CPU DCVC-RT I-frame codec on Windows.
# Produces build intermediates and a runnable folder under out\.
#
# Requires: CMake, Visual Studio 2019/2022 with "Desktop development with C++",
#           and Python (with torch/onnx/numpy) to regenerate models.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OnnxDir   = Split-Path -Parent $ScriptDir
$OutputRoot = if ($env:DCVC_OUTPUT_ROOT) { $env:DCVC_OUTPUT_ROOT } else { Join-Path $OnnxDir "out" }
$ArtifactTag = "windows-x64-msvc"
$BuildDir  = Join-Path $OutputRoot "build\$ArtifactTag"
$RunnableDir = Join-Path $OutputRoot "runnable\$ArtifactTag"

Write-Host "== Configuring (Windows, x64) =="
cmake -S $OnnxDir -B $BuildDir -A x64 `
    "-DDCVC_FXP_CUDA=OFF" `
    "-DDCVC_OUTPUT_ROOT=$OutputRoot" `
    "-DDCVC_ARTIFACT_TAG=$ArtifactTag"

Write-Host "== Building (Release) =="
cmake --build $BuildDir --config Release -- -m

Write-Host "== Packaging into $RunnableDir =="
cmake --build $BuildDir --config Release --target dcvc_package

Write-Host ""
Write-Host "Done. Runnable folder: $RunnableDir\"
Write-Host "Build directory:       $BuildDir\"
