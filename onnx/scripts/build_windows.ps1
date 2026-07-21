# Build the pure-CPU DCVC-RT I-frame codec on Windows.
# Produces test executables under build\Release and a ready-to-run package under dist\.
#
# Requires: CMake, Visual Studio 2019/2022 with "Desktop development with C++",
#           and Python (with torch/onnx/numpy) to regenerate models.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OnnxDir   = Split-Path -Parent $ScriptDir
$BuildDir  = Join-Path $OnnxDir "build"
$DistDir   = Join-Path $OnnxDir "dist"

Write-Host "== Configuring (Windows, x64) =="
cmake -S $OnnxDir -B $BuildDir -A x64

Write-Host "== Building (Release) =="
cmake --build $BuildDir --config Release -- -m

Write-Host "== Packaging into $DistDir =="
cmake --build $BuildDir --config Release --target dcvc_package

Write-Host ""
Write-Host "Done. Package: $DistDir\dcvc_onnx_codec\"
Write-Host "Quick test:    $BuildDir\Release\test_intra_analysis.exe ..\intra_analysis_standard.onnx"
