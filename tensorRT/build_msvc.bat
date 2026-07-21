@echo off
REM Build tensorRT/ without CMake using MSVC (run from "x64 Native Tools Command Prompt")
set ROOT=%~dp0
set INC=/I"%ROOT%include" /I"%ROOT%rans" /I"%ROOT%plugins\include"
set CFLAGS=/nologo /O2 /std:c11 /W3 %INC%
set CXXFLAGS=/nologo /O2 /std:c++17 /EHsc /W3 %INC%

if not exist "%ROOT%build_msvc" mkdir "%ROOT%build_msvc"
cd /d "%ROOT%build_msvc"

cl %CXXFLAGS% /c "%ROOT%rans\rans.cpp" "%ROOT%rans\rans_c.cpp" "%ROOT%rans\pmf_cdf.cpp"
cl %CFLAGS% /c "%ROOT%src\bitstream.c" "%ROOT%src\color.c" "%ROOT%src\dpb.c" "%ROOT%src\trt_runner.c" "%ROOT%src\dcvc_rt.c"
cl %CFLAGS% /c "%ROOT%plugins\src\plugins_cpu.c" "%ROOT%plugins\src\ar_prior.c"

lib /nologo /OUT:dcvc_rt.lib rans.obj rans_c.obj pmf_cdf.obj bitstream.obj color.obj dpb.obj trt_runner.obj dcvc_rt.obj plugins_cpu.obj ar_prior.obj

cl %CFLAGS% "%ROOT%tests\test_bitstream.c" dcvc_rt.lib /Fe:test_bitstream.exe
cl %CFLAGS% "%ROOT%tests\test_color.c" dcvc_rt.lib /Fe:test_color.exe
cl %CFLAGS% "%ROOT%tests\test_rans.c" dcvc_rt.lib /Fe:test_rans.exe
cl %CFLAGS% "%ROOT%tests\test_roundtrip.c" dcvc_rt.lib /Fe:test_roundtrip.exe

echo Built. Run test_*.exe in this directory.
