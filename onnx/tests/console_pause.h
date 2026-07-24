/* Pause the console window on Windows so the output stays visible when the
 * program is launched by double-click (the default for .exe files in
 * Explorer). Without this, the console closes the instant main() returns and
 * the user cannot read/copy the output.
 *
 * Behavior:
 *   - On non-Windows builds this is a no-op.
 *   - On Windows, it pauses ONLY when the process was started by Explorer
 *     (double-click), detected via the parent-process image name. When run
 *     from cmd / PowerShell / a script / redirected stdin, it returns
 *     immediately so automation is never blocked.
 *
 * Usage: call dcvc_pause_if_dblclick() right before returning from main().
 *
 * No external libs required (kernel32 only). Works with both MinGW-w64 and
 * MSVC builds (C89-compatible). Designed to fail open: on any detection
 * uncertainty it does NOT pause.
 */
#ifndef DCVC_CONSOLE_PAUSE_H
#define DCVC_CONSOLE_PAUSE_H

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>

/* Walk the toolhelp snapshot to find the parent PID of `pid`, then read its
 * image name. Returns 1 if the parent process is explorer.exe. */
static int dcvc_parent_is_explorer(void)
{
    DWORD pid = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;

    PROCESSENTRY32 pe;
    pe.dwSize = sizeof(pe);
    DWORD parent_pid = 0;

    if (Process32First(snap, &pe)) {
        do {
            if (pe.th32ProcessID == pid) { parent_pid = pe.th32ParentProcessID; break; }
        } while (Process32Next(snap, &pe));
    }

    int is_explorer = 0;
    if (parent_pid != 0 && Process32First(snap, &pe)) {
        do {
            if (pe.th32ProcessID == parent_pid) {
                /* Case-insensitive compare; explorer's image name is
                 * "explorer.exe" on all modern Windows. */
                if (lstrcmpiA(pe.szExeFile, "explorer.exe") == 0) is_explorer = 1;
                break;
            }
        } while (Process32Next(snap, &pe));
    }
    CloseHandle(snap);
    return is_explorer;
}

static void dcvc_pause_if_dblclick(void)
{
    /* Pause when either:
     *   - launched by Explorer (double-click), or
     *   - DCVC_FORCE_PAUSE is set (lets a user keep the window open even when
     *     running from a terminal, e.g. for screenshots).
     * Running from a terminal, a script, or with redirected pipes otherwise
     * leaves the window open (or the caller controls the process), so we stay
     * out of the way. */
    int should_pause = dcvc_parent_is_explorer();
    if (!should_pause && getenv("DCVC_FORCE_PAUSE") &&
        getenv("DCVC_FORCE_PAUSE")[0] != '0')
        should_pause = 1;
    if (!should_pause) return;

    /* If stdin is redirected (piped/file), a blocking read would hang; skip. */
    if (GetFileType(GetStdHandle(STD_INPUT_HANDLE)) != FILE_TYPE_CHAR) return;

    fflush(stdout);
    fflush(stderr);
    fputs("\n[Press Enter to exit] ", stdout);
    fflush(stdout);
    (void)getchar();
}
#else
/* No-op on Linux/macOS: terminals don't auto-close. */
static void dcvc_pause_if_dblclick(void) {}
#endif

#endif /* DCVC_CONSOLE_PAUSE_H */
