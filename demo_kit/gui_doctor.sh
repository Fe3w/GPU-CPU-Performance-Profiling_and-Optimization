#!/usr/bin/env bash
# gui_doctor.sh — find out why ncu-ui closes. Opens NO profile, launches the
# GUI in the foreground with Qt debug output, and checks the usual causes.
# Run it in a terminal you don't mind losing; it writes everything to a log.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/demo.conf"
LOG="/tmp/ncu-ui-doctor.log"; : > "$LOG"
REAL="$(dirname "$NCU_UI")/host/linux-desktop-glibc_2_11_3-x64"

{
echo "== session: XDG_SESSION_TYPE=$XDG_SESSION_TYPE DISPLAY=$DISPLAY"
echo "== missing shared libs for the GUI (any 'not found' is a likely cause):"
for f in "$REAL/ncu-ui.bin" "$REAL/ncu-ui" "$REAL"/Plugins/platforms/libqxcb.so "$REAL"/platforms/libqxcb.so; do
  [[ -f "$f" ]] && { echo "-- $f"; ldd "$f" 2>/dev/null | grep 'not found'; }
done
echo "== libxcb-cursor0 (required by Qt >= 6.5 on X11):"
dpkg -s libxcb-cursor0 2>/dev/null | grep Status || echo "NOT INSTALLED -> sudo apt install libxcb-cursor0"
} | tee -a "$LOG"

try() {  # label env...
  echo; echo "== attempt: $1" | tee -a "$LOG"; shift
  ( export QT_DEBUG_PLUGINS=1 "$@"; timeout 20 "$NCU_UI" ) >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc == 124 ]]; then
    echo "   stayed open for 20 s → this setting WORKS. Use it:  env $* $NCU_UI" | tee -a "$LOG"; return 0
  fi
  echo "   exited with code $rc" | tee -a "$LOG"; return 1
}

try "default"                                               QT_QPA_PLATFORM=xcb && exit 0
try "no GL integration"                                     QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=none && exit 0
try "software OpenGL"                                       QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 QT_OPENGL=software && exit 0
try "software GL + no chromium sandbox (web views)"         QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 QTWEBENGINE_CHROMIUM_FLAGS=--no-sandbox && exit 0

echo
echo "All attempts closed. Last 40 lines of $LOG:"
tail -40 "$LOG"
echo
echo "Full log: $LOG (search it for 'qt.qpa' and 'error'). Meanwhile the demo works without the GUI:"
echo "  ./demo.sh ncu   (CLI details + comparison table)   and the screenshots in the deck."
