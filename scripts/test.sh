#!/usr/bin/env bash
# Run the unit tests. Prefers Node.js; falls back to VS Code's bundled
# Electron runtime (ELECTRON_RUN_AS_NODE) on machines without Node installed.
set -e
cd "$(dirname "$0")/.."

if command -v node >/dev/null 2>&1; then
    exec node tests/run.js
fi

CODE_EXE="$LOCALAPPDATA/Programs/Microsoft VS Code/Code.exe"
if [ -f "$CODE_EXE" ]; then
    ELECTRON_RUN_AS_NODE=1 exec "$CODE_EXE" tests/run.js
fi

echo "error: Node.js not found (and no VS Code Electron fallback). Install Node.js to run tests." >&2
exit 1
