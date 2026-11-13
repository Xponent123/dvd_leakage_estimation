#!/usr/bin/env bash
# Reproduce the tested Python/ngspice environment inside this project only.
set -euo pipefail
LUSR_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LUSR_PROJECT_DIR=$(CDPATH= cd -- "$LUSR_SCRIPT_DIR/.." && pwd)
LUSR_PYTHON=${PYTHON:-python3}
LUSR_EXTERNAL_NGSPICE=${NGSPICE:-}
LUSR_SKIP_PYTHON=0
LUSR_CHECK_ONLY=0
usage() {
  cat <<'EOF'
Usage: bash scripts/00_setup.sh [--ngspice /absolute/path] [--skip-python] [--check]

Default: create/update .venv from requirements.lock.txt and prepare ngspice 42
inside tools/local. Ubuntu 24.04 amd64 is the tested local-package platform.
--ngspice PATH  Use an existing executable instead of downloading ngspice.
--skip-python   Prepare/check only the simulator.
--check         Validate existing dependencies; never download or install.
PYTHON          Base interpreter for a new environment (tested: Python 3.12).
NGSPICE         Alternative to --ngspice; used by the simulation scripts too.
EOF
}
while (($#)); do
  case "$1" in
    --ngspice)
      (($# >= 2)) || { usage >&2; exit 2; }
      LUSR_EXTERNAL_NGSPICE=$2
      shift 2 ;;
    --skip-python) LUSR_SKIP_PYTHON=1; shift ;;
    --check) LUSR_CHECK_ONLY=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! -f "$LUSR_PROJECT_DIR/requirements.lock.txt" ]]; then
  printf 'Missing requirements.lock.txt\n' >&2
  exit 1
fi
mkdir -p "$LUSR_PROJECT_DIR/.cache/matplotlib" "$LUSR_PROJECT_DIR/.cache/pip" "$LUSR_PROJECT_DIR/tools"
export MPLCONFIGDIR="$LUSR_PROJECT_DIR/.cache/matplotlib"
export PIP_CACHE_DIR="$LUSR_PROJECT_DIR/.cache/pip"
if (( ! LUSR_SKIP_PYTHON )); then
  if [[ ! -x "$LUSR_PROJECT_DIR/.venv/bin/python" ]]; then
    if (( LUSR_CHECK_ONLY )); then
      printf 'Missing .venv; rerun without --check.\n' >&2
      exit 1
    fi
    command -v "$LUSR_PYTHON" >/dev/null || { printf 'Python interpreter not found: %s\n' "$LUSR_PYTHON" >&2; exit 1; }
    "$LUSR_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Tested lock requires Python 3.12; select it using PYTHON=python3.12"'
    "$LUSR_PYTHON" -m venv "$LUSR_PROJECT_DIR/.venv"
  fi
  LUSR_ENV_PYTHON="$LUSR_PROJECT_DIR/.venv/bin/python"
  if (( ! LUSR_CHECK_ONLY )); then
    "$LUSR_ENV_PYTHON" -m pip install --disable-pip-version-check -r "$LUSR_PROJECT_DIR/requirements.lock.txt"
  fi
  "$LUSR_ENV_PYTHON" - "$LUSR_PROJECT_DIR/requirements.lock.txt" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
failures = []
for entry in Path(sys.argv[1]).read_text().splitlines():
    if not entry.strip() or entry.lstrip().startswith('#'):
        continue
    name, expected = entry.split('==', 1)
    try:
        actual = version(name)
    except PackageNotFoundError:
        actual = 'MISSING'
    if actual != expected:
        failures.append(f'{name}: installed {actual}, expected {expected}')
if failures:
    raise SystemExit('\n'.join(failures))
print('All Python dependencies match requirements.lock.txt.')
PY
  "$LUSR_ENV_PYTHON" -m pip check
fi
if [[ -n "$LUSR_EXTERNAL_NGSPICE" ]]; then
  if [[ "$LUSR_EXTERNAL_NGSPICE" != /* ]]; then
    LUSR_EXTERNAL_NGSPICE=$(command -v "$LUSR_EXTERNAL_NGSPICE")
  fi
  LUSR_NGSPICE="$LUSR_EXTERNAL_NGSPICE"
else
  LUSR_NGSPICE="$LUSR_PROJECT_DIR/tools/local/usr/bin/ngspice"
  if [[ ! -x "$LUSR_NGSPICE" ]]; then
    if (( LUSR_CHECK_ONLY )); then
      printf 'Missing local ngspice; rerun without --check or use --ngspice PATH.\n' >&2
      exit 1
    fi
    # The pinned Ubuntu binary has distribution-specific runtime requirements.
    if [[ ! -f /etc/os-release ]]; then
      printf 'Automatic ngspice extraction requires Ubuntu 24.04 amd64. Supply --ngspice PATH on other systems.\n' >&2
      exit 1
    fi
    LUSR_OS_ID=$(. /etc/os-release; printf '%s' "$ID")
    LUSR_OS_VERSION=$(. /etc/os-release; printf '%s' "$VERSION_ID")
    LUSR_ARCH=$(dpkg --print-architecture 2>/dev/null || true)
    if [[ "$LUSR_OS_ID" != ubuntu || "$LUSR_OS_VERSION" != 24.04 || "$LUSR_ARCH" != amd64 ]]; then
      printf 'Automatic ngspice extraction supports Ubuntu 24.04 amd64; use --ngspice PATH on this platform.\n' >&2
      exit 1
    fi
    command -v apt-get >/dev/null
    command -v dpkg-deb >/dev/null
    LUSR_DEBS="$LUSR_PROJECT_DIR/tools/debs"
    mkdir -p "$LUSR_DEBS" "$LUSR_PROJECT_DIR/tools/local"
    for LUSR_PACKAGE in ngspice libngspice0; do
      LUSR_ARCHIVE="$LUSR_DEBS/${LUSR_PACKAGE}_42+ds-3build1_amd64.deb"
      if [[ ! -f "$LUSR_ARCHIVE" ]]; then
        (cd "$LUSR_DEBS" && apt-get download "${LUSR_PACKAGE}=42+ds-3build1")
      fi
      case "$LUSR_PACKAGE" in
        ngspice) LUSR_EXPECTED_SHA=466c4c06418107ceaa9c9457065b3bb71a9d9dc5ec6fef186d7de9d5208ce8db ;;
        libngspice0) LUSR_EXPECTED_SHA=967d5c0c83ab264f344504fa40d8475b7bc4e80abac4395e43ac04fec1b0de98 ;;
      esac
      printf '%s  %s\n' "$LUSR_EXPECTED_SHA" "$LUSR_ARCHIVE" | sha256sum --check --status
      dpkg-deb --extract "$LUSR_ARCHIVE" "$LUSR_PROJECT_DIR/tools/local"
    done
  fi
fi
if [[ ! -x "$LUSR_NGSPICE" ]]; then
  printf 'ngspice is not executable: %s\n' "$LUSR_NGSPICE" >&2
  exit 1
fi
"$LUSR_NGSPICE" --version
printf '\nReady. Simulator: %s\n' "$LUSR_NGSPICE"
if [[ -n "$LUSR_EXTERNAL_NGSPICE" ]]; then
  printf 'For subsequent commands, export NGSPICE=%q\n' "$LUSR_NGSPICE"
fi
printf 'Use .venv/bin/python for project scripts. No system packages were installed.\n'
