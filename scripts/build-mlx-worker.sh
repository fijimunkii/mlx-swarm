#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="$ROOT_DIR/worker/mlx"
DERIVED_DATA="${MLX_SWARM_DERIVED_DATA:-$PACKAGE_DIR/.build/xcode}"
PACKAGE_CLONES="${MLX_SWARM_PACKAGE_CLONES:-$PACKAGE_DIR/.build/source-packages}"

mkdir -p "$DERIVED_DATA" "$PACKAGE_CLONES"

(
  cd "$PACKAGE_DIR"
  xcodebuild build \
    -scheme MLXWorker \
    -destination 'platform=macOS' \
    -derivedDataPath "$DERIVED_DATA" \
    -clonedSourcePackagesDirPath "$PACKAGE_CLONES" \
    -disablePackageRepositoryCache \
    -skipPackagePluginValidation \
    -skipMacroValidation \
    CODE_SIGNING_ALLOWED=NO
)

WORKER="$DERIVED_DATA/Build/Products/Debug/MLXWorker"
if [[ ! -x "$WORKER" ]]; then
  echo "MLXWorker executable not found at $WORKER" >&2
  exit 1
fi

printf '%s\n' "$WORKER"
