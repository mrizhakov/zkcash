#!/bin/bash
set -e

# Usage:
#   ./repo-install.sh [use_ssh]
#
# If use_ssh is "true" → clone via SSH, otherwise via HTTPS.

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TARGET_DIR="$BASE_DIR/repos"

USE_SSH="${1:-true}"

mkdir -p "$TARGET_DIR"

if [ "$USE_SSH" = "true" ]; then
  echo "🔑 Using SSH links"
  REPOS=(
    "git@github.com:rarimo/bionetta-witness-generator.git"
    "git@github.com:rarimo/bionetta-circom.git"
    "git@github.com:rarimo/ultragroth.git"
    "git@github.com:rarimo/ultragroth-snarkjs.git"
  )
else
  echo "🌐 Using HTTPS links"
  REPOS=(
    "https://github.com/rarimo/bionetta-witness-generator.git"
    "https://github.com/rarimo/bionetta-circom.git"
    "https://github.com/rarimo/ultragroth.git"
    "https://github.com/rarimo/ultragroth-snarkjs.git"
  )
fi

for REPO_URL in "${REPOS[@]}"; do
  REPO_NAME=$(basename -s .git "$REPO_URL")
  DEST="$TARGET_DIR/$REPO_NAME"

  if [ -d "$DEST/.git" ]; then
    echo "➡️  Updating $REPO_NAME ..."
    git -C "$DEST" pull --rebase
    continue
  fi
  
  echo "➡️  Cloning $REPO_NAME ..."
  git clone "$REPO_URL" "$DEST"

  if [ "$REPO_NAME" = "ultragroth" ]; then
    echo "🔧 Building ultragroth..."
    pushd "$DEST" > /dev/null

    git submodule init
    git submodule update

    SYSTEM=$(uname -s)-$(uname -m)
    echo "Detected system: $SYSTEM"

    case "$SYSTEM" in
      Darwin-x86_64|Linux-x86_64)
        ./build_gmp.sh host
        make host
        ;;
      Darwin-arm64)
        ./build_gmp.sh macos_arm64
        make macos_arm64
        ;;
      Linux-aarch64)
        ./build_gmp.sh host
        make arm64
        ;;
      *)
        echo "⚠️ Unsupported system: $SYSTEM"
        ;;
    esac

    popd > /dev/null
    echo "✅ ultragroth built successfully"
  fi
done

echo "✅ All repositories are up to date in $TARGET_DIR"
