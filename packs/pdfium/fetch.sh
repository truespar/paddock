#!/usr/bin/env bash
# Stage OUR prebuilt pdfium without building it.
#
#   bash packs/pdfium/fetch.sh              # linux-x64, the host
#   bash packs/pdfium/fetch.sh win-x64      # stage the Windows library instead
#
# Reads packs/pdfium/prebuilt.json, downloads the library for the platform,
# checks it against the manifest's sha256 and places it where
# crates/paddock-pdfium/build.rs looks: packs/pdfium/<platform>/. This is the
# shortcut the manifest exists for - the recipe in build/ is the provenance,
# this only saves the 15-minute Chromium build.
#
# Idempotent: a library that is already staged and matches the manifest is
# left alone, so it is safe to run on every build. The download lands in a
# .part file and is renamed only after the hash matches, so a killed transfer
# never leaves a wrong-sized library for the linker to find.
#
# Needs curl, and jq or python3 to read the manifest. The Windows twin is
# fetch.ps1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/prebuilt.json"
# Default to this box's CPU: aarch64 (DGX Spark, Jetson) takes the arm64 lane.
case "$(uname -m)" in aarch64|arm64) default_platform="linux-arm64" ;; *) default_platform="linux-x64" ;; esac
PLATFORM="${1:-$default_platform}"

# The manifest is small and its shape is ours, but parse it properly anyway:
# a hand-rolled grep would break the day the manifest is re-emitted with
# different whitespace.
if command -v jq >/dev/null 2>&1; then
    read_field() { jq -r --arg p "$PLATFORM" ".files[] | select(.platform == \$p) | .$1" "$MANIFEST"; }
    read_version() { jq -r '.version' "$MANIFEST"; }
elif command -v python3 >/dev/null 2>&1; then
    read_field() {
        python3 - "$MANIFEST" "$PLATFORM" "$1" <<'PY'
import json, sys
manifest, platform, field = sys.argv[1:]
with open(manifest) as f:
    files = json.load(f)["files"]
print(next((e[field] for e in files if e["platform"] == platform), ""))
PY
    }
    read_version() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$MANIFEST"; }
else
    echo "fetch.sh: need jq or python3 to read $MANIFEST" >&2
    exit 1
fi

url="$(read_field url)"
sha="$(read_field sha256)"
name="$(read_field name)"
if [ -z "$url" ] || [ -z "$sha" ] || [ -z "$name" ]; then
    echo "fetch.sh: no entry for platform '$PLATFORM' in $MANIFEST" >&2
    exit 1
fi

# VERSION is the source of truth for the pin; the manifest carries the same
# string. They move together, so a mismatch means someone bumped one and not
# the other - say so, but still stage what the manifest describes.
pin="$(tr -d '[:space:]' < "$HERE/VERSION")"
version="$(read_version)"
if [ "$pin" != "$version" ]; then
    echo "fetch.sh: warning: packs/pdfium/VERSION is $pin but prebuilt.json is $version" >&2
fi

dest="$HERE/$PLATFORM/$name"
mkdir -p "$HERE/$PLATFORM"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$sha" ]; then
    echo "pdfium already staged: $dest ($version)"
    exit 0
fi

echo "==> pdfium $version for $PLATFORM"
echo "    $url"
curl -fL --progress-bar -o "$dest.part" "$url"

got="$(sha256_of "$dest.part")"
if [ "$got" != "$sha" ]; then
    rm -f "$dest.part"
    echo "fetch.sh: sha256 mismatch for $name" >&2
    echo "    expected $sha" >&2
    echo "    got      $got" >&2
    exit 1
fi
mv -f "$dest.part" "$dest"
echo "    staged $dest"
