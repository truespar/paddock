#!/bin/bash
# Build pdfium from source as one static library for linux-x64 (or, with
# --arch arm64, linux-arm64) and stage it into packs/pdfium/.
#
#   bash packs/pdfium/build/build-linux.sh [--arch x64|arm64] [chromium/NNNN]
#
# Requires Docker. The Windows twin (build-windows.ps1) has to run natively -
# Chromium's Windows build needs a real MSVC install, which a Linux container
# cannot provide - so the two legs look different on the outside while doing
# exactly the same thing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

# Pinned to match pdfium-render's `pdfium_latest = ["pdfium_7881"]` - the crate's
# static bindings are a vtable, so every FPDF_* it declares must resolve at LINK
# time. Building a different vintage than we bind is a build break, not the
# harmless mismatch it is when the library is loaded dynamically.
# `--arch arm64` cross-builds the aarch64 library (DGX Spark, Jetson) on this
# same x64 host: the Dockerfile installs Chromium's arm64 sysroot and reads
# args-linux-arm64.gn. Both lanes publish to R2 beside each other.
ARCH="x64"
if [ "${1:-}" = "--arch" ]; then ARCH="$2"; shift 2; fi
case "$ARCH" in x64|arm64) ;; *) echo "unknown --arch $ARCH (x64|arm64)" >&2; exit 1 ;; esac
VERSION="${1:-chromium/8009}"
IMAGE="paddock-pdfium-builder-${ARCH}:${VERSION//\//-}"
CONTAINER="paddock-pdfium-extract-$$"

echo "==> pdfium build (linux-${ARCH})"
echo "    version: ${VERSION}"
echo "    image:   ${IMAGE}"

docker build \
  --build-arg PDFIUM_VERSION="${VERSION}" \
  --build-arg TARGET_CPU="${ARCH}" \
  -t "${IMAGE}" \
  -f "${HERE}/Dockerfile" \
  "${HERE}"

echo "==> extracting ..."
DST="${REPO}/packs/pdfium/linux-${ARCH}"
mkdir -p "${DST}"
docker create --name "${CONTAINER}" "${IMAGE}" /bin/true > /dev/null
trap 'docker rm -f "${CONTAINER}" > /dev/null 2>&1 || true' EXIT

docker cp "${CONTAINER}:/output/libpdfium.a" "${DST}/libpdfium.a"

# Licences are platform-independent (both legs build the same pin), so they sit
# beside the per-platform dirs rather than inside one. Either leg can produce
# them - whichever runs last writes identical content.
LIC="${REPO}/packs/pdfium/licenses"
rm -rf "${LIC}"
docker cp "${CONTAINER}:/output/licenses" "${LIC}"

docker cp "${CONTAINER}:/output/PDFIUM_COMMIT" /tmp/pdfium_commit
echo "${VERSION}" > "${REPO}/packs/pdfium/VERSION"

echo ""
echo "libpdfium.a  $(du -h "${DST}/libpdfium.a" | cut -f1)  ->  ${DST}"
echo "licences     $(ls "${LIC}" | wc -l) files  ->  ${LIC}"
echo "pinned       ${VERSION}  ($(cat /tmp/pdfium_commit))"
