# Build pdfium from source as ONE static library for linux-x64 or linux-arm64.
#
# Retargeted from lector's WASM build (c:\dev\lector packages/pdfium-wasm/build):
# same depot_tools + gclient recipe, native clang toolchain instead of
# emscripten, and pdf_is_complete_lib so the result is a single libpdfium.a
# that links straight into paddock-runner.
#
# Driven by build-linux.sh - see there for the extract step.
#
# Build args:
#   PDFIUM_VERSION  - pdfium branch to build (e.g. chromium/8009). The default
#                     below is the pin; packs/pdfium/VERSION is what everything
#                     else reads, and the two must agree.
#   TARGET_CPU      - x64 (default) or arm64; picks args-linux-<cpu>.gn.

# Pinned by digest, matching lector's base: the floating `24.04` tag drifts and
# a base-image update would silently invalidate the layer cache AND change the
# compiler the library was built with.
FROM ubuntu:24.04@sha256:c4a8d5503dfb2a3eb8ab5f807da5bc69a85730fb49b5cfca2330194ebcc41c7b

ENV DEBIAN_FRONTEND=noninteractive

# --- Layer 1: system dependencies (rarely changes) ---
# Deliberately thin: use_sysroot=true means the BUILD's headers and libs come
# from Chromium's bundled sysroot, not from this image, so we need only enough
# to run depot_tools and the generators.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    curl \
    ca-certificates \
    xz-utils \
    lsb-release \
    file \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 2: depot_tools (rarely changes) ---
RUN git clone --depth 1 https://chromium.googlesource.com/chromium/tools/depot_tools.git /opt/depot_tools
ENV PATH="/opt/depot_tools:${PATH}"

# --- Layer 3: gclient solution (never changes) ---
# `checkout_configuration: small` skips the test corpora and reference binaries.
WORKDIR /build
RUN cat > .gclient <<'GCLIENT'
solutions = [
  {
    "name": "pdfium",
    "url": "https://pdfium.googlesource.com/pdfium.git",
    "managed": False,
    "custom_deps": {},
    "custom_vars": {
      "checkout_configuration": "small",
    },
  },
]
GCLIENT

# --- Layer 4: fetch source at the pinned version (changes on version bumps) ---
ARG PDFIUM_VERSION=chromium/8009
RUN git clone https://pdfium.googlesource.com/pdfium.git /build/pdfium \
    && cd /build/pdfium \
    && git fetch origin ${PDFIUM_VERSION} \
    && git checkout FETCH_HEAD \
    && cd /build \
    && gclient sync --no-history --shallow --nohooks

# Hooks fetch clang, gn, ninja and the sysroot - separate layer so a hook
# failure is distinguishable from a source-fetch failure.
RUN cd /build && gclient runhooks
# --- Layer 4b: the target CPU. x64 is the release lane; arm64 is cross-built
# for aarch64 boxes (DGX Spark, Jetson) and needs Chromium's arm64 sysroot on
# top of the host one the hooks fetched.
ARG TARGET_CPU=x64
RUN if [ "${TARGET_CPU}" = "arm64" ]; then       cd /build/pdfium && python3 build/linux/sysroot_scripts/install-sysroot.py --arch=arm64;     fi

# --- Layer 5: configure + build ---
COPY args-linux-${TARGET_CPU}.gn /build/pdfium/out/static/args.gn
WORKDIR /build/pdfium
RUN gn gen out/static
RUN ninja -C out/static pdfium -j$(nproc)

# --- Layer 6: collect the deliverables ---
# The library, plus the licences we are now obliged to reproduce ourselves
# (they used to arrive inside bblanchon's archive). Over-collect on purpose:
# crediting a component we did not link is harmless, omitting one we did is a
# breach.
RUN mkdir -p /output/licenses \
    && cp out/static/obj/libpdfium.a /output/libpdfium.a \
    && cp LICENSE /output/licenses/pdfium-LICENSE.txt \
    && for d in third_party/*/; do \
         name=$(basename "$d"); \
         lic=$(ls "$d"LICENSE* 2>/dev/null | head -1); \
         if [ -n "$lic" ]; then cp "$lic" "/output/licenses/${name}-LICENSE.txt"; fi; \
       done \
    && git log -1 --format=%H > /output/PDFIUM_COMMIT \
    && echo "${PDFIUM_VERSION}" > /output/PDFIUM_VERSION \
    && ls -l /output/libpdfium.a \
    && echo "licences: $(ls /output/licenses | wc -l)"
