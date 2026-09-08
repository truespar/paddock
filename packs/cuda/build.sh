#!/usr/bin/env bash
# Linux twin of build.ps1: builds the CUDA kernel pack as a multi-arch fatbin.
#   packs/cuda/build.sh              # all default arches the toolkit supports
#   packs/cuda/build.sh 120          # a subset (faster local iterate)
#   PD_PACK_OUT=/somewhere build.sh  # write artifacts elsewhere (see below)
# Output: build/pd-cuda-sm120.so (plus a pd-cuda-sm86.dll symlink for the
# hardcoded test paths). When sm_120 is in the arch list the sm_120a feature
# target and -DPD_BS_HOST=1 are added so the block-scale MoE launchers are
# real - without them the kernel table exports NULL for those entries and the
# engine keeps the s8 mmq path (see the PD_BS_HOST notes in pack.cu).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

# PD_PACK_OUT redirects every artifact this script writes. It exists because
# a Docker builder bind-mounts the WINDOWS repo, so
# `build/` in the container is `packs\cuda\build\` on the host - and the two
# lanes do not merely coexist there. The .so lane's compatibility symlink is
# named `pd-cuda-sm86.dll`, which is exactly the file build.ps1 writes, so a
# Linux build would replace the Windows pack with a dangling link to an ELF.
# `cutgemm.o` collides the same way. Same reasoning as CARGO_TARGET_DIR in the
# builder image: Linux artifacts never land in a Windows tree.
outdir="${PD_PACK_OUT:-$here/build}"
mkdir -p "$outdir"

# --static emits build/libpd-cuda.a for `cargo build --features static-pack`
# to link into paddock-runner, instead of the.so. Orthogonal to
# the arch list deliberately: a RELEASE passes the validated set
# (`--static 86,100,120`) so the binary and the engine's gpu_support allowlist
# agree by construction, and that choice stays visible in the release script
# rather than hiding behind a linkage switch.
static=0
while :; do
    case "${1:-}" in
        --static)   static=1;   shift ;;
        *) break ;;
    esac
done

arch_arg="${1:-}"
arches=(${arch_arg//,/ })
# 103 (Blackwell Ultra) and 110 are built as PLAIN targets, no 'a' feature
# variant. The fatbin carries no PTX, so a die absent from this list cannot
# load the pack at all -- which is how 12.1 was unsupported. They are safe to
# add only because the accelerated families gate on an exact cc match
# (pd_dev_bs_sass in abi.cuh, the table resolution in exports.cuh): a 10.3
# device gets the portable paths instead of tcgen05 bodies that its target
# compiled away. Adding an 'a' variant for them needs the device feature
# macros widened first -- PD_TC5_OK is `__CUDA_ARCH__ == 1000` -- or they
# silently no-op. 121 (DGX Spark GB10) got exactly that treatment on
# 2026-09-07: PD_BS_OK accepts __CUDA_ARCH_FEAT_SM121_ALL, so 121 builds its
# 'a' twin beside the plain image (like 120) and -DPD_BS_SM121=1 tells the
# host election that the 12.1 bodies are real.
[ ${#arches[@]} -eq 0 ] && arches=(80 86 89 90 100 103 110 120 121)

supported="$(nvcc --list-gpu-arch | sed 's/compute_//')"
gencode=()
defines=()
bs_host=0
for a in "${arches[@]}"; do
    if grep -qx "$a" <<<"$supported"; then
        gencode+=("-gencode=arch=compute_$a,code=sm_$a")
        # --list-gpu-arch omits feature targets; any toolkit that knows
        # compute_120 (CUDA >= 12.8) accepts the 120a feature target too
        if [ "$a" = "120" ]; then
            # the 'a' feature target carries the block-scale (mxf8f6f4) MMA
            gencode+=("-gencode=arch=compute_120a,code=sm_120a")
            bs_host=1
        fi
        if [ "$a" = "121" ]; then
            # GB10's feature target: same block-scale MMA family as 120a
            # (PD_BS_OK accepts its macro); PD_BS_SM121 lets pd_dev_bs_sass
            # elect those bodies on a 12.1 die
            gencode+=("-gencode=arch=compute_121a,code=sm_121a")
            bs_host=1
            defines+=("-DPD_BS_SM121=1")
        fi
        # sm_100 (B200): the f8w8 family rides plain e4m3 mma + sw ue8m0 fold
        # (no 'a' target needed); PD_BS_HOST makes the launchers real and
        # paddock_pack_kernels_v1 NULLs the sm_120a-only families per device.
        # The 100a feature target carries tcgen05 (tensor-memory MMA) for the
        # rowwise-e4m3 GEMM; PD_TC5_HOST turns its launcher route on.
        if [ "$a" = "100" ]; then
            bs_host=1
            gencode+=("-gencode=arch=compute_100a,code=sm_100a")
            defines+=("-DPD_TC5_HOST=1")
        fi
    else
        echo "WARN: toolkit does not support sm_$a, skipping" >&2
    fi
done
[ "$bs_host" = 1 ] && defines+=("-DPD_BS_HOST=1")
# PD_DEFS: extra -D flags for a measurement build (e.g. PD_DEFS=-DPD_FA_KRS_OCC=3
# to sweep the krs occupancy rung). Never set for a release - a shipped default
# belongs in the header next to the measurement that elected it.
[ -n "${PD_DEFS:-}" ] && defines+=(${PD_DEFS})
# see abi.cuh: an archive is resolved by address at link time, so exporting the
# names from the consuming binary buys nothing
[ "$static" = 1 ] && defines+=("-DPD_STATIC=1")

[ ${#gencode[@]} -gt 0 ] || { echo "no supported arches" >&2; exit 1; }

out="$outdir/pd-cuda-sm120.so"
# --threads: nvcc compiles every -gencode target SERIALLY by default, so the
# 8 targets above ran one cicc at a time - measured on the Windows twin at
# 99.8% of one core on a 32-core box for ~14 minutes. 0 = one thread per
# target, capped by cores. Same generated code, just not one-at-a-time.
threads="${PD_BUILD_THREADS:-0}"


link=(--shared)
# PD_EXP_LT=1: compile the cuBLASLt datapath-ceiling EXPERIMENT arm (slot 542
# real instead of stub) and link the library. Never for a shipped pack.
if [ -n "${PD_EXP_LT:-}" ]; then
    defines+=(-DPD_EXP_LT=1)
    link+=(-lcublasLt)
fi
if [ "$static" = 1 ]; then
    # -lib archives the objects. Same fatbin, same two exports - only how the
    # engine reaches them differs. No -fPIC concern either way: it is already
    # passed above and is harmless in an archive destined for an executable.
    link=(-lib)
    out="$outdir/libpd-cuda.a"
fi
echo "building fatbin: ${gencode[*]} ${defines[*]:-} ${link[*]} --threads $threads"
nvcc -O3 --threads "$threads" "${gencode[@]}" ${defines[@]:+"${defines[@]}"} \
    -Xcompiler -fPIC "${link[@]}" -o "$out" "$here/pack.cu"
# the .dll alias exists only for the hardcoded test paths on the .so lane
[ "$static" = 1 ] || ln -sf pd-cuda-sm120.so "$outdir/pd-cuda-sm86.dll"
echo "built (multi-arch): $out"

# Record what this archive runs on, and which nvcc emitted it, BESIDE it - the
# Windows twin wrote these two files first and this one did not, which is the
# same build.ps1/build.sh drift its own comments complain about twice.
# The packaging step reads them for VERSION.txt.
#
# It matters more here than it looks: the fatbin carries no PTX, so the arch
# list is the hardware list, and once the kernels are welded into a binary
# nobody without the CUDA tools can recover it. Written at BUILD time on
# purpose - asking `nvcc --version` while packaging describes the checkout's
# toolkit, which may have moved since these bytes were produced.
if [ "$static" = 1 ]; then
    printf '%s\n' "${gencode[@]}" | sed 's/.*code=//' | paste -sd' ' - > "$outdir/pd-cuda.arches"
    nvcc --version | sed -n 's/.*release [0-9.]*, V\([0-9.]*\).*/\1/p' > "$outdir/pd-cuda.nvcc"
    echo "arches: $(cat "$outdir/pd-cuda.arches")  (nvcc $(cat "$outdir/pd-cuda.nvcc"))"
fi
