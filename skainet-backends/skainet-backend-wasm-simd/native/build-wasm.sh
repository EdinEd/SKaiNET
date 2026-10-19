#!/usr/bin/env bash
# SKEEP-007 Tier 1: build the browser kernel sidecar from the engine's native kernel sources.
#
# Produces src/wasmJsMain/kotlin/sk/ainet/exec/wasm/SidecarArtifacts.kt: the two sidecar builds —
#   skainet_kernels_st.{mjs,wasm}   single-threaded, SIMD128
#   skainet_kernels_mt.{mjs,wasm}   pthreads (needs a cross-origin-isolated page or Node), SIMD128
# — embedded as base64 constants. Embedding, not resources: Kotlin/Wasm copies a library's resources
# into neither a consumer's bundle nor its Node output, so a sidecar shipped as files would have to be
# hosted by every application. As constants it travels inside the klib and is instantiated from a
# data: URL with the wasm bytes handed to Emscripten directly (no fetch, no path).
#
# The sources are skainet-backend-native-cpu/native/src — the same files the JVM FFM and Android JNI
# packs are built from — compiled with Emscripten. Only the portable (#else) kernel bodies and the
# explicit __wasm_simd128__ bodies are used; NEON and Apple dispatch are not part of this build.
# The ternary kernels come from the portable branch of skainet_ternary_f32.c (the compile-time LUT
# with its SIMD128 body), not from the vendored NeoGPU file, which needs native pthreads and NEON.
#
# Toolchain: emsdk, pinned below. `EMSDK_PYTHON` must point at a Python >= 3.10 when the default
# python3 is older (pyenv). Usage:
#   EMSDK=~/emsdk EMSDK_PYTHON=/usr/bin/python3 skainet-backends/skainet-backend-wasm-simd/native/build-wasm.sh
set -euo pipefail

EMSDK_VERSION_PINNED="4.0.14"
HERE="$(cd "$(dirname "$0")" && pwd)"
NATIVE="$HERE/../../skainet-backend-native-cpu/native"
OUT="$HERE/build"
GEN="$HERE/../src/wasmJsMain/kotlin/sk/ainet/exec/wasm/SidecarArtifacts.kt"
EMSDK="${EMSDK:-$HOME/emsdk}"

if ! command -v emcc >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$EMSDK/emsdk_env.sh" >/dev/null 2>&1
fi
emcc --version | head -1

SOURCES=(
  "$NATIVE/src/fp32_matmul.c"
  "$NATIVE/src/fp32_matmul_wt.c"
  "$NATIVE/src/bf16_matmul.c"
  "$NATIVE/src/fp16_matmul.c"
  "$NATIVE/src/q4_0_matmul.c"
  "$NATIVE/src/q5_0_matmul.c"
  "$NATIVE/src/q5_1_matmul.c"
  "$NATIVE/src/q8_0_matmul.c"
  "$NATIVE/src/q4k_matmul.c"
  "$NATIVE/src/q5k_matmul.c"
  "$NATIVE/src/q6k_matmul.c"
  "$NATIVE/src/skainet_ternary_f32.c"
  "$NATIVE/src/bitnet_gemv.c"
  "$NATIVE/src/f32_ops.c"
  "$NATIVE/src/skainet_row_threads.c"
  "$NATIVE/src/skainet_cpu_features.c"
)

EXPORTS='_malloc,_free,_skainet_fp32_matmul,_skainet_fp32_matmul_wt,_skainet_bf16_matmul,_skainet_fp16_matmul'
EXPORTS="$EXPORTS,_skainet_ternary_f32_gemv,_skainet_ternary_lmhead_stage1,_skainet_bitnet_gemv_tq2_0"
EXPORTS="$EXPORTS,_skainet_f32_binary,_skainet_f32_unary,_skainet_f32_reduce_all,_skainet_f32_reduce_axis"
for f in q4_0 q5_0 q5_1 q8_0 q4k q5k q6k; do
  EXPORTS="$EXPORTS,_skainet_${f}_matmul,_skainet_${f}_matmul_rm"
done

COMMON=(
  -O3 -msimd128 -ffast-math -funroll-loops
  -I"$NATIVE/include"
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createSkainetKernels
  -sENVIRONMENT=web,worker,node
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB -sMAXIMUM_MEMORY=4GB
  -sFILESYSTEM=0 -sASSERTIONS=0
  "-sEXPORTED_FUNCTIONS=$EXPORTS"
  "-sEXPORTED_RUNTIME_METHODS=wasmMemory"
)

mkdir -p "$OUT"
echo "== single-threaded"
emcc "${COMMON[@]}" "${SOURCES[@]}" -o "$OUT/skainet_kernels_st.mjs"
echo "== pthreads"
emcc "${COMMON[@]}" -pthread -sPTHREAD_POOL_SIZE=4 -sPTHREAD_POOL_SIZE_STRICT=0 \
  "${SOURCES[@]}" -o "$OUT/skainet_kernels_mt.mjs"
ls -la "$OUT"/skainet_kernels_*

# The glue assumes it was loaded from a file: or http(s): URL. Embedded, it is loaded from a data:
# (Node) or blob: (browser) URL, so two spots are patched — each `grep -c` asserts the pinned emsdk
# still emits the exact text, so a toolchain change fails here and not at runtime:
#   1. Node's `createRequire(import.meta.url)` rejects non-file URLs; fall back to the cwd.
#   2. `findWasmBinary` resolves the .wasm name against import.meta.url, which throws for an opaque
#      base; the bytes are passed in as `wasmBinary` anyway, so the name only has to be a string.
patch_glue() {
  local f="$1" name="$2"
  test "$(grep -c 'createRequire(import.meta.url)' "$f")" = "1"
  sed -i 's|createRequire(import.meta.url)|createRequire(import.meta.url.startsWith("file:")?import.meta.url:process.cwd()+"/")|' "$f"
  test "$(grep -c "return new URL(\"$name.wasm\",import.meta.url).href" "$f")" = "1"
  sed -i "s|return new URL(\"$name.wasm\",import.meta.url).href|try{return new URL(\"$name.wasm\",import.meta.url).href}catch(e){return \"$name.wasm\"}|" "$f"
}
patch_glue "$OUT/skainet_kernels_st.mjs" skainet_kernels_st
patch_glue "$OUT/skainet_kernels_mt.mjs" skainet_kernels_mt
#   3. (pthreads) Node's worker_threads wants a data: URL wrapped in `new URL(...)`, a browser wants
#      the blob URL string the glue derives itself; keep both.
WORKER_OLD='worker=new Worker(pthreadMainJs,{type:"module",workerData:"em-pthread",name:"em-pthread"})'
WORKER_NEW='worker=new Worker(ENVIRONMENT_IS_NODE\&\&typeof pthreadMainJs=="string"\&\&pthreadMainJs.startsWith("data:")?new URL(pthreadMainJs):pthreadMainJs,{type:"module",workerData:"em-pthread",name:"em-pthread"})'
test "$(grep -cF "$WORKER_OLD" "$OUT/skainet_kernels_mt.mjs")" = "1"
sed -i "s|$WORKER_OLD|$WORKER_NEW|" "$OUT/skainet_kernels_mt.mjs"
test "$(grep -cF 'new URL(pthreadMainJs)' "$OUT/skainet_kernels_mt.mjs")" = "1"
b64() { base64 -w0 "$1"; }
mkdir -p "$(dirname "$GEN")"
{
  echo "// GENERATED by native/build-wasm.sh (emsdk $EMSDK_VERSION_PINNED) -- do not edit."
  echo "// SKEEP-007 Tier 1: the SIMD128 kernel sidecar, embedded so every consumer of this module carries it."
  echo "// sha256 st: $(sha256sum "$OUT/skainet_kernels_st.wasm" | cut -c1-64)"
  echo "// sha256 mt: $(sha256sum "$OUT/skainet_kernels_mt.wasm" | cut -c1-64)"
  echo "package sk.ainet.exec.wasm"
  echo
  echo "internal object SidecarArtifacts {"
  echo "    const val EMSDK: String = \"$EMSDK_VERSION_PINNED\""
  echo "    const val ST_MJS: String = \"$(b64 "$OUT/skainet_kernels_st.mjs")\""
  echo "    const val ST_WASM: String = \"$(b64 "$OUT/skainet_kernels_st.wasm")\""
  echo "    const val MT_MJS: String = \"$(b64 "$OUT/skainet_kernels_mt.mjs")\""
  echo "    const val MT_WASM: String = \"$(b64 "$OUT/skainet_kernels_mt.wasm")\""
  echo "}"
} > "$GEN"
wc -c "$GEN"
echo "built with emsdk $(emcc --version | head -1 | awk '{print $NF}') (pinned: $EMSDK_VERSION_PINNED)"
