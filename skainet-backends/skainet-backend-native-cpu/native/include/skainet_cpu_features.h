#ifndef SKAINET_CPU_FEATURES_H
#define SKAINET_CPU_FEATURES_H

/*
 * Runtime CPU feature probes for the native kernels. Internal — this header
 * is NOT part of the cinterop surface (skainet_kernels.def headerFilter).
 *
 * Returns 1 if the CPU executes FEAT_DotProd (sdot/udot). On non-Apple
 * builds this is a compile-time constant derived from the -march the TU was
 * built with; on Apple arm64 it is a cached sysctl probe, because a klib
 * embeds exactly one archive and Apple A12 (iPhone XS/XR, still supported
 * by current iOS) lacks the feature while A13+ and all Apple Silicon have
 * it (#920).
 */
int skainet_cpu_has_dotprod(void);

/*
 * Returns 1 if this x86-64 CPU and OS execute AVX2 and FMA (and the OS saves
 * the YMM state). Cached after the first call; 0 on every other architecture.
 * The shared library is built for baseline x86-64, so kernels that want these
 * instructions compile the hot body a second time with the avx2+fma target
 * and pick it per call through this probe (fp32_matmul_wt.c).
 */
int skainet_cpu_has_avx2_fma(void);

#endif /* SKAINET_CPU_FEATURES_H */
