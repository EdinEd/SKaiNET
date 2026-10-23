#include "skainet_cpu_features.h"

#if defined(__APPLE__) && defined(__aarch64__)

#include <sys/sysctl.h>

int skainet_cpu_has_dotprod(void) {
    /* Benign-race cache: concurrent first calls compute and write the same
     * value, so no synchronization is needed. */
    static int cached = -1;
    if (cached < 0) {
        int v = 0;
        size_t sz = sizeof v;
        /* The key exists since iOS 15 / macOS 12; on older OS versions
         * sysctlbyname fails => 0 => scalar fallback (always safe). A12
         * (iPhone XS/XR) reports 0; A13+ and every Apple Silicon Mac
         * report 1. */
        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &sz, NULL, 0) != 0) {
            v = 0;
        }
        cached = (v != 0);
    }
    return cached;
}

#else /* non-Apple: the compile-time -march decides, no runtime probe. */

int skainet_cpu_has_dotprod(void) {
#if defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__wasm__)

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>

int skainet_cpu_has_avx2_fma(void) {
    /* Benign-race cache, as for the dotprod probe above. */
    static int cached = -1;
    if (cached < 0) {
        int ok = 0;
        int r[4];
        __cpuid(r, 0);
        if (r[0] >= 7) {
            __cpuid(r, 1);
            const int fma = (r[2] >> 12) & 1;
            const int osxsave = (r[2] >> 27) & 1;
            const int avx = (r[2] >> 28) & 1;
            __cpuidex(r, 7, 0);
            const int avx2 = (r[1] >> 5) & 1;
            if (fma && osxsave && avx && avx2) {
                /* XCR0 bits 1 (XMM) and 2 (YMM): the OS restores the upper halves. */
                ok = (_xgetbv(0) & 0x6) == 0x6;
            }
        }
        cached = ok;
    }
    return cached;
}

#else /* GCC / Clang: libgcc's probe also checks the OS XSAVE state. */

int skainet_cpu_has_avx2_fma(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }
    return cached;
}

#endif

#else /* not x86-64 */

int skainet_cpu_has_avx2_fma(void) {
    return 0;
}

#endif
