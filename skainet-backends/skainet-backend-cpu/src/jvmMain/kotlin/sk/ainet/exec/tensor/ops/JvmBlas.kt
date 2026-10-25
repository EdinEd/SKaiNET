package sk.ainet.exec.tensor.ops

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import java.nio.file.Files
import java.nio.file.Path

/**
 * The platform BLAS's `cblas_sgemm` through Panama FFM — on macOS the Accelerate framework, whose
 * SGEMM runs on the matrix units of Apple silicon and threads itself; elsewhere OpenBLAS or the
 * reference BLAS when one is installed. Optional: [isAvailable] is false when no library resolves,
 * and the ops layer consults [JvmCpuBackendConfig.blasEnabled] before asking.
 *
 * Library resolution, first hit wins: the `skainet.cpu.blas.library` system property or the
 * `SKAINET_CPU_BLAS_LIBRARY` environment variable (a path, or a name `dlopen` resolves), the
 * Accelerate framework by its path on macOS, then the common Linux sonames. A bare `"Accelerate"`
 * never resolves — `dlopen` searches library paths for a leaf name, and a framework lives under
 * `/System/Library/Frameworks` — and a heap `MemorySegment.ofArray` cannot be passed to a downcall
 * on JDK 21, which is why the previous version of this object never reached a BLAS; both faces below
 * take off-heap segments and the heap face stages its arrays itself.
 *
 * Row-major throughout. Dimensions are the C-side ints; a product with more than `Int.MAX_VALUE`
 * elements in an operand is the caller's problem long before it reaches here.
 */
internal object JvmBlas {
    private const val CBLAS_ROW_MAJOR = 101
    private const val CBLAS_NO_TRANS = 111
    private const val CBLAS_TRANS = 112

    private const val PROP_LIBRARY = "skainet.cpu.blas.library"
    private const val ENV_LIBRARY = "SKAINET_CPU_BLAS_LIBRARY"
    private const val ACCELERATE_PATH = "/System/Library/Frameworks/Accelerate.framework/Accelerate"

    private val linker: Linker = Linker.nativeLinker()

    /** The library name or path `cblas_sgemm` was resolved from, or `null` when none has (yet). */
    @Volatile
    var libraryDescription: String? = null
        private set

    @Volatile
    private var lookedUp: Boolean = false

    @Volatile
    private var sgemmHandle: MethodHandle? = null

    /** Whether a `cblas_sgemm` has been resolved; the first call does the lookup. */
    fun isAvailable(): Boolean {
        if (lookedUp) return sgemmHandle != null
        synchronized(this) {
            if (lookedUp) return sgemmHandle != null
            resolve()
            lookedUp = true
            return sgemmHandle != null
        }
    }

    /** Drops the resolved handle so the next [isAvailable] resolves again (tests change the library property). */
    fun reload() {
        synchronized(this) {
            sgemmHandle = null
            libraryDescription = null
            lookedUp = false
        }
    }

    private fun candidates(): List<String> {
        val explicit = runCatching { System.getProperty(PROP_LIBRARY) }.getOrNull()?.takeIf { it.isNotBlank() }
            ?: runCatching { System.getenv(ENV_LIBRARY) }.getOrNull()?.takeIf { it.isNotBlank() }
        val os = runCatching { System.getProperty("os.name") }.getOrNull()?.lowercase() ?: ""
        val platform = when {
            os.contains("mac") || os.contains("darwin") -> listOf(ACCELERATE_PATH)
            os.contains("win") -> listOf("libopenblas.dll", "openblas.dll")
            else -> listOf("libopenblas.so.0", "libopenblas.so", "libblas.so.3", "libcblas.so.3")
        }
        return listOfNotNull(explicit) + platform
    }

    private fun resolve() {
        val arena = Arena.ofAuto()
        for (candidate in candidates()) {
            val lookup = runCatching {
                val asPath = Path.of(candidate)
                if (candidate.contains('/') || candidate.contains('\\')) {
                    if (!Files.exists(asPath)) return@runCatching null
                    SymbolLookup.libraryLookup(asPath, arena)
                } else {
                    SymbolLookup.libraryLookup(candidate, arena)
                }
            }.getOrNull() ?: continue
            val symbol = lookup.find("cblas_sgemm").orElse(null) ?: continue
            val handle = runCatching { linker.downcallHandle(symbol, SGEMM) }.getOrNull() ?: continue
            sgemmHandle = handle
            libraryDescription = candidate
            return
        }
    }

    // void cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, const float* A, lda,
    //                  const float* B, ldb, beta, float* C, ldc)
    private val SGEMM: FunctionDescriptor = FunctionDescriptor.ofVoid(
        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
        ValueLayout.JAVA_FLOAT,
        ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
        ValueLayout.JAVA_FLOAT,
        ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
    )

    /**
     * `C = op(A) · op(B)` over off-heap segments, row-major, `beta = 0` (C is overwritten, never read).
     * `op(A)` is `m × k` and `op(B)` is `k × n`: with [transA] false A is stored `m × k` with leading
     * dimension [lda] ≥ k, with it true A is stored `k × m` and `lda` ≥ m; likewise B is `k × n`
     * (`ldb` ≥ n) or, transposed, `n × k` (`ldb` ≥ k) — the shape a Linear layer stores its `[out, in]`
     * weight in, so `x · Wᵀ` is `sgemm(false, true, …, ldb = k)` with no copy of the weight. Offsets
     * are in bytes. Returns false when no BLAS is available or the call threw.
     */
    fun sgemm(
        transA: Boolean, transB: Boolean,
        m: Int, n: Int, k: Int,
        a: MemorySegment, aByteOffset: Long, lda: Int,
        b: MemorySegment, bByteOffset: Long, ldb: Int,
        c: MemorySegment, cByteOffset: Long, ldc: Int,
    ): Boolean {
        val handle = sgemmHandle ?: return false
        if (m <= 0 || n <= 0) return true
        if (k <= 0) {
            c.asSlice(cByteOffset).fill(0)
            return true
        }
        return runCatching {
            handle.invoke(
                CBLAS_ROW_MAJOR,
                if (transA) CBLAS_TRANS else CBLAS_NO_TRANS,
                if (transB) CBLAS_TRANS else CBLAS_NO_TRANS,
                m, n, k,
                1.0f,
                a.asSlice(aByteOffset), lda,
                b.asSlice(bByteOffset), ldb,
                0.0f,
                c.asSlice(cByteOffset), ldc,
            )
            true
        }.getOrElse { false }
    }

    /**
     * The heap face of [sgemm] for `Fp32MatmulKernel`-shaped operands: `C(m, n) = A(m, k) · B(k, n)`
     * with element offsets and row strides in floats. A, B and the `m × n` block of C are staged
     * off-heap for the call — the weight-sized copy the native heap kernel pays too — so callers keep
     * this for products large enough to amortise it. Only the `m × n` block of `out` is written.
     */
    fun sgemmRowMajorNN(
        m: Int, n: Int, k: Int,
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
    ): Boolean {
        if (sgemmHandle == null) return false
        if (m <= 0 || n <= 0) return true
        val aFloats = if (k == 0) 0 else (m - 1) * aStride + k
        val bFloats = if (k == 0) 0 else (k - 1) * bStride + n
        Arena.ofConfined().use { arena ->
            val aSeg = arena.allocate(maxOf(aFloats, 1).toLong() * Float.SIZE_BYTES, 64)
            val bSeg = arena.allocate(maxOf(bFloats, 1).toLong() * Float.SIZE_BYTES, 64)
            val cSeg = arena.allocate(m.toLong() * n * Float.SIZE_BYTES, 64)
            if (aFloats > 0) MemorySegment.copy(a, aOffset, aSeg, ValueLayout.JAVA_FLOAT, 0L, aFloats)
            if (bFloats > 0) MemorySegment.copy(b, bOffset, bSeg, ValueLayout.JAVA_FLOAT, 0L, bFloats)
            val ok = sgemm(false, false, m, n, k, aSeg, 0L, aStride, bSeg, 0L, bStride, cSeg, 0L, n)
            if (!ok) return false
            if (outStride == n) {
                MemorySegment.copy(cSeg, ValueLayout.JAVA_FLOAT, 0L, out, outOffset, m * n)
            } else {
                for (i in 0 until m) {
                    MemorySegment.copy(cSeg, ValueLayout.JAVA_FLOAT, i.toLong() * n * Float.SIZE_BYTES, out, outOffset + i * outStride, n)
                }
            }
            return true
        }
    }
}
