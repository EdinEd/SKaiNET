package sk.ainet.backend.api.kernel

import java.lang.foreign.MemorySegment

/**
 * [Fp32MatmulWtKernel] with the `[out, in]` FP32 weight supplied as an off-heap
 * `java.lang.foreign.MemorySegment` instead of a heap `FloatArray`. JVM-only sibling, for the same
 * reason [Q4KMemSegMatmulKernel] exists: a downcall cannot read heap memory on JDK 21, so the heap
 * face copies the whole weight on every call — and the weight is the large operand. A weight that
 * already lives in a segment (loaded off-heap, or made resident once by the ops layer) is passed
 * through with no copy at all; only the `m × k` activation and the `m × n` output are staged.
 *
 * Same contract as [Fp32MatmulWtKernel]: contiguous operands, `out` fully overwritten, products
 * accumulated in ascending `p`. The caller owns the segment's arena and keeps it alive for the
 * duration of the call; the kernel retains nothing past return.
 */
public interface Fp32MemSegMatmulWtKernel {
    public fun matmulWt(
        x: FloatArray, xOffset: Int,
        weight: MemorySegment, weightByteOffset: Long,
        out: FloatArray, outOffset: Int,
        m: Int, n: Int, k: Int,
    )
}
