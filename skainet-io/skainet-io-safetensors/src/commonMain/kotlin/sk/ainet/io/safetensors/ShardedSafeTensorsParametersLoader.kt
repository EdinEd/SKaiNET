package sk.ainet.io.safetensors

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import kotlin.reflect.KClass

/**
 * [ParametersLoader] for sharded SafeTensors checkpoints — the
 * `model.safetensors.index.json` + `model-NNNNN-of-NNNNN.safetensors` layout
 * HuggingFace uses for multi-file models (#1246).
 *
 * Sharded counterpart of [SafeTensorsParametersLoader]: the per-tensor
 * materialization and narrow-float policy handling are shared verbatim via
 * [SafeTensorsMaterializer], so both loaders produce identical tensors for
 * identical bytes.
 *
 * Reading rides [StreamingShardedSafeTensorsReader.openFromIndex]: every
 * shard file is opened eagerly and its handle stays open until the load
 * completes (the reader is closed when `load` returns). Tensors are
 * delivered in name-sorted order, regardless of which shard holds them.
 *
 * Unlike the single-file loader, unsupported-dtype failures are raised
 * *before* any tensor is delivered: a pre-scan over the (filtered) index
 * aggregates every tensor whose SafeTensors dtype cannot materialize into
 * the requested [DType] and throws one [IllegalArgumentException] listing
 * them all — a 40-shard load should not die on tensor 900 (mirrors the
 * GGUF loader's fail-fast contract, #919).
 *
 * Two ways to name the checkpoint: the [indexPath] constructor resolves
 * the shards beside the index file through `openRandomAccessSource`,
 * which is how a JVM, Android or Kotlin/Native host loads; the
 * `(indexJson, shardSource)` constructor takes the index as text and a
 * provider that opens each shard by file name — the form for a browser
 * or Wasm host, where there is no file system and a shard is a `Blob`
 * (`JsBlobRandomAccessSource`) or fetched bytes
 * (`ByteArrayRandomAccessSource`). Both deliver identical tensors for
 * identical bytes.
 *
 * @param indexPath Path to `model.safetensors.index.json`.
 * @param onProgress Optional progress callback (current, total, tensorName);
 *   `total` reflects the filtered tensor count.
 * @param bf16Policy How to handle `BFLOAT16` tensors (see
 *   [SafeTensorsParametersLoader]).
 * @param fp16Policy How to handle `FLOAT16` tensors.
 * @param allowPartial If true, missing shards are tolerated and only tensors
 *   from present shards are delivered; if false (default), a missing shard
 *   throws [SafeTensorsShardException.IncompleteShard].
 * @param tensorFilter Optional predicate over [ShardedTensorInfo]; tensors
 *   for which it returns false are neither materialized nor delivered, and
 *   are exempt from the fail-fast dtype pre-scan. This is the hook for
 *   family-side skip policy (size guards, name allowlists) — dtype/policy
 *   handling stays engine-side. `null` loads everything.
 */
public class ShardedSafeTensorsParametersLoader private constructor(
    private val openReader: suspend (allowPartial: Boolean) -> StreamingShardedSafeTensorsReader,
    private val onProgress: (current: Long, total: Long, message: String?) -> Unit,
    private val bf16Policy: Bf16LoadPolicy,
    private val fp16Policy: NarrowFloatLoadPolicy,
    private val allowPartial: Boolean,
    private val tensorFilter: ((ShardedTensorInfo) -> Boolean)?,
    private val weightForm: WeightForm?,
    private val weightFormFor: ((String) -> WeightForm?)?,
) : ParametersLoader {

    /** The checkpoint by the path of its index file; shards are resolved beside it. */
    public constructor(
        indexPath: String,
        onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
        bf16Policy: Bf16LoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
        fp16Policy: NarrowFloatLoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
        allowPartial: Boolean = false,
        tensorFilter: ((ShardedTensorInfo) -> Boolean)? = null,
        weightForm: WeightForm? = null,
        weightFormFor: ((String) -> WeightForm?)? = null,
    ) : this(
        openReader = { partial -> StreamingShardedSafeTensorsReader.openFromIndex(indexPath, partial) },
        onProgress = onProgress,
        bf16Policy = bf16Policy,
        fp16Policy = fp16Policy,
        allowPartial = allowPartial,
        tensorFilter = tensorFilter,
        weightForm = weightForm,
        weightFormFor = weightFormFor,
    )

    /**
     * The checkpoint by the content of its index file and a provider that opens each shard by
     * file name (`model-00001-of-00003.safetensors`, …), answering `null` for a missing one. The
     * provider may suspend — a browser opens a `Blob` per shard inside it — and is asked once per
     * shard; the sources it returns are closed when `load` returns. See
     * [StreamingShardedSafeTensorsReader.openFromIndexContent].
     */
    public constructor(
        indexJson: String,
        shardSource: suspend (shardFilename: String) -> RandomAccessSource?,
        onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
        bf16Policy: Bf16LoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
        fp16Policy: NarrowFloatLoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
        allowPartial: Boolean = false,
        tensorFilter: ((ShardedTensorInfo) -> Boolean)? = null,
        weightForm: WeightForm? = null,
        weightFormFor: ((String) -> WeightForm?)? = null,
    ) : this(
        openReader = { partial -> StreamingShardedSafeTensorsReader.openFromIndexContent(indexJson, shardSource, partial) },
        onProgress = onProgress,
        bf16Policy = bf16Policy,
        fp16Policy = fp16Policy,
        allowPartial = allowPartial,
        tensorFilter = tensorFilter,
        weightForm = weightForm,
        weightFormFor = weightFormFor,
    )

    init {
        weightForm?.let { SafeTensorsWeightForms.validate(it, "weightForm") }
    }

    /** The form [tensorName] loads under: [weightFormFor] first, then the uniform [weightForm]. */
    private fun formFor(tensorName: String): WeightForm? {
        val perTensor = weightFormFor?.invoke(tensorName) ?: return weightForm
        SafeTensorsWeightForms.validate(perTensor, "weightFormFor('$tensorName')")
        return perTensor
    }

    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        openReader(allowPartial).use { reader ->
            val tensors = tensorFilter?.let { filter -> reader.tensors.filter(filter) }
                ?: reader.tensors

            failFastOnUnsupportedTensorTypes(tensors, dtype)

            val total = tensors.size.toLong()
            var current = 0L

            for (tensorInfo in tensors) {
                val bytes = reader.loadTensorData(tensorInfo)
                val shape = Shape(*tensorInfo.shape.map { it.toInt() }.toIntArray())

                val tensor: Tensor<T, V> = SafeTensorsMaterializer.materialize(
                    ctx = ctx,
                    dtype = dtype,
                    name = tensorInfo.name,
                    dataType = tensorInfo.dataType,
                    rawDtype = tensorInfo.dtype,
                    shape = shape,
                    bytes = bytes,
                    bf16Policy = bf16Policy,
                    fp16Policy = fp16Policy,
                    form = formFor(tensorInfo.name),
                )

                onTensorLoaded(tensorInfo.name, tensor)
                current++
                onProgress(current, total, tensorInfo.name)
            }
        }
    }

    public companion object {

        /**
         * Constructs a [ShardedSafeTensorsParametersLoader] from a generalised
         * [DTypePolicy]. Signature parity with
         * [SafeTensorsParametersLoader.withPolicy]; the policy → behaviour
         * mapping is identical (documented there).
         */
        public fun withPolicy(
            indexPath: String,
            policy: DTypePolicy,
            onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
            allowPartial: Boolean = false,
            tensorFilter: ((ShardedTensorInfo) -> Boolean)? = null,
        ): ShardedSafeTensorsParametersLoader = ShardedSafeTensorsParametersLoader(
            indexPath = indexPath,
            onProgress = onProgress,
            bf16Policy = SafeTensorsMaterializer.mapPolicyToBf16(policy),
            fp16Policy = SafeTensorsMaterializer.mapPolicyToFp16(policy),
            allowPartial = allowPartial,
            tensorFilter = tensorFilter,
        )

        /** [withPolicy] for the `(indexJson, shardSource)` form. */
        public fun withPolicy(
            indexJson: String,
            shardSource: suspend (shardFilename: String) -> RandomAccessSource?,
            policy: DTypePolicy,
            onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
            allowPartial: Boolean = false,
            tensorFilter: ((ShardedTensorInfo) -> Boolean)? = null,
        ): ShardedSafeTensorsParametersLoader = ShardedSafeTensorsParametersLoader(
            indexJson = indexJson,
            shardSource = shardSource,
            onProgress = onProgress,
            bf16Policy = SafeTensorsMaterializer.mapPolicyToBf16(policy),
            fp16Policy = SafeTensorsMaterializer.mapPolicyToFp16(policy),
            allowPartial = allowPartial,
            tensorFilter = tensorFilter,
        )

        /**
         * Pre-scan [tensors] and throw one aggregated error naming every
         * tensor whose dtype cannot materialize into [dtype], before any
         * tensor is delivered.
         */
        internal fun failFastOnUnsupportedTensorTypes(
            tensors: List<ShardedTensorInfo>,
            dtype: KClass<out DType>,
        ) {
            val mismatches = tensors.filter { SafeTensorsMaterializer.requiredDType(it.dataType) != dtype }
            if (mismatches.isNotEmpty()) {
                val listing = mismatches.joinToString(separator = "\n") {
                    "  - '${it.name}' (${it.dtype}, shard ${it.shardLocation}) requires " +
                        "${SafeTensorsMaterializer.requiredDType(it.dataType).simpleName}"
                }
                throw IllegalArgumentException(
                    "ShardedSafeTensorsParametersLoader: ${mismatches.size} of ${tensors.size} tensors " +
                        "cannot materialize as ${dtype.simpleName}; no tensors were delivered. " +
                        "Use tensorFilter to exclude them or load with the required dtype:\n$listing",
                )
            }
        }
    }
}
