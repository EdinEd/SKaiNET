@file:OptIn(kotlin.js.ExperimentalWasmJsInterop::class)

package sk.ainet.io.safetensors

import kotlinx.coroutines.test.runTest
import org.khronos.webgl.Uint8Array
import org.w3c.files.Blob
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JsBlobRandomAccessSource
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@JsFun("(n) => new Uint8Array(n)")
private external fun newUint8Array(n: Int): Uint8Array

@JsFun("(a, i, v) => { a[i] = v; }")
private external fun setByte(a: Uint8Array, i: Int, v: Int)

@JsFun("(a) => new Blob([a])")
private external fun toBlob(a: Uint8Array): Blob

/** The fixtures are a few hundred bytes: one interop call per byte is fine here. */
private fun ByteArray.toBlob(): Blob {
    val a = newUint8Array(size)
    for (i in indices) setByte(a, i, this[i].toInt() and 0xFF)
    return toBlob(a)
}

/** A main-thread `FileReaderSync` for the test (see io-core's blob source test): sync XHR on a Blob URL. */
@JsFun(
    """() => {
        globalThis.FileReaderSync = class {
            readAsArrayBuffer(blob) {
                const url = URL.createObjectURL(blob);
                try {
                    const x = new XMLHttpRequest();
                    x.open('GET', url, false);
                    x.overrideMimeType('text/plain; charset=x-user-defined');
                    x.send(null);
                    const s = x.responseText;
                    const out = new Uint8Array(s.length);
                    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
                    return out.buffer;
                } finally {
                    URL.revokeObjectURL(url);
                }
            }
        };
    }""",
)
private external fun installFileReaderSyncPolyfill()

@JsFun("() => { delete globalThis.FileReaderSync; }")
private external fun removeFileReaderSyncPolyfill()

/**
 * A sharded SafeTensors checkpoint from browser Blobs, end to end: the index as text, one
 * `JsBlobRandomAccessSource` per shard through the provider constructor, tensors materialised on
 * the CPU context — the browser's counterpart of the JVM's path-based load (IO parity, WP-D1).
 */
class ShardedSafeTensorsBlobTest {

    private val shards = ShardFixtures.twoShards()
    private val index = ShardFixtures.twoShardIndex(shards)

    private suspend fun loadAll(loader: ShardedSafeTensorsParametersLoader): Pair<List<String>, Map<String, Tensor<FP32, Float>>> {
        val ctx = DirectCpuExecutionContext.create()
        val order = ArrayList<String>()
        val out = HashMap<String, Tensor<FP32, Float>>()
        loader.load<FP32, Float>(ctx, FP32::class) { name, tensor ->
            order += name
            out[name] = tensor
        }
        return order to out
    }

    private fun assertFixture(order: List<String>, tensors: Map<String, Tensor<FP32, Float>>) {
        assertEquals(listOf("alpha.weight", "bravo.weight", "zulu.weight"), order)
        assertContentEquals(ShardFixtures.alphaValues, tensors.getValue("alpha.weight").data.copyToFloatArray())
        assertContentEquals(ShardFixtures.bravoValues, tensors.getValue("bravo.weight").data.copyToFloatArray(), "BF16 dequantised exactly")
        assertContentEquals(ShardFixtures.zuluValues, tensors.getValue("zulu.weight").data.copyToFloatArray())
        assertEquals(listOf(2, 2), tensors.getValue("alpha.weight").shape.dimensions.toList())
    }

    @Test
    fun onTheMainThreadThePreloadedWindowServesTheLoad() = runTest {
        val blobs = shards.mapValues { it.value.toBlob() }
        val asked = ArrayList<String>()
        val loader = ShardedSafeTensorsParametersLoader(index, { name ->
            asked += name
            blobs[name]?.let { JsBlobRandomAccessSource.open(it) }   // default preload covers a small shard whole
        })
        val (order, tensors) = loadAll(loader)
        assertEquals(listOf(ShardFixtures.SHARD_1, ShardFixtures.SHARD_2), asked)
        assertFixture(order, tensors)
    }

    @Test
    fun insideAWorkerNothingIsPreloadedAndTheBlobIsReadBlockingly() = runTest {
        installFileReaderSyncPolyfill()
        try {
            val blobs = shards.mapValues { it.value.toBlob() }
            val sources = ArrayList<JsBlobRandomAccessSource>()
            val loader = ShardedSafeTensorsParametersLoader(index, { name ->
                blobs[name]?.let { JsBlobRandomAccessSource.openSync(it).also(sources::add) }
            })
            val (order, tensors) = loadAll(loader)
            assertTrue(sources.all { it.preloadedSize == 0 && it.supportsSyncReads }, sources.toString())
            assertFixture(order, tensors)
        } finally {
            removeFileReaderSyncPolyfill()
        }
    }

    @Test
    fun aMissingShardBlobIsReportedLikeAMissingFile() = runTest {
        val blobs = shards.mapValues { it.value.toBlob() }
        val loader = ShardedSafeTensorsParametersLoader(index, { name ->
            if (name == ShardFixtures.SHARD_2) null else JsBlobRandomAccessSource.open(blobs.getValue(name))
        })
        var failure: Throwable? = null
        try { loadAll(loader) } catch (e: SafeTensorsShardException.IncompleteShard) { failure = e }
        assertEquals(listOf(ShardFixtures.SHARD_2), (failure as SafeTensorsShardException.IncompleteShard).missingShards)

        val partial = ShardedSafeTensorsParametersLoader(index, { name ->
            if (name == ShardFixtures.SHARD_2) null else JsBlobRandomAccessSource.open(blobs.getValue(name))
        }, allowPartial = true)
        val (order, _) = loadAll(partial)
        assertEquals(listOf("alpha.weight", "bravo.weight"), order)
    }
}
