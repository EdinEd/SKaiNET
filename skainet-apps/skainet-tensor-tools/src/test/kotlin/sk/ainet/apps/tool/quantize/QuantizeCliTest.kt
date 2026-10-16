package sk.ainet.apps.tool.quantize

import kotlinx.coroutines.runBlocking
import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.io.safetensors.SafeTensorsWriter
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import java.nio.file.Files
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** SKEEP-007 §T2: the offline converter writes a GGUF the streaming loader serves packed, names and shapes intact. */
class QuantizeCliTest {
    private fun pseudo(n: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 2001 - 1000) / 500f
        }
    }

    @Test
    fun `safetensors to q8_0 gguf round trip`() {
        val w = pseudo(6 * 64, 1)
        val emb = pseudo(10 * 32, 2)
        val bias = pseudo(6, 3)
        val buffer = Buffer()
        SafeTensorsWriter.write(buffer) {
            tensorF32("layers.0.weight", intArrayOf(6, 64), w)
            tensorF32("tok_embeddings.weight", intArrayOf(10, 32), emb)
            tensorF32("layers.0.bias", intArrayOf(6), bias)
        }
        val input = Files.createTempFile("quantize-in", ".safetensors").toFile().also { it.deleteOnExit() }
        input.writeBytes(buffer.readByteArray())
        val output = Files.createTempFile("quantize-out", ".gguf").toFile().also { it.deleteOnExit() }

        val report = quantizeSafeTensorsToGguf(input.path, output.path, TensorEncoding.Q8_0, keepPrefixes = listOf("tok_embeddings"))
        assertEquals(listOf("layers.0.weight"), report.quantized)
        assertEquals(setOf("tok_embeddings.weight", "layers.0.bias"), report.kept.toSet())
        assertTrue(report.outputBytes > 0)

        val ctx = DefaultDataExecutionContext()
        val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(sourceProvider = { JvmRandomAccessSource.open(output.path) }, keepF16Native = true)
                .load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        assertEquals(setOf("layers.0.weight", "tok_embeddings.weight", "layers.0.bias"), loaded.keys)

        val packed = assertIs<Q8_0BlockTensorData>(loaded.getValue("layers.0.weight").data)
        assertEquals(listOf(6, 64), packed.shape.dimensions.toList())
        val back = packed.toFloatArray()
        val amax = w.maxOf { abs(it) }
        for (i in w.indices) assertTrue(abs(w[i] - back[i]) <= amax / 127f + 1e-4f, "w[$i]")

        val kept = assertIs<Fp16DenseTensorData>(loaded.getValue("tok_embeddings.weight").data)
        assertEquals(listOf(10, 32), kept.shape.dimensions.toList())
        val embBack = kept.copyToFloatArray()
        for (i in emb.indices) assertTrue(abs(emb[i] - embBack[i]) <= abs(emb[i]) * 1e-3f + 1e-3f, "emb[$i]")

        val biasBack = loaded.getValue("layers.0.bias").data.copyToFloatArray()
        for (i in bias.indices) assertTrue(abs(bias[i] - biasBack[i]) <= abs(bias[i]) * 1e-3f + 1e-3f, "bias[$i]")
    }
}
