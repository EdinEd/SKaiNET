package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** SKEEP-007 §T0.3: narrow-float dense tables serve rows on demand, decoded like `get`. */
class NarrowFloatRowDequantTest {
    private val rows = 5
    private val cols = 7
    private val values = FloatArray(rows * cols) { (it - 17) * 0.375f }

    @Test
    fun `fp16 table is a RowDequantSource whose rows match element access`() {
        val data = Fp16DenseTensorData.fromFloatArray(Shape(rows, cols), values)
        assertTrue(data is RowDequantSource)
        for (r in 0 until rows) {
            val expected = FloatArray(cols) { c -> data[r, c] }
            assertContentEquals(expected, (data as RowDequantSource).dequantRow(r), "row $r")
        }
    }

    @Test
    fun `bf16 table rows match element access`() {
        val data = Bf16DenseTensorData.fromFloatArray(Shape(rows, cols), values)
        for (r in 0 until rows) {
            val expected = FloatArray(cols) { c -> data[r, c] }
            assertContentEquals(expected, data.dequantRow(r), "row $r")
        }
    }

    @Test
    fun `out-of-range row and non-table shapes are refused`() {
        val table = Fp16DenseTensorData.fromFloatArray(Shape(rows, cols), values)
        assertFailsWith<IllegalArgumentException> { table.dequantRow(rows) }
        assertFailsWith<IllegalArgumentException> { table.dequantRow(-1) }
        val cube = Fp16DenseTensorData.fromFloatArray(Shape(1, rows, cols), values)
        assertFailsWith<IllegalArgumentException> { cube.dequantRow(0) }
    }
}
