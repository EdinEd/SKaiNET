package sk.ainet.data.media

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class LetterboxTest {

    @Test
    fun landscapeIsPaddedTopAndBottom() {
        val lb = Letterbox.fit(width = 1280, height = 720, targetSize = 640)
        assertEquals(0.5f, lb.scale)
        assertEquals(640, lb.resizedWidth)
        assertEquals(360, lb.resizedHeight)
        assertEquals(0, lb.padLeft)
        assertEquals(140, lb.padTop)
    }

    @Test
    fun portraitIsPaddedLeftAndRight() {
        val lb = Letterbox.fit(width = 300, height = 600, targetSize = 640)
        assertEquals(320, lb.resizedWidth)
        assertEquals(640, lb.resizedHeight)
        assertEquals(160, lb.padLeft)
        assertEquals(0, lb.padTop)
    }

    @Test
    fun toSourceInvertsTheFit() {
        val lb = Letterbox.fit(width = 1280, height = 720, targetSize = 640)
        val onCanvas = BoundingBox(x1 = 50f, y1 = 140f + 25f, x2 = 150f, y2 = 140f + 75f)
        assertEquals(BoundingBox(100f, 50f, 300f, 150f), lb.toSource(onCanvas))
    }

    @Test
    fun toSourceClampsToTheImage() {
        val lb = Letterbox.fit(width = 1280, height = 720, targetSize = 640)
        val b = lb.toSource(BoundingBox(-10f, 0f, 700f, 640f))
        assertEquals(BoundingBox(0f, 0f, 1280f, 720f), b)
    }

    @Test
    fun fillNchwPadsAndScalesValues() {
        // 2x1 source: one red pixel, one blue pixel, on a 4x4 canvas → resized 4x2, padded 1 row top and bottom
        val rgb = byteArrayOf(255.toByte(), 0, 0, 0, 0, 255.toByte())
        val lb = Letterbox.fit(width = 2, height = 1, targetSize = 4)
        assertEquals(1, lb.padTop)
        val out = lb.fillNchw(rgb, padValue = 0)
        val plane = 16
        assertEquals(3 * plane, out.size)
        // padding rows stay at the pad value
        for (x in 0 until 4) {
            assertEquals(0f, out[x])                  // R, row 0
            assertEquals(0f, out[3 * 4 + x])          // R, row 3
        }
        // left edge is pure red, right edge pure blue
        assertEquals(1f, out[1 * 4 + 0])              // R at (0,1)
        assertEquals(0f, out[2 * plane + 1 * 4 + 0])  // B at (0,1)
        assertEquals(0f, out[1 * 4 + 3])              // R at (3,1)
        assertEquals(1f, out[2 * plane + 1 * 4 + 3])  // B at (3,1)
    }

    @Test
    fun fillNchwRejectsAWrongBufferSize() {
        val lb = Letterbox.fit(width = 2, height = 2, targetSize = 4)
        assertFailsWith<IllegalArgumentException> { lb.fillNchw(ByteArray(5)) }
    }
}
