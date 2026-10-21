package sk.ainet.data.media

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BoundingBoxTest {

    @Test
    fun iouOfIdenticalBoxesIsOne() {
        val a = BoundingBox(0f, 0f, 10f, 10f)
        assertEquals(1f, a.iou(a))
    }

    @Test
    fun iouOfDisjointBoxesIsZero() {
        assertEquals(0f, BoundingBox(0f, 0f, 1f, 1f).iou(BoundingBox(2f, 2f, 3f, 3f)))
    }

    @Test
    fun iouOfHalfOverlap() {
        // overlap 5x10 = 50, union 100 + 100 - 50 = 150
        val iou = BoundingBox(0f, 0f, 10f, 10f).iou(BoundingBox(5f, 0f, 15f, 10f))
        assertEquals(1f / 3f, iou, 1e-6f)
    }

    @Test
    fun fromCenter() {
        assertEquals(BoundingBox(5f, 10f, 15f, 30f), BoundingBox.fromCenter(10f, 20f, 10f, 20f))
    }

    @Test
    fun nmsKeepsTheBestOfAnOverlappingCluster() {
        val kept = nonMaxSuppression(
            listOf(
                ScoredBox(BoundingBox(0f, 0f, 10f, 10f), classId = 0, score = 0.6f),
                ScoredBox(BoundingBox(1f, 1f, 11f, 11f), classId = 0, score = 0.9f),
                ScoredBox(BoundingBox(50f, 50f, 60f, 60f), classId = 0, score = 0.5f)
            ),
            iouThreshold = 0.45f
        )
        assertEquals(listOf(0.9f, 0.5f), kept.map { it.score })
    }

    @Test
    fun nmsIsClassAwareByDefault() {
        val same = BoundingBox(0f, 0f, 10f, 10f)
        val boxes = listOf(ScoredBox(same, 0, 0.9f), ScoredBox(same, 1, 0.8f))
        assertEquals(2, nonMaxSuppression(boxes, iouThreshold = 0.45f).size)
        assertEquals(1, nonMaxSuppression(boxes, iouThreshold = 0.45f, classAware = false).size)
    }

    @Test
    fun nmsHonoursMaxDetectionsAndEmptyInput() {
        val boxes = (0 until 5).map { ScoredBox(BoundingBox(it * 20f, 0f, it * 20f + 10f, 10f), 0, 0.1f * it) }
        assertEquals(2, nonMaxSuppression(boxes, 0.45f, maxDetections = 2).size)
        assertTrue(nonMaxSuppression(emptyList(), 0.45f).isEmpty())
    }
}
