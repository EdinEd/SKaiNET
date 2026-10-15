package sk.ainet.lang.tensor.ops

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** #1343: the single definition of the attention mask rules. */
class AttentionMaskTest {

    @Test
    fun kind_names_the_rules_in_force() {
        assertEquals("none", AttentionMask.kindOf(causal = false, window = 0, hasLengths = false))
        assertEquals("causal", AttentionMask.kindOf(causal = true, window = 0, hasLengths = false))
        assertEquals("slidingWindow", AttentionMask.kindOf(causal = false, window = 4, hasLengths = false))
        assertEquals("slidingWindow", AttentionMask.kindOf(causal = true, window = 4, hasLengths = false))
        assertEquals("keyPadding", AttentionMask.kindOf(causal = false, window = 0, hasLengths = true))
        assertEquals("combined", AttentionMask.kindOf(causal = true, window = 0, hasLengths = true))
        assertEquals("combined", AttentionMask.kindOf(causal = false, window = 4, hasLengths = true))
        assertTrue(AttentionMask.hasLengths("keyPadding") && AttentionMask.hasLengths("combined"))
        assertTrue(!AttentionMask.hasLengths("causal") && !AttentionMask.hasLengths(null))
    }

    @Test
    fun causal_aligns_the_last_query_with_the_last_key() {
        // Prefill: diagonal.
        assertEquals(0 until 3, AttentionMask.keyRange(2, 4, 4, causal = true, window = 0, rightContext = 0, length = 4))
        // KV-cache decode: one query, the whole cache.
        assertEquals(0 until 9, AttentionMask.keyRange(0, 1, 9, causal = true, window = 0, rightContext = 0, length = 9))
        // Chunked prefill: 2 queries over 6 keys attend up to positions 4 and 5.
        assertEquals(0 until 5, AttentionMask.keyRange(0, 2, 6, causal = true, window = 0, rightContext = 0, length = 6))
        assertEquals(0 until 6, AttentionMask.keyRange(1, 2, 6, causal = true, window = 0, rightContext = 0, length = 6))
    }

    @Test
    fun window_band_matches_the_transformer_formula() {
        // transformers MultiHeadAttention.buildSlidingWindowMask: absQ = seqKV - seqQ + qi,
        // allowed ki in [absQ - window + 1, absQ + rightContext].
        for ((seqQ, seqKV) in listOf(8 to 8, 1 to 20, 5 to 12)) for (window in listOf(1, 3, 16)) for (rightContext in listOf(0, 2)) {
            val got = AttentionMask.additive(1, seqQ, seqKV, causal = false, window = window, rightContext = rightContext, lengths = null)
            val expected = FloatArray(seqQ * seqKV)
            for (qi in 0 until seqQ) {
                val absQ = seqKV - seqQ + qi
                for (ki in 0 until seqKV) expected[qi * seqKV + ki] = if (ki in (absQ - window + 1)..(absQ + rightContext)) 0f else -1.0e30f
            }
            assertContentEquals(expected, got, "seqQ=$seqQ seqKV=$seqKV window=$window rightContext=$rightContext")
        }
    }

    @Test
    fun padding_intersects_with_the_positional_rules() {
        val lengths = intArrayOf(3, 0, 5)
        val m = AttentionMask.additive(3, 4, 5, causal = true, window = 0, rightContext = 0, lengths = lengths)
        fun attended(b: Int, qi: Int, ki: Int) = m[(b * 4 + qi) * 5 + ki] == 0f
        // Row 0: keys < 3 and ki <= absQ (absQ = 1 + qi).
        assertTrue(attended(0, 0, 0) && attended(0, 0, 1) && !attended(0, 0, 2))
        assertTrue(attended(0, 3, 2) && !attended(0, 3, 3))
        // Row 1: fully padded → every key masked, but finite. (Threshold, not equality: Kotlin/JS
        // keeps a Float literal at double precision while a FloatArray element is float32.)
        for (qi in 0 until 4) for (ki in 0 until 5) {
            val v = m[(1 * 4 + qi) * 5 + ki]
            assertTrue(v.isFinite() && v <= -1.0e29f, "row 1 key $ki of query $qi must be masked, got $v")
        }
        // Row 2: no padding → plain causal with offset 1.
        assertTrue(attended(2, 0, 1) && !attended(2, 0, 2) && attended(2, 3, 4))
        // keyRange agrees with attends, including an empty range.
        assertTrue(AttentionMask.keyRange(0, 4, 5, true, 0, 0, 0).isEmpty())
        for (qi in 0 until 4) for (ki in 0 until 5) {
            assertEquals(attended(0, qi, ki), AttentionMask.attends(qi, ki, 4, 5, causal = true, window = 0, rightContext = 0, length = 3))
        }
    }

    @Test
    fun argument_contract() {
        assertFailsWith<IllegalArgumentException> { AttentionMask.validate(window = -1, rightContext = 0) }
        assertFailsWith<IllegalArgumentException> { AttentionMask.validate(window = 0, rightContext = 1) }
        assertFailsWith<IllegalArgumentException> { AttentionMask.additive(2, 2, 2, false, 0, 0, intArrayOf(1)) }
        AttentionMask.validate(window = 4, rightContext = 2)
    }
}
