package sk.ainet.frontend.text

import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.InvertibleFrontEnd

/**
 * Text ↔ token ids (SKEEP-006 §Contract, FE-C-05..FE-C-08).
 *
 * `Tokenizer` is a distinct sub-interface of [InvertibleFrontEnd]: callers want [encode],
 * [decode], streaming [decodeToken] and [vocabulary] lookups; `FrontEnd<String, IntArray>` is the
 * plumbing that makes it resolvable, not the API people call.
 *
 * Compatibility with `sk.ainet.io.tokenizer.Tokenizer` at 0.57: the one-argument [encode] and
 * [decode] keep their 0.57 meaning — [encode] never adds `bos`/`eos` and [decode] renders special
 * tokens as text. The two-argument overloads apply the checkpoint's [specialTokens] policy
 * (`addSpecialTokens = true` prepends/appends per `addBos`/`addEos`; `skipSpecialTokens = true`
 * drops every declared special id before decoding).
 */
public interface Tokenizer : InvertibleFrontEnd<String, IntArray> {
    public val vocabulary: Vocabulary
    public val specialTokens: SpecialTokens

    /** Encode without adding `bos`/`eos` (0.57 semantics). Special-token *strings* inside the text still encode atomically. */
    public fun encode(text: String): IntArray

    /** Encode; with [addSpecialTokens] the checkpoint's `addBos`/`addEos` policy is applied. */
    public fun encode(text: String, addSpecialTokens: Boolean): IntArray =
        if (addSpecialTokens) specialTokens.surround(encode(text)) else encode(text)

    /** Decode, rendering special tokens as their text (0.57 semantics). */
    public fun decode(ids: IntArray): String

    /** Decode; with [skipSpecialTokens] every id in [SpecialTokens.allSpecialIds] is dropped first. */
    public fun decode(ids: IntArray, skipSpecialTokens: Boolean): String {
        if (!skipSpecialTokens) return decode(ids)
        val specials = specialTokens.allSpecialIds
        if (specials.isEmpty()) return decode(ids)
        val kept = ids.filter { it !in specials }
        return decode(IntArray(kept.size) { kept[it] })
    }

    /**
     * Decode a single token to its surface piece for **streaming** generation.
     *
     * Unlike [decode], this must NOT apply any sequence-level leading-space normalisation: each
     * piece keeps its own leading word-boundary space, so concatenating a stream of per-token
     * pieces reconstructs spacing (llama.cpp `token_to_piece` semantics). Implementations whose
     * [decode] strips a leading space override this.
     */
    public fun decodeToken(id: Int): String = decode(intArrayOf(id))

    override fun process(input: String): IntArray = encode(input)
    override fun invert(output: IntArray): String = decode(output)

    // ---- source compatibility with sk.ainet.io.tokenizer.Tokenizer (0.57) ----

    public val vocabSize: Int get() = vocabulary.size
    public val bosTokenId: Int? get() = specialTokens.bos
    public val eosTokenId: Int? get() = specialTokens.eos

    override val spec: FrontEndSpec get() = FrontEndSpec.Text(algorithm = "unknown")
}
