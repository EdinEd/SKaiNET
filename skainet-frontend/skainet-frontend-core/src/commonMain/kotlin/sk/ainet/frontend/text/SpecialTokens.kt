package sk.ainet.frontend.text

/**
 * Checkpoint-declared token policy (SKEEP-006 §Contract). Replaces the bos/eos/add_bos reads
 * scattered across both repositories. [bos] and [eos] stay nullable (FE-C-06): the `-1` sentinel
 * is a defect source and is not reintroduced.
 *
 * @property bos beginning-of-sequence id, if the checkpoint declares one
 * @property eos end-of-sequence id, if the checkpoint declares one
 * @property pad padding id, if declared
 * @property unk unknown-token id, if declared
 * @property addBos whether `encode(text, addSpecialTokens = true)` prepends [bos] (GGUF
 *   `tokenizer.ggml.add_bos_token`, `tokenizer_config.json#add_bos_token`, or the post-processor)
 * @property addEos whether `encode(text, addSpecialTokens = true)` appends [eos]
 * @property extraStop additional end-of-turn ids the family treats as stop, e.g. Gemma
 *   `<end_of_turn>`, Qwen `<|im_end|>`
 * @property declared every id the checkpoint marks special (GGUF CONTROL / USER_DEFINED types,
 *   `added_tokens[].special = true`); what `decode(ids, skipSpecialTokens = true)` drops
 */
public data class SpecialTokens(
    val bos: Int? = null,
    val eos: Int? = null,
    val pad: Int? = null,
    val unk: Int? = null,
    val addBos: Boolean = false,
    val addEos: Boolean = false,
    val extraStop: Set<Int> = emptySet(),
    val declared: Set<Int> = emptySet(),
) {
    /** Ids that end generation: [eos] plus [extraStop]. */
    public val stopIds: Set<Int> get() = setOfNotNull(eos) + extraStop

    /** Every special id: [declared] plus the named ones plus [extraStop]. */
    public val allSpecialIds: Set<Int> get() = declared + setOfNotNull(bos, eos, pad, unk) + extraStop

    /** Prepend [bos] / append [eos] to [ids] according to [addBos] / [addEos]. */
    public fun surround(ids: IntArray): IntArray {
        val b = bos.takeIf { addBos }
        val e = eos.takeIf { addEos }
        if (b == null && e == null) return ids
        val out = IntArray(ids.size + (if (b != null) 1 else 0) + (if (e != null) 1 else 0))
        var p = 0
        if (b != null) out[p++] = b
        ids.copyInto(out, p); p += ids.size
        if (e != null) out[p] = e
        return out
    }

    public companion object {
        /** No special tokens at all. */
        public val NONE: SpecialTokens = SpecialTokens()
    }
}
