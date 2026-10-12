package sk.ainet.frontend.tokenizer

/**
 * One `tokenizer.json#added_tokens` entry. Every entry is matched atomically before
 * pre-tokenization, special or not (SKaiNET #1324); [special] only decides whether
 * `decode(skipSpecialTokens = true)` drops it.
 */
public data class AddedToken(
    val content: String,
    val id: Int,
    val special: Boolean = true,
    val lstrip: Boolean = false,
    val rstrip: Boolean = false,
    val singleWord: Boolean = false,
    val normalized: Boolean = false,
)
