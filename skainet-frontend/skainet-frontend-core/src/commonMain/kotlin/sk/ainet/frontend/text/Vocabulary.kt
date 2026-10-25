package sk.ainet.frontend.text

/**
 * Token ↔ id tables of a tokenizer (FE-C-07). Required on every [Tokenizer]: stop-token
 * resolution and chat templates need [tokenToId]; every engine implementation holds the tables.
 */
public interface Vocabulary {
    /** Number of ids, including added and special tokens (`get_vocab_size(with_added_tokens=True)`). */
    public val size: Int

    /** The id of a vocabulary piece (surface form as stored, e.g. `Ġhello`, `▁hello`, `##ing`), or `null`. */
    public fun tokenToId(piece: String): Int?

    /** The stored piece for [id], or `null` when the id is out of range or a gap. */
    public fun idToToken(id: Int): String?
}

/** A [Vocabulary] over an id-indexed array of pieces. */
public class ArrayVocabulary(pieces: List<String>) : Vocabulary {
    private val idToPiece: Array<String> = pieces.toTypedArray()
    private val pieceToId: Map<String, Int> = HashMap<String, Int>(pieces.size * 2).also { m ->
        // first occurrence wins, matching how llama.cpp and `tokenizers` build their maps: inserting from the last id
        // down, the lowest id of a duplicated piece is written last, at one map operation per piece
        for (i in idToPiece.indices.reversed()) m[idToPiece[i]] = i
    }

    override val size: Int get() = idToPiece.size
    override fun tokenToId(piece: String): Int? = pieceToId[piece]
    override fun idToToken(id: Int): String? = idToPiece.getOrNull(id)
}

/** A [Vocabulary] over a piece → id map (WordPiece `vocab.txt`, `tokenizer.json#model.vocab`). */
public class MapVocabulary(private val pieceToId: Map<String, Int>) : Vocabulary {
    private val idToPiece: Map<Int, String> = pieceToId.entries.associate { (k, v) -> v to k }
    override val size: Int = (pieceToId.values.maxOrNull() ?: -1) + 1
    override fun tokenToId(piece: String): Int? = pieceToId[piece]
    override fun idToToken(id: Int): String? = idToPiece[id]
}
