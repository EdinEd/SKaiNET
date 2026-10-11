package sk.ainet.frontend.tokenizer

/**
 * Rank-based BPE merging over a list of symbols, shared by the byte-level and the Metaspace
 * tokenizers. Rank 0 is the highest-priority merge; the pair with the lowest rank is merged
 * first, repeatedly, until no adjacent pair is in the table (the GPT-2 / `tokenizers` rule, as
 * opposed to SentencePiece's score-wins rule in [SentencePieceTokenizer]).
 */
internal class BpeCore(merges: List<Pair<String, String>>) {
    private val mergeRank: Map<Pair<String, String>, Int> = HashMap<Pair<String, String>, Int>(merges.size * 2).also { m ->
        for (i in merges.indices) if (merges[i] !in m) m[merges[i]] = i
    }

    val size: Int get() = mergeRank.size

    /** Merge [symbols] (code points or byte-level glyphs) in place by rank. */
    fun merge(symbols: MutableList<String>): MutableList<String> {
        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until symbols.size - 1) {
                val rank = mergeRank[symbols[i] to symbols[i + 1]]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if (bestIdx == -1) break
            symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1]
            symbols.removeAt(bestIdx + 1)
        }
        return symbols
    }

    /** Split [word] into code-point symbols (surrogate pairs kept together) and merge. */
    fun mergeWord(word: String): List<String> {
        if (word.isEmpty()) return emptyList()
        val symbols = ArrayList<String>(word.length)
        var i = 0
        while (i < word.length) {
            val c = word[i]
            val w = if (c.isHighSurrogate() && i + 1 < word.length && word[i + 1].isLowSurrogate()) 2 else 1
            symbols.add(word.substring(i, i + w))
            i += w
        }
        return merge(symbols)
    }

    companion object {
        /** Parse one `merges` entry: `"a b"` string form or `["a", "b"]` array form (SKaiNET #1323). */
        fun parseMergeLine(line: String): Pair<String, String> {
            val sp = line.indexOf(' ')
            require(sp > 0 && sp < line.length - 1) { "malformed merge line: '$line'" }
            return line.substring(0, sp) to line.substring(sp + 1)
        }
    }
}
