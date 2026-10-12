@file:Suppress("DEPRECATION")

package sk.ainet.io.tokenizer

/*
 * SKEEP-006: the tokenizer implementations moved to `skainet-frontend-text`
 * (`sk.ainet.frontend.tokenizer`) and the contract to `skainet-frontend-core`
 * (`sk.ainet.frontend.text`). These aliases keep 0.57 sources compiling for one release cycle
 * (WARNING in 0.58, ERROR in 0.59, removed in 0.60). `TokenizerFactory` stays a real object
 * because Java callers pin its static methods.
 */

private const val MOVED = "Moved to skainet-frontend-core / skainet-frontend-text (SKEEP-006); the alias is removed in 0.60."

@Deprecated(MOVED, ReplaceWith("Tokenizer", "sk.ainet.frontend.text.Tokenizer"))
public typealias Tokenizer = sk.ainet.frontend.text.Tokenizer

@Deprecated(MOVED, ReplaceWith("QwenByteLevelBpeTokenizer", "sk.ainet.frontend.tokenizer.QwenByteLevelBpeTokenizer"))
public typealias QwenByteLevelBpeTokenizer = sk.ainet.frontend.tokenizer.QwenByteLevelBpeTokenizer

@Deprecated(MOVED, ReplaceWith("SentencePieceTokenizer", "sk.ainet.frontend.tokenizer.SentencePieceTokenizer"))
public typealias SentencePieceTokenizer = sk.ainet.frontend.tokenizer.SentencePieceTokenizer

@Deprecated(MOVED, ReplaceWith("SpecialTokenSplitter", "sk.ainet.frontend.tokenizer.SpecialTokenSplitter"))
public typealias SpecialTokenSplitter = sk.ainet.frontend.tokenizer.SpecialTokenSplitter

@Deprecated(MOVED, ReplaceWith("TekkenTokenizer", "sk.ainet.frontend.tokenizer.TekkenTokenizer"))
public typealias TekkenTokenizer = sk.ainet.frontend.tokenizer.TekkenTokenizer

@Deprecated(MOVED, ReplaceWith("ByteToUnicode", "sk.ainet.frontend.tokenizer.ByteToUnicode"))
public typealias ByteToUnicode = sk.ainet.frontend.tokenizer.ByteToUnicode
