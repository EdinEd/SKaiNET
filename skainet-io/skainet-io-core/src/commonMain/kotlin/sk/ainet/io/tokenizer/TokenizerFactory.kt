package sk.ainet.io.tokenizer

import sk.ainet.frontend.FrontEndResolver
import sk.ainet.frontend.UnsupportedFrontEndException
import sk.ainet.frontend.text.TextFrontEnd
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.tokenizer.TextFrontEnds
import sk.ainet.model.CheckpointView
import sk.ainet.model.ModelFamilyRegistry
import kotlin.jvm.JvmStatic

/**
 * Selects the right [Tokenizer] implementation for a model.
 *
 * Since SKEEP-006 the logic lives in `skainet-frontend-text` ([TextFrontEnds]) and the
 * `FrontEndResolver`; this object forwards to it so the `@JvmStatic` entry points Java consumers
 * pin keep their signatures. New code should call [TextFrontEnds] or [FrontEndResolver] directly.
 *
 * Supported: byte-level BPE (Qwen, GPT-2, ModernBERT), Metaspace BPE with byte fallback
 * (Llama 2, TinyLlama, mmBERT), SentencePiece (GGUF `llama` / `gemma4`), Unigram (T5, ALBERT,
 * Gemma exports), WordPiece (BERT) and Tekken (Mistral). Failures surface as
 * [UnsupportedTokenizerException].
 */
public object TokenizerFactory {

    /**
     * Build a tokenizer from a GGUF metadata field map (`streamingReader.fields` or
     * `ggufModelMetadata.rawFields`).
     */
    @JvmStatic
    public fun fromGguf(fields: Map<String, Any?>): Tokenizer = translate { TextFrontEnds.fromGguf(fields) }

    /** Build a tokenizer from a Hugging Face `tokenizer.json` string. */
    @JvmStatic
    public fun fromTokenizerJson(json: String): Tokenizer = translate { TextFrontEnds.fromTokenizerJson(json) }

    /** Build a tokenizer from `tokenizer.json` plus its sibling `tokenizer_config.json` (FE-R-03). */
    @JvmStatic
    public fun fromTokenizerJson(json: String, configJson: String?): Tokenizer = translate { TextFrontEnds.fromTokenizerJson(json, configJson) }

    /** Build the Mistral Tekken tokenizer from a `tekken.json` string. */
    @JvmStatic
    public fun fromTekkenJson(json: String): Tokenizer = translate { TextFrontEnds.fromTekkenJson(json) }

    /** Build a WordPiece tokenizer from a BERT `vocab.txt`. */
    @JvmStatic
    public fun fromVocabTxt(content: String, doLowerCase: Boolean): Tokenizer = translate { TextFrontEnds.fromVocabTxt(content, doLowerCase) }

    /** Resolve the text front end of a checkpoint view through the built-in family registry. */
    @JvmStatic
    public fun fromCheckpoint(view: CheckpointView): TextFrontEnd = translate { resolver().resolveText(view) }

    /** A text-only [FrontEndResolver] over [registry]. */
    @JvmStatic
    public fun resolver(registry: ModelFamilyRegistry = ModelFamilyRegistry.builtIn): FrontEndResolver = TextFrontEnds.resolver(registry)

    private inline fun <T> translate(block: () -> T): T = try {
        block()
    } catch (e: UnsupportedTokenizerException) {
        throw e
    } catch (e: UnsupportedFrontEndException) {
        throw UnsupportedTokenizerException(e.message ?: "unsupported tokenizer", e)
    }
}

/** A tokenizer source no implementation supports; the message names the file or field. */
public class UnsupportedTokenizerException(message: String, cause: Throwable? = null) : UnsupportedFrontEndException(message, cause)
