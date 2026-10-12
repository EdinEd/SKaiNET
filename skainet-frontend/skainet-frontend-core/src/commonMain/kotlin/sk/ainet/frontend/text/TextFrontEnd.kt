package sk.ainet.frontend.text

import sk.ainet.frontend.FrontEnd
import sk.ainet.frontend.FrontEndSpec

/**
 * Marker for family-provided prompt renderers (FE-C-09). Chat-template rendering is text-to-text
 * before tokenization and depends on a message schema the engine does not own; SKaiNET-transformers'
 * `ChatTemplate` implements this with its typed `apply(messages, tools, …)`.
 */
public interface PromptRenderer

/**
 * A [Tokenizer] plus the checkpoint's prompt-rendering material.
 *
 * @property tokenizer the resolved tokenizer
 * @property chatTemplateSource raw template string from GGUF `tokenizer.chat_template` or
 *   `tokenizer_config.json#chat_template`, if shipped
 * @property promptRenderer family-provided renderer; the engine only knows that it exists
 */
public class TextFrontEnd(
    public val tokenizer: Tokenizer,
    public val chatTemplateSource: String? = null,
    public val promptRenderer: PromptRenderer? = null,
) : FrontEnd<String, IntArray> by tokenizer {

    override val spec: FrontEndSpec get() = tokenizer.spec

    /** The same tokenizer with a different renderer / template (used by family decoration). */
    public fun copy(
        tokenizer: Tokenizer = this.tokenizer,
        chatTemplateSource: String? = this.chatTemplateSource,
        promptRenderer: PromptRenderer? = this.promptRenderer,
    ): TextFrontEnd = TextFrontEnd(tokenizer, chatTemplateSource, promptRenderer)
}
