package sk.ainet.frontend.tokenizer

private fun jsNormalize(text: String, form: String): String = js("text.normalize(form)")

public actual object UnicodeNormalization {
    public actual val isExact: Boolean = true

    public actual fun normalize(text: String, form: NormalizationForm): String = jsNormalize(text, form.name)
}
