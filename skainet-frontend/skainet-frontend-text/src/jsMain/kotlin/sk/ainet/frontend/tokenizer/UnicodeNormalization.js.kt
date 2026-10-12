package sk.ainet.frontend.tokenizer

public actual object UnicodeNormalization {
    public actual val isExact: Boolean = true

    public actual fun normalize(text: String, form: NormalizationForm): String =
        text.asDynamic().normalize(form.name).unsafeCast<String>()
}
