package sk.ainet.frontend.tokenizer

/** No normalisation tables on this target: pass-through, reported through [isExact]. */
public actual object UnicodeNormalization {
    public actual val isExact: Boolean = false

    public actual fun normalize(text: String, form: NormalizationForm): String = text
}
