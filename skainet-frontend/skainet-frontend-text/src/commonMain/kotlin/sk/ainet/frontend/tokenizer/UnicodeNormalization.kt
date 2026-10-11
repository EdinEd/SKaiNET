package sk.ainet.frontend.tokenizer

/** Unicode normalisation forms a `tokenizer.json` normalizer can request. */
public enum class NormalizationForm { NFC, NFD, NFKC, NFKD }

/**
 * Platform Unicode normalisation.
 *
 * JVM and Android use `java.text.Normalizer`; JS and Wasm/JS use `String.prototype.normalize`.
 * Kotlin/Native and Wasm/WASI have no normalisation tables in the standard library and return the
 * input unchanged — [isExact] reports `false` there so goldens can mark the expectation, and a
 * `Normalizer` built for such a target still fails loudly on nothing: for pre-composed input
 * (the overwhelming majority of real text) the result is identical.
 */
public expect object UnicodeNormalization {
    /** `true` when this target implements the forms, `false` when [normalize] is a pass-through. */
    public val isExact: Boolean

    public fun normalize(text: String, form: NormalizationForm): String
}
