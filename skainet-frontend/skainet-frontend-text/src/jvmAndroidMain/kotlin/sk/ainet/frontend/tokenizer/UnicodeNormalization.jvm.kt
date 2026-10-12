package sk.ainet.frontend.tokenizer

public actual object UnicodeNormalization {
    public actual val isExact: Boolean = true

    public actual fun normalize(text: String, form: NormalizationForm): String {
        val f = when (form) {
            NormalizationForm.NFC -> java.text.Normalizer.Form.NFC
            NormalizationForm.NFD -> java.text.Normalizer.Form.NFD
            NormalizationForm.NFKC -> java.text.Normalizer.Form.NFKC
            NormalizationForm.NFKD -> java.text.Normalizer.Form.NFKD
        }
        return if (java.text.Normalizer.isNormalized(text, f)) text else java.text.Normalizer.normalize(text, f)
    }
}
