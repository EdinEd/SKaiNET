package sk.ainet.io.model

/**
 * Known model architectures that SKaiNET can load and run.
 */
public enum class ModelArchitecture {
    LLAMA,
    GEMMA,
    BERT,
    QWEN,
    @Deprecated("Model-specific detectors live outside SKaiNET core (e.g. in their cartridge blueprint); nothing in core maps to this value. Removed in the next minor release.")
    YOLO,
    UNKNOWN;

    public companion object
}
