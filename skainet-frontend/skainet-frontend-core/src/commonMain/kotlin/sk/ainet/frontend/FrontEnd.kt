package sk.ainet.frontend

/**
 * Converts raw model input of type [I] into what a model consumes ([O]) — SKEEP-006 §Contract.
 *
 * A front end is stateless and immutable (FE-C-01); per-utterance state lives only in a
 * [FrontEndStream]. An `ExecutionContext` is never a parameter of [process] (FE-C-02):
 * tensor-producing front ends take it at construction, so a resolved front end is a closed object
 * a cartridge facade can hand around.
 */
public interface FrontEnd<I, O> {
    /** Static, serialisable description: modality, algorithm, parameters, source artifact. */
    public val spec: FrontEndSpec

    public fun process(input: I): O
}

/** Front ends that map model output back to the input domain (text today). */
public interface InvertibleFrontEnd<I, O> : FrontEnd<I, O> {
    public fun invert(output: O): I
}

/**
 * Chunked operation for unbounded inputs (audio). One session per utterance; sessions are not
 * thread-safe. Specified with the contract (FE-C-03); the first implementor is the audio lane.
 */
public interface StreamingFrontEnd<I, O> : FrontEnd<I, O> {
    public fun openStream(): FrontEndStream<I, O>
}

public interface FrontEndStream<I, O> : AutoCloseable {
    public fun push(chunk: I)

    /** Every completed output since the last poll; empty if none is ready. */
    public fun poll(): List<O>

    /** Flushes the tail (padded to the model window) and returns the final output, or `null` if nothing is pending. */
    public fun finish(): O?
}
