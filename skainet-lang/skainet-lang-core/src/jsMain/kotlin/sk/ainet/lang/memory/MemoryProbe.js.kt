package sk.ainet.lang.memory

/**
 * Kotlin/JS: no process to ask, but Chromium reports the JavaScript heap — where every Kotlin/JS
 * object lives — through `performance.memory.usedJSHeapSize`; other engines do not, and then the
 * probe says `null`. Page faults are not observable in a browser.
 */
public actual object MemoryProbe {
    public actual fun rssBytes(): Long? = jsHeapUsedBytes()
    public actual fun majorFaults(): Long? = null
    public actual fun minorFaults(): Long? = null

    /** `performance.memory.usedJSHeapSize`, or `null` where the engine does not expose it. */
    public fun jsHeapUsedBytes(): Long? {
        val used: Double = js("(typeof performance !== 'undefined' && performance.memory) ? performance.memory.usedJSHeapSize : -1")
        return if (used >= 0) used.toLong() else null
    }
}
