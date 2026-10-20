package sk.ainet.data.source

import kotlinx.io.Source

/**
 * [DataSourceArtifactStore] that keeps everything in memory: the store for a host without a file
 * system — a browser tab — and for tests. Remote artifacts are cached for the life of the store
 * (one per `cacheKey`, so a second `resolve` of the same URI is a cache hit that fetches nothing),
 * and `file:` URIs resolve only to the bytes registered in [localArtifacts] under their path —
 * what a page received from a file picker or a drop, keyed by the name the application chose.
 * Stored artifacts have no `localPath`: there is no file.
 */
public class InMemoryDataSourceArtifactStore(
    private val localArtifacts: Map<String, ByteArray> = emptyMap(),
) : DataSourceArtifactStore {
    private val cache = LinkedHashMap<String, DataSourceStoredArtifact>()

    /** Cache keys held right now, in insertion order. */
    public val cachedKeys: Set<String> get() = cache.keys.toSet()

    /** Bytes held by the cache (registered local artifacts not counted). */
    public val cachedBytes: Long get() = cache.values.sumOf { it.sizeBytes ?: 0L }

    /** Drops every cached remote artifact; registered local artifacts stay. */
    public fun clear() {
        cache.clear()
    }

    /** Forgets one cached remote artifact; `true` when there was one. */
    public fun evict(cacheKey: String): Boolean = cache.remove(cacheKey) != null

    override suspend fun readLocal(path: String): DataSourceStoredArtifact? =
        localArtifacts[path]?.let { DataSourceStoredArtifact.inMemory(it, localPath = path) }

    override suspend fun readCache(cacheKey: String): DataSourceStoredArtifact? = cache[cacheKey]

    override suspend fun writeCache(
        cacheKey: String,
        source: Source,
        sizeBytes: Long?,
        validate: suspend (DataSourceStoredArtifact) -> Unit,
    ): DataSourceStoredArtifact {
        val stored = DataSourceStoredArtifact.inMemoryFrom(source, localPath = null, sizeBytes = sizeBytes)
        validate(stored)
        cache[cacheKey] = stored
        return stored
    }
}
