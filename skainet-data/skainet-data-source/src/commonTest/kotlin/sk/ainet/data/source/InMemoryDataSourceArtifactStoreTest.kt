package sk.ainet.data.source

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The resolver over the in-memory store — the browser's configuration — on every target. */
class InMemoryDataSourceArtifactStoreTest {

    private class CountingFetcher(private vararg val payloads: ByteArray) : RemoteDataSourceFetcher {
        var calls = 0
        var lastHeaders: Map<String, String> = emptyMap()
        override suspend fun fetch(uri: String, headers: Map<String, String>): DataSourceRemoteContent {
            lastHeaders = headers
            val bytes = payloads[minOf(calls, payloads.size - 1)]
            calls++
            return DataSourceRemoteContent.fromBytes(bytes.copyOf())
        }
    }

    private fun resolver(store: InMemoryDataSourceArtifactStore, fetcher: RemoteDataSourceFetcher, token: DataSourceAuthToken? = null) =
        DefaultDataSourceResolver(
            store = store,
            fetcher = fetcher,
            checksum = Sha256DataSourceChecksum,
            headerProvider = HuggingFaceTokenHeaderProvider { _, parsed -> if (parsed.provider == DataSourceProvider.HuggingFace) token else null },
        )

    @Test
    fun remoteArtifactsAreCachedInMemoryWithoutAPath() = runTest {
        val store = InMemoryDataSourceArtifactStore()
        val fetcher = CountingFetcher("first".encodeToByteArray())
        val request = DataSourceRequest("https://example.test/data.bin")

        val first = resolver(store, fetcher).resolve(request)
        val second = resolver(store, fetcher).resolve(request)

        assertEquals(1, fetcher.calls)
        assertFalse(first.cacheHit)
        assertTrue(second.cacheHit)
        assertNull(second.localPath)
        assertEquals(5L, second.sizeBytes)
        assertContentEquals("first".encodeToByteArray(), second.readBytes())
        assertEquals(1, store.cachedKeys.size)
        assertEquals(5L, store.cachedBytes)
    }

    @Test
    fun refreshBypassAndOfflineFollowTheCommonPolicies() = runTest {
        val store = InMemoryDataSourceArtifactStore()
        val fetcher = CountingFetcher("old".encodeToByteArray(), "new".encodeToByteArray())
        val uri = "https://example.test/data.bin"

        assertFailsWith<DataSourceException> {
            resolver(store, fetcher).resolve(DataSourceRequest(uri, cachePolicy = CachePolicy.Offline))
        }
        resolver(store, fetcher).resolve(DataSourceRequest(uri))
        val refreshed = resolver(store, fetcher).resolve(DataSourceRequest(uri, cachePolicy = CachePolicy.Refresh))
        assertEquals(2, fetcher.calls)
        assertContentEquals("new".encodeToByteArray(), refreshed.readBytes())

        store.clear()
        val bypassed = resolver(store, fetcher).resolve(DataSourceRequest(uri, cachePolicy = CachePolicy.Bypass))
        assertEquals(3, fetcher.calls)
        assertFalse(bypassed.cacheHit)
        assertTrue(store.cachedKeys.isEmpty(), "bypass writes nothing")
        assertContentEquals("new".encodeToByteArray(), bypassed.readBytes())
    }

    @Test
    fun checksumsAreVerifiedWithTheKotlinSha256() = runTest {
        val store = InMemoryDataSourceArtifactStore()
        val fetcher = CountingFetcher("payload".encodeToByteArray())
        val good = "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5"

        val ok = resolver(store, fetcher).resolve(DataSourceRequest("https://example.test/payload.bin", expectedSha256 = good.uppercase()))
        assertContentEquals("payload".encodeToByteArray(), ok.readBytes())

        store.clear()
        assertFailsWith<DataSourceException> {
            resolver(store, fetcher).resolve(DataSourceRequest("https://example.test/payload.bin", expectedSha256 = "0000"))
        }
        assertTrue(store.cachedKeys.isEmpty(), "a failed checksum caches nothing")
        // The cached copy is verified again on a hit.
        resolver(store, fetcher).resolve(DataSourceRequest("https://example.test/payload.bin"))
        assertFailsWith<DataSourceException> {
            resolver(store, fetcher).resolve(DataSourceRequest("https://example.test/payload.bin", expectedSha256 = "0000"))
        }
    }

    @Test
    fun fileUrisResolveOnlyToRegisteredBytes() = runTest {
        val store = InMemoryDataSourceArtifactStore(localArtifacts = mapOf("/picked/sample.txt" to "hello".encodeToByteArray()))
        val fetcher = CountingFetcher(ByteArray(0))

        val artifact = resolver(store, fetcher).resolve(DataSourceRequest("file:///picked/sample.txt"))
        assertEquals("sample.txt", artifact.filename)
        assertEquals("/picked/sample.txt", artifact.localPath)
        assertTrue(artifact.cacheHit)
        assertContentEquals("hello".encodeToByteArray(), artifact.readBytes())

        assertFailsWith<DataSourceException> { resolver(store, fetcher).resolve(DataSourceRequest("/picked/missing.txt")) }
        assertEquals(0, fetcher.calls)
    }

    @Test
    fun aConfiguredHuggingFaceTokenReachesOnlyHuggingFace() = runTest {
        val store = InMemoryDataSourceArtifactStore()
        val fetcher = CountingFetcher("x".encodeToByteArray())
        val r = resolver(store, fetcher, DataSourceAuthToken.from("hf_configured"))

        r.resolve(DataSourceRequest("hf://org/repo@main/file.bin"))
        assertEquals("Bearer hf_configured", fetcher.lastHeaders["Authorization"])
        r.resolve(DataSourceRequest("https://example.test/plain.bin"))
        assertEquals(emptyMap(), fetcher.lastHeaders)
    }

    @Test
    fun evictDropsOneEntry() = runTest {
        val store = InMemoryDataSourceArtifactStore()
        val fetcher = CountingFetcher("a".encodeToByteArray())
        resolver(store, fetcher).resolve(DataSourceRequest("https://example.test/a.bin"))
        resolver(store, fetcher).resolve(DataSourceRequest("https://example.test/b.bin"))
        val key = store.cachedKeys.first()
        assertTrue(store.evict(key))
        assertFalse(store.evict(key))
        assertEquals(1, store.cachedKeys.size)
    }
}
