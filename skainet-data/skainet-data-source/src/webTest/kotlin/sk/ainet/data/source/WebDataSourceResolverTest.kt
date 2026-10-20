package sk.ainet.data.source

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The page's origin — the Karma server that serves this test — or `""` outside a browser. */
private fun pageOrigin(): String = js("(typeof location !== 'undefined' && location.origin) ? location.origin : ''")

/**
 * The browser resolver under Chromium: the Ktor JS engine really fetches from the test server,
 * the in-memory cache and the Kotlin SHA-256 do the rest exactly as on the JVM.
 */
class WebDataSourceResolverTest {

    private class CountingFetcher(private val bytes: ByteArray) : RemoteDataSourceFetcher {
        var calls = 0
        var lastHeaders: Map<String, String> = emptyMap()
        override suspend fun fetch(uri: String, headers: Map<String, String>): DataSourceRemoteContent {
            calls++
            lastHeaders = headers
            return DataSourceRemoteContent.fromBytes(bytes.copyOf())
        }
    }

    @Test
    fun fetchesFromTheTestServerThroughTheJsEngineAndCachesInMemory() = runTest {
        val origin = pageOrigin()
        assertTrue(origin.startsWith("http"), "runs in a browser served over HTTP, origin was '$origin'")
        val fetcher = KtorRemoteDataSourceFetcher()
        try {
            val resolver = WebDataSourceResolver(fetcher = fetcher)
            // Karma serves its own test page at the origin.
            val request = DataSourceRequest("$origin/context.html")
            val first = resolver.resolve(request)
            val bytes = first.readBytes()
            assertTrue(bytes.isNotEmpty(), "the page has a body")
            assertTrue(bytes.decodeToString().contains("<", ignoreCase = true), "the body is HTML")
            assertFalse(first.cacheHit)
            assertNull(first.localPath)
            assertEquals("context.html", first.filename)

            val second = resolver.resolve(request)
            assertTrue(second.cacheHit)
            assertContentEquals(bytes, second.readBytes())

            val bypassed = resolver.resolve(DataSourceRequest("$origin/context.html", cachePolicy = CachePolicy.Bypass))
            assertContentEquals(bytes, bypassed.readBytes())
        } finally {
            fetcher.close()
        }
    }

    @Test
    fun aMissingUrlFailsInsteadOfCachingAnErrorPage() = runTest {
        val origin = pageOrigin()
        val fetcher = KtorRemoteDataSourceFetcher()
        try {
            val resolver = WebDataSourceResolver(fetcher = fetcher)
            assertFails { resolver.resolve(DataSourceRequest("$origin/no-such-artifact-404.bin")) }
            assertFailsWith<DataSourceException> {
                resolver.resolve(DataSourceRequest("$origin/no-such-artifact-404.bin", cachePolicy = CachePolicy.Offline))
            }
        } finally {
            fetcher.close()
        }
    }

    @Test
    fun verifiesSha256AndForwardsTheConfiguredToken() = runTest {
        val fetcher = CountingFetcher("payload".encodeToByteArray())
        val resolver = WebDataSourceResolver(fetcher = fetcher, huggingFaceToken = DataSourceAuthToken.from("hf_configured"))

        val ok = resolver.resolve(
            DataSourceRequest(
                "hf+https://huggingface.co/example/model/resolve/main/payload.bin",
                expectedSha256 = "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5",
            ),
        )
        assertContentEquals("payload".encodeToByteArray(), ok.readBytes())
        assertEquals("Bearer hf_configured", fetcher.lastHeaders["Authorization"])

        assertFailsWith<DataSourceException> {
            resolver.resolve(DataSourceRequest("https://example.test/other.bin", expectedSha256 = "0000"))
        }
        assertEquals(emptyMap(), fetcher.lastHeaders, "a generic HTTP request carries no token")
    }

    @Test
    fun fileUrisResolveToRegisteredBytesOnly() = runTest {
        val resolver = WebDataSourceResolver(
            fetcher = CountingFetcher(ByteArray(0)),
            localArtifacts = mapOf("/dropped/model.bin" to byteArrayOf(1, 2, 3)),
        )
        val artifact = resolver.resolve(DataSourceRequest("file:///dropped/model.bin"))
        assertContentEquals(byteArrayOf(1, 2, 3), artifact.readBytes())
        assertEquals("/dropped/model.bin", artifact.localPath)
        assertFailsWith<DataSourceException> { resolver.resolve(DataSourceRequest("file:///dropped/other.bin")) }
    }
}
