package sk.ainet.data.source

import io.ktor.client.HttpClient
import io.ktor.client.engine.js.Js
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.statement.bodyAsBytes

/**
 * Ktor fetcher over the browser's `fetch` (the Ktor JS engine) for Kotlin/JS and Kotlin/Wasm — the
 * same class name and contract as the JVM's CIO-backed fetcher, so a resolver is configured the same
 * way on every target. The body is read whole: `fetch` hands the page the bytes, and the browser
 * store keeps artifacts in memory anyway, so there is nothing a streaming source would save here.
 * Cross-origin hosts must allow the page (CORS); Hugging Face does.
 */
public class KtorRemoteDataSourceFetcher(
    private val client: HttpClient = HttpClient(Js) {
        expectSuccess = true
        install(HttpTimeout) {
            requestTimeoutMillis = 600_000
            connectTimeoutMillis = 60_000
            socketTimeoutMillis = 600_000
        }
    },
) : RemoteDataSourceFetcher, AutoCloseable {
    override suspend fun fetch(uri: String, headers: Map<String, String>): DataSourceRemoteContent {
        val response = client.get(uri) {
            headers.forEach { (name, value) -> header(name, value) }
        }
        return DataSourceRemoteContent.fromBytes(response.bodyAsBytes())
    }

    override fun close() {
        client.close()
    }
}

/**
 * The browser's [DataSourceResolver] (Kotlin/JS and Kotlin/Wasm): HTTP(S) and Hugging Face URIs
 * over `fetch`, the same cache policies, checksum verification and token handling as
 * `JvmDataSourceResolver`, with these differences a tab imposes:
 *
 *  - **No file system.** Remote artifacts are cached in memory for the life of the resolver
 *    ([InMemoryDataSourceArtifactStore]); a resolved artifact has no `localPath`. A `file:` URI
 *    resolves only to bytes the application registered in [localArtifacts] under that path — what
 *    it received from a file picker or a drop — and fails with [DataSourceException] otherwise.
 *    An application that wants a persistent cache gives [store] its own implementation over
 *    the Cache API, IndexedDB or OPFS.
 *  - **No environment.** A Hugging Face token comes from the request or [huggingFaceToken]; there
 *    is no `HF_TOKEN` to read.
 *  - **SHA-256 in Kotlin** ([Sha256DataSourceChecksum]); `crypto.subtle` is asynchronous and
 *    cannot serve the incremental hash the resolver streams through.
 */
public class WebDataSourceResolver(
    fetcher: RemoteDataSourceFetcher = KtorRemoteDataSourceFetcher(),
    huggingFaceToken: DataSourceAuthToken? = null,
    localArtifacts: Map<String, ByteArray> = emptyMap(),
    store: DataSourceArtifactStore = InMemoryDataSourceArtifactStore(localArtifacts),
) : DataSourceResolver {
    private val delegate = DefaultDataSourceResolver(
        store = store,
        fetcher = fetcher,
        checksum = Sha256DataSourceChecksum,
        headerProvider = HuggingFaceTokenHeaderProvider { request, parsedUri ->
            if (parsedUri.provider == DataSourceProvider.HuggingFace) huggingFaceToken else null
        },
    )

    override suspend fun resolve(request: DataSourceRequest): DataSourceArtifact = delegate.resolve(request)
}
