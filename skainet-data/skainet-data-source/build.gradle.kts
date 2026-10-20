plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

// Targets: jvm, js, wasmJs — see skainet.targets in this module's gradle.properties. The two web
// targets share `webMain` / `webTest` (the default hierarchy's shared source set for js + wasmJs):
// the Ktor JS engine over `fetch` and an in-memory cache, since a tab has no file system.

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(libs.kotlinx.coroutines)
            // `api`: kotlinx.io.Source / Sink are part of this module's public API (DataSourceArtifact.openSource(),
            // copyTo(), DataSourceRemoteContent, DataSourceArtifactStore), so consumers need them on their compile
            // classpath. As `implementation` a consumer could not call those functions without adding kotlinx-io itself.
            api(libs.kotlinx.io.core)
            implementation(libs.kotlinx.serialization.json)
        }

        commonTest.dependencies {
            implementation(libs.kotlinx.coroutines.test)
        }

        jvmMain.dependencies {
            implementation(libs.ktor.client.cio)
            implementation(libs.ktor.client.core)
            implementation(libs.ktor.client.plugins)
            implementation(libs.kotlinx.coroutines.core.jvm)
        }

        webMain.dependencies {
            implementation(libs.ktor.client.core)
            implementation(libs.ktor.client.js)
        }
    }
}
