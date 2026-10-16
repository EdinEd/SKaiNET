import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

// SKEEP-007 Tier 1: the browser kernel pack. A wasmJs-only module whose kernels are the engine's own
// C sources (skainet-backend-native-cpu/native/src) compiled by Emscripten into a SIMD128 sidecar
// module (src/wasmJsMain/resources/skainet_kernels_{st,mt}.{mjs,wasm}, produced by
// native/build-wasm.sh and committed). The Kotlin side registers a ViewKernelPack and a
// KernelProvider for the sidecar and a TensorOps subclass that routes dense matmuls through it.
//
// Tests run on Node (`wasmJsNodeTest`): the sidecar is loaded from the module's resources.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

kotlin {
    explicitApi()

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        nodejs()
    }

    sourceSets {
        wasmJsMain.dependencies {
            api(project(":skainet-lang:skainet-lang-core"))
            api(project(":skainet-backends:skainet-backend-api"))
            api(project(":skainet-backends:skainet-backend-cpu"))
            implementation(libs.kotlinx.coroutines)
        }
        wasmJsTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}
