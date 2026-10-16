import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

// SKEEP-007 §T0.5: the wasm-lane baseline. A Kotlin/Wasm executable that times the matmul shapes of
// the shipped models through the eager CPU backend and prints one `[SKAINET_PERF_WASM]` line per
// case. Not published.
//
//   ./gradlew :skainet-backends:benchmarks:wasm-bench:wasmJsNodeDevelopmentRun        # quick, unoptimised
//   ./gradlew :skainet-backends:benchmarks:wasm-bench:wasmJsNodeProductionRun         # the numbers to record
//   ./gradlew :skainet-backends:benchmarks:wasm-bench:wasmJsBrowserProductionRun      # same in Chrome
plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

kotlin {
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        nodejs()
        browser()
        binaries.executable()
    }

    sourceSets {
        wasmJsMain.dependencies {
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(project(":skainet-backends:skainet-backend-api"))
            implementation(project(":skainet-backends:skainet-backend-cpu"))
            implementation(project(":skainet-backends:skainet-backend-wasm-simd"))
            implementation(libs.kotlinx.coroutines)
        }
    }
}
