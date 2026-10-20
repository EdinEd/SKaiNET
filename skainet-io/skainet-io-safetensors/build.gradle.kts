import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

kotlin {
    targets.configureEach {
        compilations.configureEach {
            compileTaskProvider.get().compilerOptions {
                freeCompilerArgs.add("-Xexpect-actual-classes")
            }
        }
    }

    jvm()
    android {
        namespace = "sk.ainet.io.safetensors"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_1_8)
        }
    }

    iosArm64()
    iosSimulatorArm64()
    macosArm64()
    linuxX64()
    linuxArm64()
    // androidNative (edge NPU / phone). io-safetensors has no posix in its own nativeMain —
    // createRandomAccessSource / readTextFile are stubs and currentTimeMillis uses a monotonic
    // TimeSource — so there is no bit-width metadata issue here (unlike io-core, whose posix pread
    // needed the native64Main split). File-backed reads route through io-core's RandomAccessSource.
    androidNativeArm32()
    androidNativeArm64()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(libs.kotlinx.io.core)
                implementation(libs.kotlinx.coroutines)
                implementation(project(":skainet-lang:skainet-lang-core"))
                implementation(project(":skainet-io:skainet-io-core"))
                implementation(libs.kotlinx.serialization.json)
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.kotlinx.coroutines.test)
            }
        }
        // The browser-side sharded load end to end: Blobs through io-core's JsBlobRandomAccessSource
        // into the loader, materialised on the common CPU context.
        val wasmJsTest by getting {
            dependencies {
                implementation(libs.kotlinx.browser)
                implementation(project(":skainet-backends:skainet-backend-cpu"))
            }
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.junit)
                implementation(libs.kotlinx.coroutines)
                implementation(libs.kotlinx.coroutines.test)
                implementation(project(":skainet-backends:skainet-backend-cpu"))
            }
        }
    }
}
