@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.net.URI

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    id("sk.ainet.dokka")
}

// SKEEP-006: the text front ends — every tokenizer algorithm the engine ships, the tokenizer.json /
// GGUF / tekken.json / vocab.txt readers, and the TextFrontEnds provider the FrontEndResolver
// composes. Moved here from skainet-io-core (which keeps deprecated aliases for one cycle).
kotlin {
    explicitApi()

    jvm()

    android {
        namespace = "sk.ainet.frontend.text"
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
    androidNativeArm64()
    androidNativeArm32()

    js {
        browser()
    }

    wasmJs {
        browser()
    }

    wasmWasi {
        nodejs()
    }

    applyDefaultHierarchyTemplate()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":skainet-frontend:skainet-frontend-core"))
                implementation(libs.kotlinx.serialization.json)
            }
        }
        // Unicode normalisation (NFC/NFD/NFKC/NFKD) via java.text.Normalizer, shared by the JVM
        // and Android compilations — same shared-directory pattern as skainet-io-core.
        getByName("jvmMain") {
            kotlin.srcDir("src/jvmAndroidMain/kotlin")
        }
        getByName("androidMain") {
            kotlin.srcDir("src/jvmAndroidMain/kotlin")
        }
        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlinx.serialization.json)
            }
        }
    }
}

// ============================================================================
// Golden-corpus fixtures (SKEEP-006 acceptance): public tokenizer files the
// jvmTest goldens in src/jvmTest/resources/goldens/*.json were recorded
// against with the Hugging Face `tokenizers` library 0.23.3. Tests skip
// cleanly when a fixture is absent, so offline builds stay green.
//
//   ./gradlew :skainet-frontend:skainet-frontend-text:downloadTokenizerFixtures
// ============================================================================
val fixturesDir = layout.buildDirectory.dir("test-fixtures")

val downloadTokenizerFixtures by tasks.registering {
    group = "verification"
    description = "Download the public tokenizer files the golden tests compare against"
    val outDir = fixturesDir
    outputs.dir(outDir)
    doLast {
        val dir = outDir.get().asFile.apply { mkdirs() }
        val laya = "https://huggingface.co/convaiinnovations/laya/resolve/7b928d828b7b0e022f929d9bd2e44165aa270148"
        val files = listOf(
            "laya-modernbert-tokenizer.json" to "$laya/tokenizer/tokenizer.json",
            "laya-mmbert-tokenizer.json" to "$laya/multilingual/tokenizer/tokenizer.json",
            "qwen25-tokenizer.json" to "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
            "qwen25-tokenizer_config.json" to "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer_config.json",
            "tinyllama-tokenizer.json" to "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json",
            "tinyllama-tokenizer_config.json" to "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer_config.json",
            "gpt2-tokenizer.json" to "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
            "bert-base-uncased-vocab.txt" to "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt",
            "bert-base-uncased-tokenizer.json" to "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json",
            "bert-base-uncased-tokenizer_config.json" to "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer_config.json",
            "t5-small-tokenizer.json" to "https://huggingface.co/google-t5/t5-small/resolve/main/tokenizer.json",
            "t5-small-tokenizer_config.json" to "https://huggingface.co/google-t5/t5-small/resolve/main/tokenizer_config.json",
            "gemma4-e2b-tokenizer.json" to "https://huggingface.co/google/gemma-4-E2B-it/resolve/main/tokenizer.json",
            "gemma4-e2b-tokenizer_config.json" to "https://huggingface.co/google/gemma-4-E2B-it/resolve/main/tokenizer_config.json",
        )
        for ((name, url) in files) {
            val target = dir.resolve(name)
            if (target.exists() && target.length() > 0) {
                logger.lifecycle("fixture already present: ${target.name}")
                continue
            }
            logger.lifecycle("downloading $name from $url")
            URI(url).toURL().openStream().use { input ->
                target.outputStream().use { out -> input.copyTo(out) }
            }
            logger.lifecycle("  -> ${target.length()} bytes")
        }
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("skainet.test.fixturesDir", fixturesDir.get().asFile.absolutePath)
}
