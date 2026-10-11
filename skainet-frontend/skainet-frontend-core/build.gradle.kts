@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    id("sk.ainet.dokka")
}

// SKEEP-006: the input front-end contract. Root FrontEnd<I, O>, the serialisable FrontEndSpec,
// the per-modality contracts (text today; audio and image declared for their lanes), multimodal
// composition and the FrontEndResolver. Kernels live in skainet-frontend-<modality>.
kotlin {
    explicitApi()

    jvm()

    android {
        namespace = "sk.ainet.frontend.core"
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
                api(project(":skainet-lang:skainet-lang-core"))
                api(project(":skainet-models:skainet-model-identity"))
                api(libs.kotlinx.serialization.json)
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
    }
}
