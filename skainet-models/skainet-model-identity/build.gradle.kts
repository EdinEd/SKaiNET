@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    id("sk.ainet.dokka")
}

// SKEEP-006: model identity — the string-based ModelIdentity, the immutable ModelFamilyRegistry,
// the per-format CheckpointView contract and the NameMap tables that weight mapping keys on.
// Same target set as skainet-io-core so every reader module can depend on it.
kotlin {
    explicitApi()

    jvm()

    android {
        namespace = "sk.ainet.model.identity"
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
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
    }
}
