plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    alias(libs.plugins.kotlinSerialization)
    application
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(libs.kotlinx.cli)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines)
    implementation(libs.pbandk.runtime)

    implementation(project(":skainet-lang:skainet-lang-core"))
    implementation(project(":skainet-lang:skainet-lang-models"))
    implementation(project(":skainet-io:skainet-io-core"))
    implementation(project(":skainet-io:skainet-io-onnx"))
    implementation(project(":skainet-io:skainet-io-image"))
    implementation(project(":skainet-io:skainet-io-gguf"))
    implementation(project(":skainet-io:skainet-io-safetensors"))
    implementation(libs.kotlinx.io.core)
    implementation(project(":skainet-backends:skainet-backend-cpu"))
    implementation(libs.pbandk.runtime)

    testImplementation(kotlin("test"))
}

application {
    mainClass.set("sk.ainet.apps.onnx.tools.WeightExportKt")
    applicationDefaultJvmArgs = listOf(
        "--add-modules=jdk.incubator.vector"
    )
}

tasks.test {
    useJUnitPlatform()
}

// SKEEP-007 §T2: SafeTensors → quantized GGUF, offline.
//   ./gradlew :skainet-apps:skainet-tensor-tools:quantize --args="--in model.safetensors --out model.q8_0.gguf --keep tok_embeddings"
tasks.register<JavaExec>("quantize") {
    group = "application"
    description = "Quantize a SafeTensors checkpoint's weight matrices into a GGUF (Q8_0 or Q4_0)"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("sk.ainet.apps.tool.quantize.QuantizeCliKt")
    jvmArgs("--add-modules=jdk.incubator.vector")
}
