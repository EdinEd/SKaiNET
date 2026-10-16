package sk.ainet.io.safetensors

import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.tensor.data.Requantizers
import sk.ainet.lang.types.FP32

/**
 * What the SafeTensors loaders accept in a [WeightForm] (SKEEP-007 §T2), checked at construction so
 * an impossible request fails before the first tensor is read — the same contract the GGUF loader
 * has had since #1109.
 *
 * - `KeepAsStored`: the narrow-float policies decide, as before.
 * - `DequantizeTo(FP32)`: the historical widening; any other dtype is refused.
 * - `RequantizeTo(encoding)`: accepted for every encoding [Requantizers] has a quantizer for.
 * - `WeightByteOrder.KERNEL_FEED` is honoured for the requantized formats that define it and is a
 *   no-op for dense tensors; `WeightShapeOrientation` is moot (SafeTensors weights are `[out, in]`);
 *   `WeightResidency.MAPPED` means heap for anything this loader produces.
 */
internal object SafeTensorsWeightForms {
    fun validate(form: WeightForm, where: String) {
        when (val request = form.encoding) {
            is EncodingRequest.RequantizeTo -> require(Requantizers.supports(request.encoding)) {
                "$where: EncodingRequest.RequantizeTo(${request.encoding.name}) is not supported: no quantizer " +
                    "exists for that encoding (quantize-at-load supports ${Requantizers.supported.joinToString { it.name }})."
            }
            is EncodingRequest.DequantizeTo -> require(request.dtype == FP32) {
                "$where: EncodingRequest.DequantizeTo(${request.dtype.name}) is not supported: this loader widens to FP32 only."
            }
            EncodingRequest.KeepAsStored -> Unit
        }
    }
}
