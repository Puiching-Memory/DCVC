# Pure-ONNX (no-FXP) full-model INT16 end-to-end test

Base: `onnx/models_fp32` (standard Conv, no FxpConv/FxpWsRelu)
Method: ORT `quantize_static`, QDQ, **act=QInt16 weight=QInt16**, per-channel,
MinMax, calibrated on 512x512 crops of beauty/bosphorus/jockey @ qp{12,22,32,42,52}.
Output: `onnx/models_int16` (all 10 intra nets quantized; inter nets copied as-is).
Scripts: `onnx/python/int16_full_quantize.py`, `onnx/python/int16_rd_compare.py`.

## Per-net quantization error (cosine vs FP32, all ~1.00000)
maxdiff 2.7e-4 (synthesis) .. 1.1e-2 (hyper_enc). No residual FP32 Conv.

## End-to-end RD (512x512, 3 frames x qp{12,22,32,42,52}, mean over frames)
| qp  | bpp32  | bpp16  | ratio | dPSNR(dB) | t32(s) | t16(s) | speed |
| --- | ------ | ------ | ----- | --------- | ------ | ------ | ----- |
| 12  | 0.0267 | 0.0267 | 0.998 | -0.016    | 1.56   | 2.93   | 0.53x |
| 22  | 0.0398 | 0.0397 | 0.998 | -0.021    | 1.55   | 2.81   | 0.55x |
| 32  | 0.0579 | 0.0578 | 0.998 | +0.004    | 1.50   | 2.77   | 0.54x |
| 42  | 0.0863 | 0.0864 | 1.002 | +0.005    | 1.53   | 2.85   | 0.54x |
| 52  | 0.1330 | 0.1331 | 1.001 | -0.005    | 1.54   | 2.82   | 0.55x |

**OVERALL: drate = -0.04%, dPSNR = -0.007 dB, speed = 0.54x (1.84x slower).**

## Size
fp32 264.3 MB -> int16 175.2 MB (33.7% smaller; intra nets ~2x smaller).

## Conclusion
- **RD-transparent**: with representative calibration, full int16 PTQ is
  rate/PSNR-equivalent to FP32 (well within +/-0.05 dB / +/-0.4% per point).
- **No speed gain on CPU**: ORT CPU EP has no int16 conv kernel -> it
  dequantizes activations to FP32 per conv and runs FP32 convs, ~1.8x slower
  than native FP32 due to Q/DQ overhead. int16 only buys 2x weight storage.
- Earlier -1.9 dB PSNR loss with 256x256/calib{12,32,52} was a pure
  calibration-distribution mismatch (analysis y rel-RMSE 11.7% -> 0.3%).
