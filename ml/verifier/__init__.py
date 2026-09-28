"""
Crop verifier (stage 2 of the cascade) — offline prototype, NAL only.

Training side (torch): out-of-fold candidate mining, crops cache,
TwoCropVerifier (MobileNetV3-Small), nested cross-fit, logistic stacker,
ONNX export, pre-registered exit rule, cascade evaluation. The torch-free
inference half is ml/infer/verifier.py. Entry point: verifier.cli.main(argv);
see verifier/README.md and verifier/PREREGISTRATION.md.
"""
