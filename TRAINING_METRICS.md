# Training metrics included in this package

These are measured runs in the build environment, not estimates.

- `models/ocr_svm.joblib`: HOG + LinearSVC glyph classifier. Procedural held-out validation accuracy: **1.0000** on 528 samples (2640 total generated glyphs, fixed seed 20261025).
- `models/detector_svm.joblib`: HOG + HSV LinearSVC candidate filter. Held-out synthetic validation: **accuracy 1.0000, precision 1.0000, recall 1.0000, F1 1.0000** on 720 validation crops (3600 total samples, fixed seed 20261025).
- `models/ocr_cnn.pt`: small CNN bootstrap checkpoint trained on procedural glyphs. It is retained as an experimental checkpoint; production inference uses the SVM + deterministic template fallback because its measured validation was stronger in this build.

These synthetic validation metrics do **not** claim hidden-test or real-camera accuracy. The supplied scripts support retraining/fine-tuning with legally sourced real data.
