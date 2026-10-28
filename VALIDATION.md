# Validation record

Checked October 28, 2026. These are implementation checks, not a completed language-model training experiment.

- PyTorch 2.15.0+cpu / Python 3.12: all four unittest cases pass (parameter count and finite gradients; causal masking; fixed-batch learning and exact weight round-trip; shipped corpus checksums).
- TensorFlow.js 4.22.0 / Node 24 CPU: full 205,376-parameter model, fixed 8-byte example, loss 5.5853 → 4.6547 after three optimizer updates. This tiny overfit check does not establish language quality.
- TensorFlow.js CPU: checkpointed model plus Adam state reproduces the next update within 1e-6; repeated updates do not leak tensors; all tensors are released after tests.
- PyTorch versus TensorFlow.js: default full model, same weights and input, maximum absolute logit difference 3.5762786865234375e-7.
- Desktop headless Chrome 131 with software WebGL/float32: page load and corpus checksums, two training updates, validation, separate best/latest autosaves, reload/restore, checkpoint download/import, and generation pass. This is not an iPhone GPU or performance benchmark.
- Browser early stopping exercised by importing a checkpoint with a synthetic non-improvement history: training stops at the next validation after the tenth non-improvement, and Restore best returns the earlier best checkpoint. This test is about control flow, not evidence of real-world overfitting.
- Responsive page visually inspected at a 390-pixel viewport. No horizontal overflow observed.
- PyTorch CLI: 200 updates on real KJV training data, batch 4, context 128, accumulation 1, warmup 20, peak LR 5e-4, cosine schedule, seed 42. The model processes 102,400 sampled byte positions (~0.026 corpus passes). Validation uses 8 fixed batches (4,096 held-out positions) per evaluation. This is an early-learning check, not convergence or comparison with the 1M model. best.pt/last.pt and metrics are written; generation and loading last.pt with --resume pass.

| Update | Validation nats/byte | Validation bits/byte |
|---|---:|---:|
| 50 | 3.6246 | 5.229 |
| 100 | 2.9665 | 4.280 |
| 150 | 2.7873 | 4.021 |
| 200 | 2.7317 | 3.941 |

Not established: convergence, final text quality, the optimal architecture/tokenizer for KJV, physical iPhone compatibility/performance, thermal behavior, or uninterrupted long-duration mobile training. No trained checkpoint is shipped.
