# KJV-200K

A **205,376-parameter, randomly initialized causal transformer**, trained only on King James Bible verse text. Includes a local browser trainer for iPhone/Safari and a PyTorch trainer for a computer. No pretrained model, external tokenizer, synthetic text, or instruction tuning.

This repository is a training project. **It does not include a fully trained checkpoint.** Short correctness tests confirm learning and checkpoint resume; they are not evidence of final language quality. A trained model will produce KJV-style continuations, including invented or misquoted verses. It is not a reliable Bible search tool or question-answering assistant.

## Why start at 200k?

The training stream contains 3,945,547 byte tokens per corpus pass, approximately **19.2 token positions per model parameter**. That ratio is descriptive, not a proof of optimality: the often-cited 20-token guideline was derived for much larger models using different tokenization/data. Repeated passes provide more optimization opportunities, not new source information. This version keeps the same Bible-only training/validation split and starts with substantially less model capacity and computation.

Watch both losses: falling training loss with consistently worsening validation loss suggests overfitting; both losses plateauing at an unsatisfactory level can indicate insufficient capacity, insufficient context, or optimization limitations. Use the best validation checkpoint and compare samples. No fully trained model or comparison against the 1M version is included.

Existing 1M checkpoints have different tensor shapes and cannot be imported into this smaller model. Browser autosaves use a separate database so the previous model's saved state is not overwritten. The checkpoint container retains its legacy `kjv-1m-v1` format identifier; model configuration and shapes are checked explicitly. Corpus split IDs and hashes are unchanged for comparability.

## Start on an iPhone

You do not need Python, an App Store terminal, a Mac, Xcode, or an API key to use the browser trainer.

1. Extract the project ZIP and upload the **contents of `kjv-200k/`** to a GitHub repository, preserving the folders. `index.html`, `app.mjs`, `model.mjs`, `checkpoint.mjs`, `data/`, and `vendor/` must be at the repository root. Upload the extracted files, not just the ZIP. GitHub's desktop web upload accepts folders; a desktop is the simplest way to preserve the directory structure.
2. In the repository, open **Settings → Pages**. Choose **Deploy from a branch**, then the `main` branch and **/ (root)**. Save. Your account/repository must support Pages hosting.
3. Open the resulting `https://YOUR-USERNAME.github.io/YOUR-REPOSITORY/` address in Safari on the iPhone. Opening `index.html` inside Files is not enough: the module scripts and corpus need an HTTP(S) origin.
4. Start with **10–100 additional steps**, **batch 1**, **64-byte context**, learning rate **0.0005**. The first step compiles shaders and may be much slower than later steps.
5. Measure seconds per step. If stable, try **128-byte context** and **batch 2**, then batch 4. Prefer 128 or 256 bytes for a serious run; 64 bytes is primarily a phone benchmark. Context is bytes, not words or subword tokens.
6. Use **Stop & save** before leaving the page, and **Download checkpoint** to keep a copy in Files. Return to the same site and choose **Restore autosave**, or import the downloaded checkpoint.

GitHub Pages serves static code and the corpus. **The phone computes forward passes, gradients, and optimizer updates locally.** No model weights, prompts, or training data are uploaded by the app. The TensorFlow.js runtime is bundled locally, with no CDN dependency at runtime. The site still needs a network connection to load its files; there is no offline service-worker cache.

Keep Safari in the foreground. iOS can suspend a hidden or locked page; a wake lock is requested when supported, but it cannot guarantee uninterrupted execution. Autosaves occur every 100 completed steps and at normal run/stop completion. A crash or suspension can lose work after the last completed autosave. Browser storage may be evicted, so downloaded checkpoints matter.

**iPhone support is provisional until tested on your device.** The numerical engine is tested on Node/CPU and desktop headless Chrome with both CPU and software WebGL backends; no physical iPhone benchmark has been performed. Safari/WebGL support, GPU precision, available memory, and heat determine feasibility. The app falls back to the CPU backend when float32 WebGL rendering is unavailable. It can be very slow. Long runs may take many hours or more; do not infer a completion time from model size alone.

### How much training?

The training corpus contains **3,945,547 bytes**, and validation contains **206,420 bytes**. One effective corpus pass means that many training bytes have been sampled; sampling is random with replacement, so this does not mean every passage has been visited.

At batch 1 and context 64, one effective corpus pass is about **61,650 steps**. At batch 4 and context 128, it is about **7,706 steps**. A 100-step run checks functionality and speed; it is far too short to learn the Bible's language. Several passes may help, but holdout loss should decide when to stop. There is no promised number of passes or guaranteed output quality.

The UI reports measured throughput and effective corpus passes. Estimate duration using measured seconds/step after shader warmup. Browser training uses AdamW with beta=(0.9, 0.95), epsilon=1e-8, weight decay=0.1 on matrix weights, global gradient clipping at 1.0, and 100-step linear learning-rate warmup followed by a constant learning rate. Its schedule differs from the desktop trainer. The UI saves both the latest checkpoint and the best evaluated checkpoint. It evaluates every 200 steps and at run completion, and stops after 10 training evaluations without improvement. Use **Restore best** before generation or downloading the best weights. Changing training context or batch size restarts the validation comparison. These noisy validation estimates and early stopping help monitor overfitting but cannot guarantee its absence.

## Train on a computer

Requires Python 3.10+ and PyTorch 2.6+. From the extracted project directory:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python model.py
python train.py --steps 10000
python generate.py --checkpoint runs/kjv/best.pt --prompt "And the Lord said"
```

The trainer selects CUDA, Apple MPS, or CPU automatically. Use `--device cpu` to force CPU. Install a suitable CUDA build of PyTorch if needed using the official PyTorch installer. All training math defaults to float32. PyTorch's scaled dot-product attention chooses an available implementation; the browser uses ordinary dense attention.

Defaults: context 256 bytes, microbatch 8, 4 gradient accumulation steps, peak learning rate 5e-4, AdamW beta=(0.9, 0.95), decay 0.1 on matrices only, clip norm 1.0, 200-step warmup, cosine decay to 10% of peak, evaluation every 200 steps, and early stopping after 10 evaluations without improvement. At the defaults, 10,000 steps process 81,920,000 bytes, about 20.8 effective corpus passes. This is a starting budget, not an established optimum.

For a small CPU, reduce batch/accumulation and set a sensible CPU thread count. Example on macOS/Linux:

```bash
OMP_NUM_THREADS=4 python train.py --steps 10000 --batch-size 4 --accum 1 --context 128
```

`runs/kjv/best.pt` is the best evaluated checkpoint; `last.pt` is the latest evaluated checkpoint. Checkpoints are written atomically and contain weights, optimizer, data sampler RNG, PyTorch RNG, configuration, corpus provenance, and schedule settings. Ctrl-C falls back to the last evaluation checkpoint; it does not save a partial update.

```bash
python train.py --steps 10000 --resume runs/kjv/last.pt
```

Keep the model, corpus, batch, accumulation, total steps, seed, learning rate, warmup, and decay settings unchanged for a normal resume. To start a **new** optimizer and schedule from learned weights, use `--init` instead. Keep the checkpoint's `--context` when loading; context changes are not accepted implicitly.

```bash
python train.py --steps 20000 --init runs/kjv/best.pt --out runs/continued
```

Only load checkpoint files you trust.

## Architecture and why

There is no established “absolute best” architecture for this exact corpus, parameter budget, and iPhone training setup. This is a compact, practical **Llama-style dense decoder**. It is an engineering recommendation, not a benchmark winner or a scaled-down pretrained Llama checkpoint.

| Component | Choice |
|---|---|
| Transformer blocks | 4, pre-normalized residual blocks |
| Model width | 64 |
| Attention | 4 full causal attention heads, head dimension 16 |
| Feed-forward | SwiGLU, hidden width 160 |
| Normalization | RMSNorm, epsilon 1e-5 |
| Positions | RoPE, base 10,000, split-half rotation |
| Vocabulary | 256 UTF-8 byte values |
| Input and output embeddings | Tied, one shared matrix |
| Biases / dropout | None |
| Maximum configured context | 256 bytes |
| Training objective | Next-byte cross entropy |

Parameter accounting, including normalization weights:

| Part | Parameters |
|---|---:|
| Shared token embedding/output matrix | 16,384 |
| Attention projections, all 4 blocks | 65,536 |
| SwiGLU projections, all 4 blocks | 122,880 |
| Two RMSNorm scales per block | 512 |
| Final RMSNorm scale | 64 |
| **Total unique trainable parameters** | **205,376** |

That is 2.69% above 200,000 and about 80% fewer parameters than the original 1,017,472-parameter version. RoPE tables are nontrainable buffers and add no trainable parameters. Raw float32 model weights occupy 821,504 bytes (0.78 MiB). Gradients, Adam moments, activations, GPU texture padding, and runtime overhead make the real training footprint much larger.

Tied embeddings keep the vocabulary from consuming the budget. RMSNorm, RoPE, and SwiGLU are well-established modern decoder components. Ordinary multi-head attention keeps all four key/value heads at this tiny scale. GQA's main benefit is smaller inference KV caches; this implementation recomputes its short context during generation and does not implement a KV cache. MoE, attention sinks, and complex long-context machinery are not justified by this small starting experiment.

**The tokenizer is the main tradeoff.** Bytes are transparent, portable, lossless, and require no external tokenizer training data. They leave almost all parameters in the transformer, but sequences are longer than with subwords. A tokenizer trained only on the training chapters with a 512–1,024-token BPE vocabulary is a reasonable candidate for a faster/better next experiment, trading embedding parameters for fewer training tokens. It has not been implemented or compared here, so byte tokenization is not claimed to be the quality optimum. Compare alternatives with held-out bits/byte, throughput, and generated samples, not raw token loss across different tokenizers.

The 4-block, width-64 model is a smaller starting experiment, not a claim that 200k is optimal. It may overfit less than a larger model, or underfit if capacity is insufficient; compare held-out loss before increasing size. Model size cannot guarantee either outcome.

## Pure KJV data and evaluation

Included data is the conventional **66-book, 1769 KJV text**, excluding Apocrypha, extracted from [scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases/blob/master/formats/json/KJV.json). The upstream collection's title mentions Strong's numbers and morphology; this project reads only each verse's `text` field and rejects angle-bracket markup. It does not train on the JSON metadata.

`prepare.py` checks for 66 books, 1,189 chapters, and 31,102 verses. It keeps verse text, strips surrounding whitespace, inserts newlines between verses and blank lines between chapters, and excludes book headings, chapter/verse identifiers, introductions, and annotations. The shipped streams are immediately usable, without another download.

The deterministic chapter split holds out 59 entire chapters (~5%). Validation chapters never occur in the training stream. The fixed byte tokenizer has no learned vocabulary and cannot learn from validation text. Repeated language and parallel passages in different chapters can still overlap, so this split is not a strong test of factual knowledge or freedom from memorization. Random windows can cross the boundary between two training chapters; no window can cross into a validation chapter because the streams are separate.

The manifest records the original source URL, raw JSON SHA-256, preprocessing, split IDs, and both corpus SHA-256 hashes. Trainers verify the shipped stream hashes. A checksum records identity, not an independent theological/textual audit. The source URL tracks upstream `master`; rerunning data preparation later can produce a different snapshot if upstream changes. Preserve the shipped data and manifest to reproduce this project's dataset.

To regenerate, or use a local JSON in the same schema:

```bash
python prepare.py
python prepare.py --source /path/to/KJV.json
```

Training and validation loss are **nats per byte**. Divide by `ln(2)` for **bits per byte**; lower is better. This is not word perplexity. The desktop evaluates 20 fixed random batches; the browser evaluates 8, so its validation is a noisier estimate. Compare runs at the same context, with a larger final evaluation when choosing a model. Never treat sampled text as a verbatim Bible quotation without checking the source.

## Moving weights between phone and computer

Each framework's own checkpoint preserves its optimizer. Cross-framework conversion moves **weights only**, restarting optimizer and sampler state.

```bash
# Desktop weights → JSON to import in the browser:
python convert.py to-browser --input runs/kjv/best.pt --output desktop.checkpoint.json

# A downloaded phone checkpoint → PyTorch:
python convert.py to-torch --input kjv-200k-step-1000.checkpoint.json --output phone.pt
python generate.py --checkpoint phone.pt

# Continue with a new desktop optimizer/schedule. Match the imported config context (256 by default):
python train.py --init phone.pt --out runs/from-phone
```

Do not pass a converted weights-only checkpoint to `--resume`. Browser inference can use a shorter context than the model's configured 256-byte maximum. A native checkpoint configured with another context requires matching that context when initializing native training.

## Verify the project

```bash
python -m unittest discover -s tests -v
python tests/parity.py parity.json
node tests/browser.test.mjs parity.json
```

The tests verify the unique parameter count, causal masking, finite gradients, learning on a fixed batch, corpus identity, exact weight conversion, browser optimizer continuation, tensor cleanup, and generation. Numerical parity compares the full default model's logits between PyTorch and TensorFlow.js. `parity.json` is temporary and should not be committed. GitHub Actions runs the tests on CPU; the production browser app uses the available device backend.

For a local desktop browser, run `python -m http.server 8000` and open `http://localhost:8000`. To access it from another device, use appropriate HTTPS hosting; the app's checksum verification requires a secure context.

## References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971): pre-normalization/RMSNorm, SwiGLU, and rotary positions.
- [TensorFlow.js platform/environment guide](https://www.tensorflow.org/js/guide/platform_environment): browser/mobile backends, explicit memory management, and float32 versus float16 WebGL constraints.
- [MLX Swift](https://github.com/ml-explore/mlx-swift): another route for native on-device training; its iOS MNIST trainer demonstrates training support. A native transformer app would require a separate Swift implementation, build/signing workflow, and device testing. This project uses the browser to avoid that setup.

## Licenses and attribution

Original project code is MIT-licensed; see `LICENSE`. The bundled, unmodified TensorFlow.js 4.22.0 distribution is Apache-2.0-licensed; see `vendor/LICENSE.tensorflow-js`, its retained embedded notices, and `vendor/NOTICE.md`. Bible source attribution and provenance are in `data/README.md` and `data/manifest.json`. The code license does not relicense third-party content.
