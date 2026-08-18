# PRD — KLA PS01: AI-Based Restoration of Degraded Images

Status: draft
Owner: TBD
Last updated: 2026-08-16

Markers used in this document:
- **[VERIFY]** — a factual claim we have not yet confirmed against the official problem statement or dataset. Must be resolved before it can be relied on.
- **[DECIDE]** — an open decision. Must be made by a human, not inherited from a draft.

---

## 1. Problem statement

Microscopic inspection images used in semiconductor manufacturing arrive degraded by pixel-level noise and reduced spatial resolution. Degradation can conceal defects that determine whether a chip passes or fails, and engineers currently work with the degraded images as-is. This project produces a model that recovers clean, full-resolution images from degraded observations.

## 2. User and use case

There are two users, and they want different things. We are optimizing for the first and constrained by the second.

**Primary user — KLA benchmarking team.** Runs our `eval.py` as-is on an H100 against a held-out test set we never see. Measures SSIM, PSNR, LPIPS and inference time. Never contacts us. If the script requires a manual edit, the submission is unscored, and unscored submissions cannot win. This user's needs are met by correctness and robustness of the submission artifact, not by model sophistication.

**Secondary user — fab inspection engineer.** Would use restored images to make defect calls. Cares that recovered detail is *real*. This user is not scoring us, but their needs are what make hallucination a defect rather than a metric footnote — which is the basis of our Innovation slide.

## 3. Scope

- A supervised restoration model mapping degraded low-resolution observations to clean full-resolution ground truth, handling speckle noise, Gaussian noise and 2× spatial resolution reduction jointly in a single network.
- A standalone evaluation script that takes an input directory and an output directory, runs inference over every image, and writes restored outputs. No notebooks, no manual edits.
- A metrics harness computing PSNR, SSIM and LPIPS on a held-out split, plus baselines.
- A data-consistency check: downsample the prediction, compare against the observed input, to detect reconstructed content unsupported by the observation.
- Spatial error analysis (error localisation conditioned on ground-truth gradient magnitude).
- Measured end-to-end and model-only inference timing.
- A 9-slide deck and a public GitHub repository per the submission contract.

## 4. Non-goals

These are deliberately excluded. Adding any of them requires an explicit decision to change this document.

- **GAN or adversarial training.** Increases hallucination risk on inspection imagery and cannot be tuned reliably in the time available.
- **Four-way loss ablation (L1 / MSE / mixed / perceptual).** We will train one loss to convergence and report at most one comparison. A 320-image validation set cannot resolve the differences these ablations produce.
- **Architecture search or model scaling.** Current model is 776,705 parameters. It stays that size unless measured metrics justify otherwise *and* P0 is complete.
- **Synthetic degradation generator.** We train on the organizer's real paired data. Any synthetic pipeline risks mismatching their speckle formulation.
- **Web UI, frontend, or interactive demo.** Not part of the submission contract and does not affect scoring.
- **Extensive augmentation experiments.**

## 5. Data model

**Sample.** A paired observation.
- `input`: `.npy`, float32, 128×128 in the provided training package. **[CONFIRMED, new risk — see below]** The problem statement explicitly describes two size regimes: 512×512→256×256 and 256×256→128×128. Both are in scope for the held-out test set.
- `ground_truth`: `.npy`, float32, 256×256, values in [0, 1]
- Filenames: flat sequential (`000000.npy`, …). No source/wafer grouping is exposed.
- Grayscale, single channel — confirmed by problem statement. Relevant to LPIPS channel-replication convention in §6.3.
- Test set explicitly contains both in-distribution and out-of-distribution (different source) samples — confirmed by problem statement §"Test Data." Generalization, not memorization, is being scored.
- Problem statement states inference speed is benchmarked ("a model that takes 10 minutes per image is less useful than one that takes 10 seconds") but gives no numeric threshold. No pass/fail timing bar exists yet.

**[CONFIRMED]** The downloaded training package (`train/GT`, `train/NoisyLR`) contains exactly 3,200 pairs, all 128×128 (input, float32) ↔ 256×256 (ground truth, float32). Full-corpus scan (not a sample): input range `[-0.278563, 2.158005]`, ground truth range `[0.0, 1.0]` exactly. No 256×256 input or 512×512 ground-truth files exist anywhere in the package — the 256→512 regime described by the problem statement has zero training representation.

**Second noisy source — `NoisyLR 2 copy/` (400 files, 128×128, filenames `000000`–`000399`).** Confirmed by the user to be a second, independently-sampled noisy realization of the same 400 ground-truth images already in `train/GT` (i.e. `train/GT/000000.npy` has two valid noisy inputs: `train/NoisyLR/000000.npy` and `NoisyLR 2 copy/000000.npy`, pixel-different, no separate GT needed). This is real data, not synthetic — fine to use for training under the §4 non-goal. **Rule:** if incorporated, an index's original and second-realization inputs must be assigned to the same split (both train or both val). Splitting them across train/val would put identical ground-truth content on both sides of the split under different noise, which is a leakage case, not a near-duplicate — worse than the near-duplicate risk in §5 because it's a *guaranteed* match, not a suspected one.

**[DECIDE — new, blocks nothing yet but should be resolved before Phase 4]** The provided training package contains only 128×256 pairs. The problem statement confirms 256×512 is also a real degradation case in the test distribution. We have zero training examples of that regime. Options once confirmed:
  - Rely on the network being fully convolutional / resolution-agnostic and hope 128→256 learning transfers to 256→512. Unverified, and likely to underperform since the noise statistics may not scale identically.
  - Ask whether the full data package actually contains a 256→512 subset we haven't located.
  - Any synthetic construction of 256→512 pairs conflicts with the non-goal in §4 ("no synthetic degradation generator... avoids mismatching their speckle formulation") and would need an explicit decision to override that non-goal.
  Do not pick one of these silently — surface it before Phase 4 training starts.

**Measured ranges.**

| Array | Min | Max |
|---|---|---|
| Degraded input | ≈ −0.2786 | ≈ 2.1580 |
| Ground truth | 0 | 1 |

Input values fall outside the ground-truth range. This is consistent with the problem statement's description of speckle pushing pixel values beyond the true image range, and implies clipping occurred in their pipeline.

**Split.**
- Total: 3,200 pairs. Train: 2,880. Validation: 320 (10%). Random pair-level split, seed 42.
- **[CONFIRMED — leakage found]** Ran the detection method below via `scripts/check_split_leakage.py` on 2026-08-16. Nearest-neighbour distance distribution on 16×16 GT thumbnails shows a sharp spike at the bottom: p0–p0.1 = 0.0, vs. p5 = 0.020, median = 1.50 — a clearly separated near-duplicate cluster, not a smooth tail. Verified two of these pairs against the full-resolution arrays (not just thumbnails): `000257`↔`000258` and `001538`↔`001539`, max pixel difference ≈0.004 on a [0,1] scale (visually imperceptible). Both pairs are **adjacent indices**, matching the predicted failure mode exactly. Both pairs **span the current train/val split**: `000257` train / `000258` val, `001538` val / `001539` train.
- **Consequence:** every metric reported so far (27.6319 dB / 0.738996 SSIM) was measured on a validation set containing at least 2 images whose near-twin was in the training set. The split is confirmed not leakage-free, not merely unproven.
- Detection method (as run): compute a 16×16 thumbnail per ground-truth image (block-averaged, not resized), nearest-neighbour L2 distance in thumbnail space, inspect the distribution for a spike. Confirmed effective — found real leakage on the first run.
- **[DONE — split corrected, retrain deferred]** Built `scripts/build_split.py`: union-find clustering over all image pairs with 16×16-thumbnail L2 distance below 0.01, found **50 near-duplicate clusters** (49 pairs + 1 triple, all adjacent-or-near-adjacent indices), reassigned whole clusters to train or val, wrote `data/splits/split_v2_leakage_safe.json` (2,880 train / 320 val, 0 cross-split duplicate edges remaining — verified programmatically). `train.py`/`evaluate.py`/`baseline.py` now load this split via `src/datasets/split.py` instead of recomputing a seeded permutation.
  - Bicubic on the corrected split (measured): **22.7054 dB / 0.520718 SSIM**.
  - **Retraining on the corrected split was started, then stopped before completion** — this dev machine is CPU-only (~30 batches/min measured), and a from-scratch run (required: warm-starting from `checkpoints/best.pth` would carry over information about images now held out, since ~50 image pairs moved sides) would take many hours to reach convergence. Stopped after ~350/720 batches of epoch 1, no checkpoint was overwritten.
  - **Consequence:** `checkpoints/best.pth` (27.6319 dB / 0.738996 SSIM) is still the only trained checkpoint, and it was trained **and validated on the old, leaky split**. `data/splits/split_v2_leakage_safe.json` exists and the scripts can use it, but no checkpoint has been trained against it yet. **Do not run `evaluate.py`/`baseline.py` against `best.pth` using the new split** — the new validation set contains some images that were in `best.pth`'s *training* set under the old split, which would be worse than the original near-duplicate leakage, not better. Retraining from scratch on the corrected split (on faster hardware, or accepting a multi-hour CPU run) is required before the corrected split can be used to report any number.

## 6. Interfaces and contracts

### 6.1 Evaluation script

```
python eval.py --input <input_dir> --output <output_dir>
```

**[BUILT — found via GitHub push, not written by this thread]** `eval.py` (340 lines) already exists, committed by a teammate ("matin impex", commit `1014598`, 2026-08-16). Reviewed in full and verified end-to-end 2026-08-17 against the real 400-image official test set (`data/official/Test_NoisyLR/NoisyLR`): 400/400 restored, 0 failures, output filenames match input exactly, output `(256,256)` float32 in `[0,1]`.

Requirements and their status:
- Discovers files by explicit extension filter; ignores anything else (`.DS_Store`, `README`, nested directories). **DONE** — `discover_input_files` filters on `.suffix.lower() == ".npy"`, skips hidden files, does not recurse.
- Output filenames match input filenames **exactly**. No suffixes. **DONE, verified** — see above.
- Per-file try/except: a single bad file is logged and skipped, never aborts the run. **DONE** — load errors and per-image restore errors are both caught and counted as failures without aborting.
- Handles arbitrary spatial input size, not only 128×128. **DONE** — only checks `arr.ndim == 2`, no fixed-size assertion; consistent with the `KLADataset` fix made the same day.
- Handles a final partial batch. **DONE** — batch flushes on `file_idx == total_files - 1`.
- Defined behaviour on an empty input directory. **DONE** — logs a warning and exits 0.
- Runs under `torch.inference_mode()`, with a warmup, and `torch.cuda.synchronize()` before any timer stops. **DONE.**
- Device selection follows the same pattern already used in `scripts/train.py`/`evaluate.py`/`predict.py`: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. On KLA's H100, this must resolve to `cuda` — **eval.py must log the selected device explicitly on startup**, so a broken CUDA environment during grading fails loudly (wrong-looking timing numbers, visible in the log) instead of silently completing on CPU with numbers that look plausible but aren't what H100 would produce. Do not hardcode `"cuda"` unconditionally — that would crash on a machine without a GPU rather than degrading, which is worse for local testing/CI even though the grading machine always has one. **DONE** — logs `Target Device: {device.type.upper()}` on startup.
- No hardcoded paths. No manual edits required. **DONE** — checkpoint default is relative to `SCRIPT_DIR`, input/output are required CLI args.
- **[FIXED 2026-08-17]** End-to-end vs. model-only timing, reported separately (Phase 5 requirement) — originally only reported one combined number. `process_batch` now brackets the forward pass alone with `torch.cuda.synchronize()` and returns `model_only_seconds` separately from the outer end-to-end timer. Verified: on this CPU dev machine, end-to-end 684.40 ms/image vs. model-only 683.44 ms/image (nearly identical here since 4-way TTA dominates on CPU; expect the gap to be much more visible on H100, where the forward pass is fast and I/O becomes a larger fraction).
- TTA: ships with 4-way flip TTA on by default, `--no-tta` to disable — resolves the "does eval.py match the TTA used to report numbers" open question below.

**[VERIFY — still highest priority, blocks all other P0 work, NOT resolved by eval.py existing]** The official submission contract for: input extension, output extension, output dtype, output shape, output filename convention, directory structure. `eval.py`'s `.npy`-in/`.npy`-out/float32 design is a reasonable, working assumption — consistent with everything else built so far — but it has not been checked against the actual KLA problem PDF's submission section (Component 2 was never provided in this thread). Good code built on an unverified assumption is still an unverified assumption. Read the problem PDF before trusting this. If they diff `.npy` float arrays and we write PNGs, we quantize to 8 bits before scoring and the hardening work targets the wrong output path.

### 6.2 Normalization contract

The rule is **symmetry with training**, not independent reasoning about the data.

- Read `train_dataset.py` / the dataloader and record exactly what it does to inputs.
- Inference must apply the identical transform, even if we would now choose differently. A mismatch means the network sees activations it never encountered during training, and every metric we measured stops predicting test behaviour.
- **[RESOLVED]** Read from `src/datasets/kla_dataset.py:69-70`: `float32 cast: yes (np.load(...).astype(np.float32)) / clip: no / normalize: no`. Matches the "don't clip/normalize unless training did" convention.
- **[RESOLVED]** Final activation: `torch.sigmoid` (`src/models/restoration_net.py:152`). The `[0,1]` clamp applied in `predict.py`/`evaluate.py` after inference is therefore a no-op, not a correction — consistent with the note this item was written against.
- **[BUG — found, not yet fixed]** `KLADataset.__getitem__` (`src/datasets/kla_dataset.py:73-83`) raises `ValueError` if input shape is not exactly `(128,128)` or GT shape is not exactly `(256,256)`. This violates the "no hardcoded image dimensions" convention and would reject any 256×256 input, even though the model itself (`restoration_net.py`) is fully convolutional with a single `PixelShuffle(2)` and has no architectural dependency on a fixed spatial size. `eval.py` must not inherit this assertion — it needs to accept whatever spatial size arrives and rely on the conv/PixelShuffle architecture to produce a correctly-doubled output.

### 6.3 Metrics harness

- PSNR with `data_range=1.0` explicitly set. **[RESOLVED]** Confirmed explicit in both `baseline.py` and `evaluate.py`; the bicubic figure has since been recomputed on the correct population (see §6.4).
- SSIM: record window size and Gaussian weighting; these are backend-dependent and shift results non-trivially.
- LPIPS: record backbone (AlexNet vs VGG give different numbers), channel-replication convention for single-channel input, and input range convention.
- If the official contract does not specify these, state our convention explicitly on the results slide.

### 6.4 Baselines

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Bicubic ×2 | **23.0831 dB** — measured 2026-08-16 via `scripts/baseline.py` on the same 320-image validation split as the model | **0.546892** | not measured |
| Denoise → bicubic | not measured | not measured | not measured |
| Ours (`checkpoints/best.pth`, with TTA) | **27.6319 dB** — measured 2026-08-16 via `scripts/evaluate.py` against the real 320-image validation split | **0.738996** | not measured |

**[CORRECTED]** `README.md` claimed 28.50 dB / 0.7624 for a checkpoint named `best_28_4458.pth`, which does not exist in this repo (see §7 `requirements.txt` row and the checkpoint bug below). The assumption that `checkpoints/best.pth` was that checkpoint under a different name is **false** — re-running `evaluate.py` against `best.pth` measured 27.6319 dB / 0.738996, about 0.87 dB / 0.023 SSIM worse. `best_28_4458.pth` is genuinely lost; decision made to accept `best.pth` and move forward rather than chase it.

**[FIXED]** The bicubic population mismatch is resolved: `scripts/baseline.py` now recreates the identical seed-42 validation split used by `scripts/evaluate.py` and reports 23.0831 dB / 0.546892 on those same 320 images — slightly *higher* than the old (wrong) full-dataset figure of 22.8530 dB / 0.5361, so the corrected comparison is actually **less** flattering to the model than previously reported: **+4.5488 dB PSNR, +0.192104 SSIM** improvement over bicubic (not +4.78 dB as briefly computed against the uncorrected baseline).

**[DECIDE — new]** `evaluate.py`'s 28.50 dB figure was measured *with test-time augmentation* (4-way flip-group averaging, `scripts/predict.py:36-55`, `scripts/evaluate.py:30-49`). Whatever `eval.py` ultimately does (TTA or not) must be the exact code path used to produce every reported number — otherwise the deck reports a number KLA's benchmarking run cannot reproduce. TTA also roughly quadruples per-image forward-pass cost, which must be reflected honestly in Phase 5 timing if kept.

Bicubic alone is a weak bar — it upsamples noise without denoising. Denoise-then-bicubic is the classical pipeline we claim to replace and is the baseline that actually tests us.

Alongside the table, report **the fraction of validation images on which our model beats each baseline, per metric.** Aggregate means alone hide distribution.

## 7. Stack decisions

| Choice | Decision | Why |
|---|---|---|
| Framework | PyTorch | Model already trained in it; benchmark environment is CUDA/H100. |
| Model | Compact conv restoration net, 776,705 params | Small enough that inference latency is a non-issue on H100; large enough to have shown learning. Capacity is a known constraint for joint denoise + deblur + 2× SR — revisit only after P0. |
| Loss | ~~L1, λ=1.0~~ **[CORRECTED]** `0.7·L1 + 0.3·MSE` per `scripts/train.py:47-48` | This row previously didn't match the actual training code. Not a non-goal violation (§4 only forbids a *four-way ablation*), but the PRD must describe what was actually trained. |
| Data I/O | NumPy `.npy` | Matches provided dataset **[VERIFY against output contract]**. |
| Metrics | PSNR / SSIM / LPIPS | Specified by the problem statement. LPIPS not yet in `requirements.txt` or implemented anywhere in the repo. |
| Weights distribution | **[DECIDE]** Git LFS vs Google Drive / HuggingFace link | Required to be downloadable; choose based on final file size. `checkpoints/best.pth` is 9.36 MB — small enough that Git LFS is unlikely to be necessary, but this is still an open decision. |
| `requirements.txt` | **[BUG — found, not yet fixed]** Currently unpinned package names (`numpy`, `torch`, …), not an actual `pip freeze` | Violates the `CLAUDE.md` requirement that it be a real freeze from the training environment. Must be regenerated from a working install before the fresh-machine test in Phase 1 means anything. |

## 8. Build phases

Each phase is independently testable and ordered by what blocks scoring.

**Phase 0 — Submission contract (blocks everything).**
Read the problem PDF. Record input/output extension, dtype, shape, filename convention, directory structure in §6.1. Nothing downstream is designed correctly until this is written down.

**Phase 1 — Harden the evaluation script.**
Implement every requirement in §6.1. Test against a deliberately hostile directory: mixed file types, an empty file, a wrong-size image, a nested subdirectory, an odd file count that breaks batch remainder, a `float64` file. Then clone the repo into a fresh directory, fresh venv, install only from `requirements.txt`, and run it. This is the single highest-variance risk in the project and it is entirely mechanical to eliminate.

**Phase 2 — Metrics harness and baselines.**
Implement PSNR/SSIM/LPIPS per §6.3. Recompute the bicubic baseline. Add denoise-then-bicubic. Run the existing 2-epoch checkpoint through it — this is a five-minute check that tells us whether we beat bicubic at all.

**Phase 3 — Split integrity.**
Run the near-duplicate detection in §5. If clustering appears, re-split at cluster level and re-run Phase 2. Every number reported afterwards depends on this being resolved.

**Phase 4 — Train to convergence.**
Train until validation metrics stop improving. **[DECIDE] Checkpoint selection rule — best-val or final-epoch — written down before training starts.** With 320 validation images, best-val selection is mildly overfit to that split. Pick the rule now so it is not chosen at 3am based on which number looks better.

**Phase 5 — Timing.**
Measure on H100: end-to-end wall-clock per image (discover → read → tensor → H2D → forward → D2H → convert → write) and model-only forward latency, reported separately. Record images, batch size, warmup batches, total wall time, throughput. Measured, never estimated from parameter count.

**Phase 6 — Analysis (P1).**
Spatial error maps with error conditioned on ground-truth gradient magnitude — report mean error inside vs outside a high-structure mask as a single number per image. Data-consistency check: downsample prediction, compare against input, with the noise floor established by downsampling ground truth and comparing against input. Rank validation images by consistency residual; the worst ones are where fabrication lives.

**Phase 7 — Deliverables.**
README (clone → install → run inference, with no file edits), 9-slide deck exported to `TeamName_KLA_PS01.pdf`, restored test outputs folder, `requirements.txt` from `pip freeze`, weights uploaded and linked.

### Slide 6 crop selection

Four crops, 4× zoom, from the **validation** set, arranged degraded / bicubic / ours / ground truth. Selected automatically by rule, not by eye:

- 2 × representative success: largest `E_bicubic − E_model`, plus one near median validation error
- 1 × difficult case: highest ground-truth gradient magnitude
- 1 × deliberate failure: largest `E_model − E_bicubic`, annotated with what the model got wrong

Per-image error ranks images, not crop locations — a second spatial step is needed to choose where within the image to zoom.

## 9. Definition of done

The submission is finished when all of the following are true:

- [ ] `eval.py` runs on a fresh machine, fresh venv, from `requirements.txt` alone, against a hostile test directory, with zero manual edits.
- [ ] Output extension, dtype, shape and filenames verified against the official contract, not assumed.
- [ ] Inference-time preprocessing verified identical to training-time preprocessing.
- [ ] PSNR, SSIM and LPIPS measured on the held-out split for our model, bicubic, and denoise-then-bicubic, with metric conventions recorded.
- [ ] Validation split either proven free of source-level leakage or re-split at cluster level.
- [ ] Checkpoint selection rule recorded before training, and followed.
- [ ] End-to-end and model-only inference time measured on H100 with warmup and CUDA synchronization.
- [ ] Four validation crops generated by the automated selection rule, including the failure case.
- [ ] README lets a stranger clone and run inference without contacting us.
- [ ] Deck ≤ 9 slides, exported as `TeamName_KLA_PS01.pdf`, instruction slide removed.
- [ ] Every number on every slide traceable to a run that actually happened. Nothing marked "expected" or "projected."

## 10. Open items

Consolidated for tracking. Nothing in §9 can be checked off while the corresponding item here is open.

| # | Item | Type | Blocks |
|---|---|---|---|
| 1 | Official submission contract: I/O extension, dtype, shape, filenames | VERIFY | Everything |
| 2 | Training-time preprocessing, read from the dataloader | VERIFY | §6.2, all metrics |
| 3 | Final activation (sigmoid vs linear) | VERIFY | Output clamp policy |
| 4 | ~~Whether 256×256 inputs can appear in the held-out set~~ — CONFIRMED yes, by problem statement | RESOLVED | — |
| 9 | No training data exists for the 256→512 regime, which the problem statement confirms is in scope for the held-out set | DECIDE | Phase 4, model scale-agnosticism |
| 10 | Submission/deliverable format (file extension, dtype, directory layout, `eval.py` I/O contract) — not covered by the background/description text read so far | VERIFY | Everything (Phase 0) |
| 11 | Whether to incorporate `NoisyLR 2 copy/`'s 400 extra pairs into training, and enforce same-split-as-original-index if so | DECIDE | Phase 3, Phase 4 |
| 5 | Source-level grouping in the 3,200 samples | VERIFY | Split validity |
| 6 | Bicubic PSNR recomputed with explicit `data_range=1.0` | VERIFY | Baseline table |
| 7 | Checkpoint selection rule | DECIDE | Phase 4 |
| 8 | Weights hosting: Git LFS vs external link | DECIDE | Phase 7 |
