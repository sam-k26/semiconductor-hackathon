# Handoffs

Newest first. One entry per completed phase. Keep entries short — if an entry needs more than a screen, the phase was too big.

---

## Handoff 2 — eval.py found and hardened; retrain moved to Colab (2026-08-17)

The user pushed the repo to GitHub (`origin: sam-k26/semiconductor-hackathon`) to move training onto Google Colab (this dev machine is CPU-only and too slow — ~25 min/epoch, made unusable by a from-scratch retrain and repeated OOM kills of the demo backend under memory pressure). The push revealed a teammate ("matin impex") had independently authored a full `eval.py` (340 lines, commit `1014598`), not written by this thread.

**Built:** `eval.py` reviewed in full and found to already satisfy nearly all of PRD §6.1 — proper CLI, `.npy`-only discovery, exact filename matching, per-file error isolation, no hardcoded shape, empty-dir handling, `inference_mode`+warmup+`cuda.synchronize()`, device auto-detect with explicit startup logging, TTA on by default with `--no-tta` override. Fixed the one real gap found: it reported a single combined timing number, not the end-to-end vs. model-only split Phase 5 requires — `process_batch` now returns `model_only_seconds` (forward pass only, bracketed by `cuda.synchronize()`) separately from the caller's end-to-end timer.

**Working:**
- `eval.py` run end-to-end 2026-08-17 against the real 400-image official test set: 400/400 restored, 0 failures, output filenames match input exactly, `(256,256)` float32 in `[0,1]`.
- New timing split verified: on this CPU machine, 684.40 ms/image end-to-end vs. 683.44 ms/image model-only (nearly identical here — 4-way TTA dominates on CPU; expect the gap to be far more visible on H100).
- Confirmed the GitHub push includes every fix from Handoff 1's session: `data/splits/split_v2_leakage_safe.json`, `RESUME_CHECKPOINT = None` in `train.py`, the corrected `baseline.py`/`evaluate.py` split-loading — so training on Colab from this repo state will use the corrected split, not the leaky one.

**Not built:** the corrected-split retrain itself — moved to Colab, not run to completion anywhere yet. LPIPS. Denoise-then-bicubic baseline. Data-consistency check (downsample prediction, compare to input) — this is cited in the current pitch narrative as a differentiator but does not exist in code. Deck. `requirements.txt` still unpinned.

**Known issues:**
- **`eval.py`'s `.npy`-in/`.npy`-out/float32 design is still an unverified assumption**, not a confirmed fact — Component 2 of the submission requirements (the official file-format contract) was never actually provided in this thread. Good code doesn't resolve that risk; only reading the actual problem PDF does.
- Local `checkpoints/best.pth` is unchanged — still trained on the old, leaky split. Whatever Colab produces needs to be brought back and re-measured before it replaces this checkpoint.

**Decisions made:**
- Training moved off this Mac entirely, onto Colab, via the GitHub push rather than a zip upload (this repo already has a Colab-ready README section from Handoff 0's era).
- Demo backend was stopped locally during the (since-abandoned) local retrain attempt to free memory, then restarted.

**Next phase needs:**
- Bring back whatever checkpoint Colab produces, verify it against the corrected split's bicubic baseline (22.7054 dB / 0.520718 SSIM, already measured), and only then update the reported headline numbers.
- Still need Component 2 of the submission requirements — everything eval.py does is a well-reasoned guess until that's read.
- Decide whether to build the data-consistency check before or after the retrain lands, since the pitch narrative already claims it.

---

## Handoff 1 — Repo consolidation and current-state audit (2026-08-16)

Two separate locations existed: `PRD.md`/`CLAUDE.md`/`HANDOFF.md` in one directory describing a structure that was never built, and a working `semicon-restore/` git repo (uncommitted, no remote) one level down with a real trained model, dataset loader, training/eval/predict scripts, and 400 predictions already generated against the official test set. This handoff merges them — `semicon-restore/` is now the project root.

**Built:** everything Handoff 0 described, plus: `KLADataset` (`src/datasets/kla_dataset.py`), `RestorationNet` (`src/models/restoration_net.py`, sigmoid final activation, fully convolutional), `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`, `scripts/baseline.py`, a checkpoint at `checkpoints/best.pth`, and 400 predictions in `results/predictions/` against the official `Test_NoisyLR` set.

**Working:**
- `checkpoints/best.pth` measured 2026-08-16 via `scripts/evaluate.py` on the real 320-image validation split: **27.6319 dB PSNR, 0.738996 SSIM**, with test-time augmentation (4-way flip averaging).
- Preprocessing confirmed by reading the dataloader: float32 cast only, no clip, no normalize.
- Final activation confirmed: `torch.sigmoid`.
- The official test input format is now known first-hand: `data/official/Test_NoisyLR/NoisyLR/*.npy`, 128×128, float32, 400 files, no ground truth provided (as expected).

**Not built:** `eval.py` still does not exist — `predict.py` is not it (hardcoded paths, no CLI args, no per-file error handling, assumes fixed 128×128 input via `KLADataset`'s shape assertion). LPIPS. Denoise-then-bicubic baseline. Re-split at cluster level (leakage now confirmed, see below — detection is done, the re-split itself is not). Timing on any hardware, let alone H100. Error analysis, crop selection.

**Known issues:**
- **The 28.50 dB / 0.7624 SSIM number in the old `README.md` is not reproducible.** It was measured against `checkpoints/best_28_4458.pth`, which does not exist in this repo. It was *assumed* to be `best.pth` renamed; direct measurement disproved that (27.6319 dB, ~0.87 dB worse). Treat `best_28_4458.pth` as lost unless it turns up elsewhere. `README.md` and `PRD.md` have been corrected to the measured number.
- ~~`scripts/baseline.py`'s bicubic figure was computed over all 3,200 images, not the 320-image validation split~~ **FIXED**: `baseline.py` now recreates the identical seed-42 split. Corrected bicubic on the real validation set: 23.0831 dB / 0.546892 (higher than the old, wrong, full-dataset figure). Corrected improvement: **+4.5488 dB PSNR, +0.192104 SSIM**.
- **Split leakage confirmed and corrected split built, but no checkpoint trained on it yet.** `scripts/check_split_leakage.py` found a genuine near-duplicate spike (p0–p0.1 = 0.0 vs. p5 = 0.020 in 16×16-thumbnail nearest-neighbour distance). `scripts/build_split.py` found **50 near-duplicate clusters** total (union-find, threshold 0.01) and wrote a corrected split, `data/splits/split_v2_leakage_safe.json`, with 0 cross-split duplicate edges. `train.py`/`evaluate.py`/`baseline.py` are wired to read it. Bicubic on the corrected split: 22.7054 dB / 0.520718 SSIM. **Retraining from scratch was started and stopped early** (this Mac is CPU-only, ~30 batches/min, a full run would take many hours) — `checkpoints/best.pth` is still the only trained checkpoint, and it was trained on the *old* split. **Do not evaluate `best.pth` against the new split** — some new-validation images were in its old training set, which is worse leakage than what was fixed. The corrected split is ready to use; a from-scratch retrain against it (ideally on real GPU hardware, not this Mac) is still needed before it produces a trustworthy number.
- `KLADataset.__getitem__` hardcodes shape assertions (`== (128,128)`, `== (256,256)`) — violates the no-hardcoded-dimensions convention and will reject the 256→512 case the problem statement confirms is in scope, even though the model architecture itself has no fixed-size dependency.
- `requirements.txt` is unpinned package names, not a real `pip freeze`. Was not regenerated from this machine's venv because this machine (Apple Silicon Mac, CPU-only) is not the training environment (Windows CPU-only, per the old README) — pinning from the wrong machine would misrepresent what was actually used.
- Training loss is actually `0.7·L1 + 0.3·MSE` (`scripts/train.py`), not pure L1 as the PRD previously stated. Corrected.
- The reported PSNR/SSIM used test-time augmentation (4-way flip averaging). Whatever `eval.py` ends up doing must match, or the deck reports a number KLA's run can't reproduce. Not yet decided either way.
- The submission's file-format/directory contract for `eval.py` (Phase 0's actual blocker) is still unknown — only the Idea Submission Template's slide-by-slide deck requirements have been read so far (Component 1). Component 2 (GitHub repo requirements) hasn't been provided yet.

**Decisions made:**
- `semicon-restore/` is the project root going forward; the old flat-file location is retired.
- Checkpoint references in `predict.py`/`evaluate.py`/`train.py` repointed from the missing `best_28_4458.pth` to the present `best.pth`.

**Next phase needs:**
- Component 2 of the i4C submission requirements (GitHub repo contract) — still blocks Phase 0 sign-off.
- Decide the TTA question before writing `eval.py`.
- Fix `KLADataset`'s hardcoded shape assertions before `eval.py` is built on top of it.

---

## Handoff 0 — State at PRD adoption (2026-08-16)

Retroactive. Work done before phases were defined, recorded so Phase 0 starts from a known state rather than an assumed one.

**Built:** training pipeline end to end. Dataset loading, model definition, training loop, one bicubic baseline measurement.

**Working:**
- Dataset loads. 3,200 paired samples, split 2,880 train / 320 val, random pair-level split, seed 42.
- Model trains. 776,705 parameters. L1 loss, λ=1.0.
- Validation L1: 0.035933 after epoch 1, 0.034382 after epoch 2.
- Bicubic ×2 baseline: PSNR 22.8530 dB, SSIM 0.5361.
- Measured data ranges: degraded input ≈ [−0.2786, 2.1580], ground truth [0, 1].

**Not built:**
- `eval.py` — no standalone inference script exists.
- Metrics harness — no PSNR/SSIM/LPIPS computed for our model at all.
- Validation output set — model has never been run over the full val split and written to disk.
- Timing — nothing measured, on any hardware.
- Error analysis, consistency check, crop selection.
- README, `requirements.txt`, deck.

**Known issues:**
- The model is 2 epochs in. It is not trained. Treat every number above as evidence the loop runs, nothing more.
- Validation split is **not proven leakage-free**. Filenames are flat sequential (`000000.npy`, …) with no exposed source grouping; distinct source count and patches-per-source are unknown. If sequential numbering preserves acquisition order, adjacent indices are the most likely near-duplicates and a random split scatters them across train and val.
- Bicubic PSNR of 22.8530 dB may have been computed with a dtype-inferred `data_range`. Unverified, and wrong if so.
- Model trained on 128→256 only. The problem statement describes a 256→512 case not present in the provided package. Behaviour on 256 inputs is untested.
- Inference preprocessing has never been checked against training preprocessing, because inference does not exist yet.

**Decisions made:**
- Train on the organizer's real paired data. No synthetic degradation for the primary training set — avoids mismatching their speckle formulation.
- L1 as the baseline loss. Full four-way loss ablation cut; 320 validation images cannot resolve the differences it would produce.
- GAN cut. Hallucination risk on inspection imagery, and untunable in the time available.
- Report end-to-end *and* model-only inference time separately, so I/O overhead is not hidden behind a flattering GPU number.
- Slide 6 crops selected by automated rule from validation error statistics, including one deliberate failure case.

**Next phase needs:**
- Phase 0 is a read, not a build: the official submission contract (input/output extension, dtype, shape, filename convention, directory structure) from the problem PDF. Nothing else in P0 is correctly designed until this is written into `PRD.md` §6.1.
- Second read: `src/data.py`, to record what training actually does to inputs. Required before `eval.py` can be written.

---

## Template

```markdown
## Phase N handoff — <name> (date)

**Built:** what exists now, and where.
**Working:** what has been verified, with the command or test that verifies it.
**Not built:** what a reader might assume is done and is not.
**Known issues:** what will break, and under what conditions.
**Decisions made:** what was chosen, and the one-line reason.
**Next phase needs:** preconditions, seed steps, files to read first.
```

Two rules for writing these:

- **"Working" means measured, not implemented.** If a script runs but its output has never been checked, it goes under Not built.
- **"Not built" is the load-bearing section.** It is the one that stops the next session from assuming a thing exists. Write it before the others.