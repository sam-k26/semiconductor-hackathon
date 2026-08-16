# CLAUDE.md

Agent instructions for this repository. Read this and `PRD.md` before proposing anything.

## Project

AI-based restoration of degraded semiconductor inspection images (KLA PS01). A supervised model maps degraded low-resolution `.npy` observations to clean full-resolution ground truth, handling speckle noise, Gaussian noise and 2× spatial resolution reduction jointly. The deliverable is a public repository whose `eval.py` will be run **as-is, without manual edits**, by KLA's benchmarking team on an H100 against a held-out test set we never see.

Source of truth for requirements: `PRD.md`. It contains `[VERIFY]` and `[DECIDE]` markers for open items. Do not treat a marked item as resolved.

## Stack

<!-- FILL IN from `python --version` and `pip freeze`. Do not let the agent guess these. -->
- Python: `___`
- PyTorch: `___`
- CUDA: `___`
- NumPy: `___`
- Metrics: scikit-image `___`, lpips `___`

Target inference hardware: NVIDIA H100. Development hardware: `___`

## Structure

```
eval.py              # standalone inference script — the most important file in the repo
train.py             # training entry point
src/
  model.py           # network definition
  data.py            # dataset + dataloader (the normalization contract lives here)
  metrics.py         # PSNR / SSIM / LPIPS with explicit conventions
  analysis.py        # error maps, consistency check, crop selection
checkpoints/         # not committed; weights linked externally
outputs/             # restored test outputs
requirements.txt     # pip freeze from the training environment
PRD.md
```

## Conventions

- Type hints on every function.
- No new dependencies without asking first. Every dependency added is a line in `requirements.txt` that must install cleanly on a fresh machine.
- Errors are raised, not swallowed. **One exception:** per-image error handling inside `eval.py`'s inference loop, where a single bad file is logged and skipped so it cannot abort a 400-image run.
- No hardcoded paths. Ever. Not in `eval.py`, not in `train.py`, not "temporarily".
- No hardcoded image dimensions. The model and script must handle arbitrary spatial input size.
- Do not write comments that restate what the code does. Do write comments recording *why* a convention was chosen where it is non-obvious — normalization, `data_range`, LPIPS backbone.

## Rules

- Read `PRD.md` before proposing architecture changes.
- **If a requirement is ambiguous, ask instead of guessing.** Applies especially to the submission contract, output format, and metric conventions. A confident guess here costs the entire submission.
- Do not refactor code outside the scope of the current task.
- Do not add features from the non-goals list in `PRD.md` §4 (GAN, loss ablations, architecture search, synthetic degradation, web UI). If one seems necessary, say so and stop.

## Project-specific rules

These exist because they are the specific ways this project can fail. They are not general good practice.

- **Never write a number you have not measured.** Not in code comments, not in the README, not in the deck. If a metric has not been computed, write `NOT MEASURED`. Do not estimate inference time from parameter count, do not infer PSNR from L1 loss, do not carry forward a number from a previous run without re-measuring.
- **Inference preprocessing must be byte-identical to training preprocessing.** Read `src/data.py` to determine what training does; do not reason from the data's properties. If they diverge, the network receives activations it never saw and every measured metric stops predicting test behaviour. Any change to preprocessing on one side requires the same change on the other, in the same commit.
- **Do not clip, normalize or rescale inputs unless training did.** Measured input range is roughly [−0.28, 2.16]; ground truth is [0, 1]. Values outside the ground-truth range are real information, not error.
- **`eval.py` is the most important file in the repository.** Changes to it require re-running the fresh-machine test: clone to a new directory, new venv, install from `requirements.txt` only, run against a hostile input directory (mixed file types, nested subdirectory, empty file, wrong-size image, odd file count). No exceptions for "small" changes.
- **Output filenames must match input filenames exactly.** No suffixes, no reformatting, no zero-pad changes.
- **Every metric call states its conventions explicitly.** PSNR with `data_range=1.0` passed in, never inferred from dtype. SSIM with window size and Gaussian weighting named. LPIPS with backbone, channel-replication and input range named.
- **All CUDA timing uses warmup + `torch.cuda.synchronize()` before the timer stops.** Timing without synchronization measures queueing, not execution, and produces impossibly good numbers.
- **Inference runs under `torch.inference_mode()`.**
- **Report validation results as distributions, not just means.** Alongside any aggregate metric, report the fraction of validation images on which we beat the baseline.
- **Crops and examples come from the validation split, never training.**

## Maintaining this file

When a correction has to be given twice, it becomes a rule here. Add it under Project-specific rules with one line on why it exists.
