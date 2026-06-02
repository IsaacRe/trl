# Multi-GPU (4×H200) adaptation — handoff

Goal: run `configs/Qwen3-32B/Kimi-K2.5-Reasoning-1M-Cleaned/4xH200/sft_lora.yaml`
across all 4 GPUs for **training** and **full/spec_dec eval**.

Current state: the script runs single-process (`python scripts/train_sft.py`); the
4×H200 config exists but the launch + eval code are NOT distributed yet. The 61 GB
base + LoRA fits on one 141 GB H200 → use **DDP** (replicate per GPU), not FSDP/DeepSpeed.

---

## 1. Training (required)

- Launch distributed instead of plain python:
  ```
  accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    --num_processes 4 scripts/train_sft.py \
    --config configs/Qwen3-32B/Kimi-K2.5-Reasoning-1M-Cleaned/4xH200/sft_lora.yaml
  ```
  Update the 4×H200 config header comment to this. The Trainer auto-detects the
  distributed env (`trainer.accelerator` already used), so no code change needed for
  DDP training itself. Gives ~4× train throughput.
- Batch math: per_device(2) × accum(4) × 4 GPUs = 32 effective (matches 1×RTX's 1×32).
  Consider LR scaling for the larger global batch; optionally
  `attn_implementation: flash_attention_2` on H200.
- Streaming train data: Trainer shards the IterableDataset per rank
  (`IterableDatasetShard`), so each rank sees distinct data — verify on first run.

## 2. Do NOT break DDP with the single-GPU load trick

If the single-GPU `device_map={"": 0}` load speedup is added (to skip the ~61 GB CPU
materialization), it MUST be gated to `world_size == 1` — under DDP it would pin all
ranks to GPU 0. Under DDP each rank loads to CPU then the Trainer places it on its
local GPU (so the slow ~61 GB materialization happens ×4 concurrently — watch CPU RAM).

## 3. Eval correctness under DDP (required)

The eval callback in `scripts/speculative_eval.py`
(`on_train_begin` / `on_step_end` / `_run_evals`) has **zero distributed-awareness** —
it fires on **all 4 ranks**.

- As-is it is **correct but 4× redundant**: eval is forward-only (no collectives), all
  ranks load identical eval samples (entries use explicit datasets, not the sharded
  train stream), and the wandb push is already rank-0-guarded (`wandb.run is not None`).
  So it runs correctly, it just doesn't get faster.
- Gate `_load_samples` (in `train_sft.py`) and the eval `logger.warning` lines to the
  main process — otherwise every eval dataset is downloaded/materialized ×4 (×4 full
  materialization for any `shuffle: true` entry) and logs spam ×4.
- Guard the `cfg.dataset == script_args.dataset_name` branch in `_load_samples`: it
  reads the per-rank **sharded** train stream → different samples per rank under DDP.
  Not hit by the current interleave configs (dataset_name is None) but unsafe if used.

## 4. Parallelize eval across the 4 GPUs (the one substantive code change)

To make eval ~4× faster instead of 4× redundant (worth it: full_eval fires every 10
steps, and spec-dec generation is the expensive part):

- In `_run` (spec-dec) and `_run_full_eval`: shard `entry.eval_samples` by rank
  (`samples[rank::world_size]`), run locally, then **gather to rank 0** —
  `accelerator.gather_object` for the spec-dec per-step records, all-reduce the
  full-eval count accumulators — aggregate + log on rank 0.
- Handle ranks with an **empty shard** (when `n_samples < world_size`): they must still
  call the gather/collective or it deadlocks.
- `_load_samples` then only needs each rank's shard (or keep loading all and slice).

---

## Pointers

- `scripts/speculative_eval.py` — eval callback + `_run` / `_run_full_eval` (the files
  to make rank-aware in §3–§4). Per-process async batcher is fine unchanged.
- `scripts/train_sft.py` — model load, `_load_samples` (§3 gating), callback wiring.
- `examples/accelerate_configs/multi_gpu.yaml` — DDP launch config.
- Env note: `.git/objects` has mixed ownership (some root-owned subdirs); commits from
  this environment used a temp `GIT_OBJECT_DIRECTORY` + pack consolidation.
