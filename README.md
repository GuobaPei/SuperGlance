# SuperGlance: Turning a Vision-Language Model into an Encoder-Style Driving Policy

[Paper](TBD) | [QA data](https://huggingface.co/HF_ORG/SuperGlance-QA) | [Weights](https://huggingface.co/HF_ORG/SuperGlance)

Authors: see the paper.

SuperGlance reads one hidden state of a vision-language model through two heads, each in one forward pass. The
language-model head answers multiple-choice questions by its option-letter logits at the answer position. A linear
Policy Head on the same hidden state scores a fixed set of candidate trajectories (8 poses, 2 Hz, 4 s), and the plan is
the probability-weighted mean of the 64 most probable candidates at temperature 0.2. The VLM is Qwen3.5-9B with
rank-32 LoRA adapters, trained in three stages:

| stage | data | objective | start | steps x rows |
|---|---|---|---|---|
| 1 general QA | 1.16 M visual and text questions | joint Brier policy gradient on two option orders (32 draws) + 0.1 KL to the base | Qwen3.5-9B | 7,813 x 64 |
| 2 driving QA | DriveLM questions + stage-1 questions | the same, KL to the stage-1 model | stage 1, merged | 3,000 x 64 |
| 3 supervised | NAVSIM, PAI-AV, WOD-E2E rows + QA replay | log score toward a soft target over 5,121 candidates; QA rows as stage 2, KL 0.1 to stage 2 | stage 2, merged | 20,900 x 128 |
| 3 last phase | the same pool with WOD-E2E frames around the rated ones (weight 0.4) | the same | the supervised run | 2,500 x 128 |
| 3 RL | NAVSIM, PAI-AV, WOD-E2E rows | exact expected reward over all 5,273 candidates + 0.3 KL to the supervised policy | last phase, head refit to 5,273 | 2,000 x 64 |

Each stage's adapter is merged before the next stage (stages 1 and 2); stage 3 trains one adapter and the Policy Head.

## Repository layout

```
configs/      one config per training run (the values the paper's runs read) and the DeepSpeed ZeRO-2 file
scripts/      train_stage1.sh, train_stage2.sh, train_stage3.sh (steps: sft, last, reference, rl)
superglance/
  model/      prompt.py (option prompt, letter readout), history.py (past frames), merge.py (LoRA merge, CPU)
  data/       qa.py + qa_pairs.py (stages 1-2 rows and their two option orders), policy.py + policy_pairs.py
              (stage-3 rows: compact pools, past frames, Policy Head readout), mixture.py (source mixture),
              build_qa.py (stage-1/2 trainer directory from the released QA data)
  objectives/ permutation.py: joint Brier policy gradient, soft log score, exact RL loss, KL
  train/      qa.py (stages 1-2), policy.py (stage 3), rl_reference.py (the supervised policy's logits for RL's KL)
  executor/   pdm.py (NAVSIM PDM simulator and batched PDMS), sidecar.py + stitched.py (the AlpaSim executor)
tests/        test_objective.py: CPU self-check of the objectives
```

## Installation

Training and inference (8 GPUs):

```bash
pip install -r requirements.txt
```

The pinned versions are those of the paper's runs: torch 2.10, transformers 5.17, peft 0.21, DeepSpeed 0.18.1 and
flash-linear-attention 0.5.2 for Qwen3.5's linear-attention layers. Train with Triton 3.7.1: on Hopper GPUs
flash-linear-attention 0.5.2 refuses its gated backward kernel under Triton 3.4.0 to 3.7.0, which give wrong results.

The executor (`superglance/executor/`) and the NAVSIM rewards run in a separate environment with the NAVSIM devkit v1.1
and the nuPlan devkit, installed as NAVSIM describes. If NAVSIM's comfort scores come out all zero, set
`OPENBLAS_CORETYPE=Haswell` (seen with numpy 1.23.4 / OpenBLAS 0.3.20 on Intel Sapphire Rapids CPUs).

Run every command from the repository root (`python -m superglance....`).

## Data

### Stages 1 and 2: the QA dataset

The questions are released as [SuperGlance-QA](https://huggingface.co/HF_ORG/SuperGlance-QA):

```
stage1/train/<source>.jsonl    one file per source (counts.json lists every source, with a reason if one is held back)
stage2/train/<source>.jsonl
counts.json
images/text/blank.png          the blank image of the text-only questions
build/build_images.py          writes every other image from its source dataset's public release (build/README.md)
```

Each line is one question: `question` (without options), `answers` (the options in the order the model sees them,
shuffled once with a fixed seed), `gold_index`, `source`, `sampling_task` and `sampling_bucket` (the mixture's groups),
`images` (one path relative to the data directory), and `row` (its line in the paper's training file). Yes/no rows
also carry `yes_index`; the remaining fields (ids, the source's own options and answer, image references for the image
build) are kept and ignored by the trainer. Build a trainer directory per stage:

```bash
python -m superglance.data.build_qa --hf <dataset snapshot> --stage 1 --out data/stage1   # train.jsonl, dev.jsonl
python <dataset snapshot>/build/build_images.py data/stage1                              # data/stage1/images/
```

`train.jsonl` is the paper's training file row for row. The development split is not released: `build_qa.py` writes a
fixed sample of training rows as `dev.jsonl`, which the trainer only evaluates (the paper's runs used their last step).
Stage 2's mixture includes stage-1 questions, which its source files repeat.

### Stage 3: policy pools

Stage 3 trains on pools built from NAVSIM (navtrain), PAI-AV and WOD-E2E, which their licenses do not let us
redistribute, plus QA replay rows. A pool directory holds `train.jsonl` and `dev.jsonl` and, per manifest stem,
memory-mapped arrays with one row per policy row:

| file | shape | content |
|---|---|---|
| `<stem>_targets.npy` | N x K, float16 | the supervised target over the K candidates (normalized when read) |
| `<stem>_rewards.npy` | N x K, float16 | each candidate's reward: PDMS x DDC minus a penalty (NAVSIM), -0.2 x ADE (PAI-AV), RFS / 10 (WOD-E2E) |
| `<stem>_ref.npy` | N x K, float16 | RL pools only: the supervised policy's head logits (`scripts/train_stage3.sh reference`) |

A policy row has `no_options: true`, `fixed_order: true`, `compact: {"k": K, "row": j}` (its row in the arrays), the
question text (ego speed, acceleration, past positions, navigation command, any instruction), `images` (the current
front frame, then the past frames 1.5, 1.0 and 0.5 s earlier, oldest first, with `history: true`), `source` and the
mixture fields. QA replay rows are stage-1/2 rows. K is 5,121 for the supervised pools and 5,273 for RL.

## Weights

[SuperGlance](https://huggingface.co/HF_ORG/SuperGlance) holds the merged stage-2 model (stage 3's base) and the stage-3
adapters with their Policy Heads.

## Training

```bash
BASE=<Qwen3.5-9B> DATA=data/stage1 bash scripts/train_stage1.sh          # -> runs/stage1/merged
DATA=data/stage2 bash scripts/train_stage2.sh                            # -> runs/stage2/merged
POOL_SFT=<pool> POOL_LAST=<pool> POOL_RL=<pool> REFIT=<5,273-row adapter> bash scripts/train_stage3.sh
```

Between the last supervised phase and RL, the Policy Head is refit from 5,121 to 5,273 rows on the supervised model's
features: each new row starts from its nearest old candidate's row with the bias lowered by 4 and is fit by L-BFGS
with the language model frozen (L2 1e-3). `REFIT` is that adapter; the refit code will follow in this repository.
Stage 3's QA rows keep the stage-1/2 objective (`pair_reward` "joint"); the RL pool has policy rows only, so its
config's `pair_reward` is never read.

## Evaluation

The plan is read from the Policy Head as the softmax-weighted mean (temperature 0.2) of the 64 most probable
candidates' poses. In closed loop the executor runs the plan through NAVSIM's PDM simulator from the car's current
state, continuing the previous plan's steering, and returns 40 poses 0.1 s apart (`python -m
superglance.executor.stitched`, JSON lines on stdin and stdout; `--check` tests it).
The evaluation code (NAVSIM navtest, WOD-E2E, PAI-AV, AlpaSim, Language2Drive and the QA benchmarks) and the AlpaSim
driver will be added.

## Self-check

```bash
python tests/test_objective.py   # CPU, about 10 s
```

It checks the stage-1/2 estimator against the exact gradient for 2 to 9 options and every option shift, including the
shifts where reading the shifted view's gold at the wrong position would change the gradient, and the RL loss's
gradient against the expected reward with its KL term.

## Citation

```bibtex
@inproceedings{superglance2027,
  title     = {SuperGlance: Turning a Vision-Language Model into an Encoder-Style Driving Policy},
  author    = {TBD},
  year      = {2027}
}
```

## License

Apache License 2.0 (`LICENSE`). Third-party code and models are listed in `NOTICE`.
