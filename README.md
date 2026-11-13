# SuperGlance: Driving at a Glance with a Vision-Language Model

[Paper](TBD) | [QA data](https://huggingface.co/datasets/HF_ORG/SuperGlance-QA) | Weights:
[SuperGlance-9B](https://huggingface.co/HF_ORG/SuperGlance-9B),
[SuperGlance-9B-Stage1](https://huggingface.co/HF_ORG/SuperGlance-9B-Stage1),
[SuperGlance-9B-Stage2](https://huggingface.co/HF_ORG/SuperGlance-9B-Stage2)

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
| 3 last phase | the supervised pool with its WOD-E2E rows replaced by training frames 135-165, around the rated frame (new targets, WOD-E2E weight 0.4) | the same | the supervised run | 2,500 x 128 |
| 3 RL | NAVSIM, PAI-AV, WOD-E2E rows | exact expected reward over all 5,273 candidates + 0.3 KL to the supervised policy | last phase, head refit to 5,273 | 2,000 x 64 |

Stages 1 and 2 are merged into the model before the next stage; stage 3 trains one adapter and the Policy Head. All runs
use 8 GPUs.

## Repository layout

```
configs/      one config per training run (the values the paper's runs read) and the DeepSpeed ZeRO-2 file
scripts/      train_stage{1,2,3}.sh; eval_{navsim,pai,wod,alpasim,l2d,qa}.sh; build_qa.sh, build_pools.sh (training data)
superglance/
  model/      prompt.py (option prompt, letter readout), history.py (past frames), merge.py (LoRA merge, CPU)
  data/       qa.py + qa_pairs.py (stage-1/2 rows, two option orders), policy.py + policy_pairs.py (stage-3 rows, compact
              pools, Policy Head readout), scene.py (scene row -> policy prompt), mixture.py, build_qa.py (released QA data)
  objectives/ permutation.py: joint Brier policy gradient, soft log score, exact RL loss, KL
  train/      qa.py (stages 1-2), policy.py (stage 3), rl_reference.py (the supervised policy's logits for RL's KL)
  refit/      the Policy Head refit 5,121 -> 5,273: rows.py, fit.py (L-BFGS), extend_head.py, kin.py
  eval/       features.py (hidden states at the answer position), readout.py (top-64 mean), heads.py, and per benchmark:
              pai_*.py (PAI-AV), wod_*.py (WOD-E2E), qa_*.py (QA benchmarks)
  executor/   pdm.py (NAVSIM PDM simulator, batched PDMS), sidecar.py + stitched.py (the closed-loop executor)
  alpasim/    driver.py (AlpaSim driver service), clean.py, table.py, km.py, pair.py (916-scenario summaries)
  builders/   qa/ (stage-1/2 question pools), pools/ (stage-3 pools, rewards, history frames)
tests/        test_objective.py: CPU self-check of the objectives
```

## Installation

Training and inference (8 GPUs):

```bash
pip install -r requirements.txt
pip install --no-deps triton==3.7.1
```

The pinned versions are those of the paper's runs: torch 2.10, transformers 5.17, peft 0.21, DeepSpeed 0.18.1 and
flash-linear-attention 0.5.2 for Qwen3.5's linear-attention layers, with torchvision 0.25 (the release that pairs with
torch 2.10). Triton 3.7.1 is installed in a second step without the dependency check, because torch's wheel pins another
Triton; this is how the paper's environment was built. On Hopper GPUs flash-linear-attention 0.5.2 refuses its gated
backward kernel under Triton 3.4.0 to 3.7.0, which give wrong results. The stage-1 and stage-2 scripts set
`FLA_TILELANG=0` (flash-linear-attention's Triton kernels), as the paper's runs did; it has no effect without TileLang.
The paper's supervised stage-3 runs also loaded Liger kernels and causal-conv1d through a path overlay; the release uses
the stock kernels, and `liger-kernel` and `causal-conv1d` are an optional speed-up.

The executor, the NAVSIM rewards and the NAVSIM scene export run in a separate environment with the NAVSIM devkit v1.1
and the nuPlan devkit, installed as NAVSIM describes. If NAVSIM's comfort scores come out all zero, set
`OPENBLAS_CORETYPE=Haswell` (seen with numpy 1.23.4 / OpenBLAS 0.3.20 on Intel Sapphire Rapids CPUs).

Run every command from the repository root (`python -m superglance....`).

## Data

### Stages 1 and 2: the QA dataset

The questions are released as [SuperGlance-QA](https://huggingface.co/datasets/HF_ORG/SuperGlance-QA):

```
stage1/train/<source>.jsonl    one file per source with its released rows
stage2/train/<source>.jsonl
counts.json                    per source: rows (the paper's), uploaded (released), held (why rows are not, or null)
images/text/blank.png          the blank image of the text-only questions
build/build_images.py          writes every other image from its source dataset's public release (build/README.md)
```

Each line is one question: `question` (without options), `answers` (the options in the order the model sees them,
shuffled once with a fixed seed), `gold_index`, `source`, `sampling_task` and `sampling_bucket` (the mixture's groups),
`images` (one path relative to the data directory), and `row` (its line in the paper's training file). Yes/no rows also
carry `yes_index`; the remaining fields (ids, the source's own options and answer, image references for the image build)
are kept and ignored by the trainer.

Four sources are not in the dataset: we used them as published, with their own options, so they come from their original
releases:
[SAT](https://huggingface.co/datasets/array/SAT-v2) (`SAT`),
[Social IQa](https://huggingface.co/datasets/allenai/social_i_qa) (`text_siqa`),
[PIQA](https://huggingface.co/datasets/ybisk/piqa) (`text_piqa`) and AI2D from
[The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) (`ai2d`).
GQA, ChartQA and the yes/no rows built from Social IQa and PIQA questions are in the dataset, since we processed them.
The four sources are rebuilt from those releases with the stage-1 builders (Building the training data, below), not
dropped: option shifts and mixture draws are keyed by line number, so a missing source, or a zero weight in its place,
changes every other row's options and draws. The trainers stop when a source of the mixture has no rows, and `build_qa`
writes nothing while a source is incomplete (`--allow-incomplete` writes the released rows alone, for a smoke test):

```bash
python -m superglance.data.build_qa --hf <dataset snapshot> --stage 1 --out data/stage1   # train.jsonl
python <dataset snapshot>/build/build_images.py data/stage1                              # data/stage1/images/
```

With every source complete, `train.jsonl` is the paper's training file row for row. The development split is not
released: without `dev.jsonl` the trainer runs no evaluation and keeps the last step, the checkpoint both of the paper's QA
runs selected on their development split. Stage 2's mixture includes stage-1 questions, which its source files repeat.
The DriveLM test questions of our scene split are not released either.

### Stage 3: policy pools

Stage 3 trains on pools built from NAVSIM (navtrain), PAI-AV and WOD-E2E, which their licenses do not let us
redistribute, plus QA replay rows (the stage-2 questions, held sources included). `scripts/build_pools.sh` rebuilds them
(Building the training data). A pool directory holds `train.jsonl` and `dev.jsonl` and, per manifest stem, memory-mapped
arrays with one row per policy row:

| file | shape | content |
|---|---|---|
| `<stem>_targets.npy` | N x K, float32 (float16 also read) | the supervised target over the K candidates (normalized when read) |
| `<stem>_rewards.npy` | N x K, float32 | each candidate's reward: PDMS x DDC minus 10 when NC, DAC or TTC is below 1 (NAVSIM), -0.2 x ADE (PAI-AV), RFS / 10 (WOD-E2E) |
| `<stem>_ref.npy` | N x K, float16 | RL pools only: the supervised policy's head logits (`scripts/train_stage3.sh reference`) |

A policy row has:
- `no_options: true`, `fixed_order: true`, `compact: {"k": K, "row": j}` (its row in the arrays), and `gold_index` with
  `correct_answer` (`t%04d`): the argmax of the target;
- `question`: the scene prompt of `superglance.data.scene` (the sentence "the image is its front camera" exactly once, ego
  speed and acceleration, past positions, navigation command, then any instruction);
- `images`: the current front frame, then with `history: true` the frames 1.5, 1.0 and 0.5 s earlier, oldest first,
  each stored as `superglance.model.history` makes it (at most 174,080 pixels, JPEG quality 92);
- `source`, `sampling_task`, `sampling_bucket`. The source names the configs mix: `navtrain`, `pai_human`, `pai_dagger`,
  `wod` and the instruction rows `if_navtrain_{follow,junction,lane,speed,stopgo}`, `if_pai_{follow,speed,stopgo}`,
  `if_wod_pace` (supervised pools, plus the stage-2 QA sources); `navtrain`, `pai_human`, `wod` (RL pool).

K is 5,121 for the supervised pools and 5,273 for RL.

## Weights

| repository | model |
|---|---|
| [SuperGlance-9B](https://huggingface.co/HF_ORG/SuperGlance-9B) | the final model (stage-3 RL, with its 5,273-row Policy Head) |
| [SuperGlance-9B-Stage1](https://huggingface.co/HF_ORG/SuperGlance-9B-Stage1) | the stage-1 model (general QA) |
| [SuperGlance-9B-Stage2](https://huggingface.co/HF_ORG/SuperGlance-9B-Stage2) | the stage-2 model (driving QA), stage 3's base |

## Training

```bash
BASE=<Qwen3.5-9B, local snapshot> DATA=data/stage1 bash scripts/train_stage1.sh      # -> runs/stage1/merged
DATA=data/stage2 bash scripts/train_stage2.sh                                        # -> runs/stage2/merged
POOL_SFT=<pool> POOL_LAST=<pool> POOL_RL=<pool> CANDS=<5,121 candidates> REFIT_ROWS=<rows> REFIT_DATA=<refit data> \
  HELDOUT=<navtrain held-out tokens> bash scripts/train_stage3.sh                    # sft, last, refit, reference, rl
```

- `BASE` is a local directory with the full snapshot: the merge reads its `model.safetensors.index.json`.
- Stage 1 needs gradient checkpointing on 80 GB GPUs, and its config has it on. The paper's run had it off for steps
  1-4,000, stopped out of memory near step 4,056 and continued from checkpoint-4000 with it on: the same updates, more
  compute. A stopped run continues with `bash scripts/train_stage1.sh --resume-from-checkpoint`.
- Stage 3 runs five steps; pass step names to run some of them. Stage 3's QA rows keep the stage-1/2 objective
  (`pair_reward` "joint").
- RL's objective is `soft_objective` "exact_rl": the exact expected reward over every candidate, advantage
  r - E_p[r] with no draws and no normalization, plus the KL term. Its earlier name "grpo" is still accepted. The RL
  config's `reward_samples` and `pair_reward` are read but unused: the loss uses no draws and the RL pool has no QA rows.
  The config key `policy_head` (earlier name `choice256`) adds the Policy Head of `choice_k` rows.

### Head refit

Between the last supervised phase and RL, the Policy Head is widened from 5,121 to 5,273 rows on the last phase's
features (`refit` step): `superglance.eval.features` on five sets (navtrain fit and held-out scenes, a PAI-AV subset, navtest,
PAI-AV val and test), an L-BFGS fit of the 152 new rows with the language model and the old rows frozen
(`superglance.refit.fit`: each new row starts from its nearest old candidate's row with the bias lowered by 4; L2 weights
1e-1, 1e-2 and 1e-3, chosen on the navtrain held-out PDMS and the PAI-AV check ADE), the widened adapter
(`superglance.refit.extend_head`), and a stop rule: navtest PDMS at most 0.1 below the 5,121-row head's and PAI-AV val
ADE@3 s at most 0.005 m above it. `REFIT_ROWS` comes from `superglance.refit.rows`; `REFIT_DATA` holds the 5,273
candidates, the navtrain targets over them, the PAI-AV subset and the candidate score tables that `superglance/refit/fit.py`
names. `REFIT` defaults to the step's output; set it to start `reference` and `rl` from another 5,273-row adapter.

## Evaluation

The plan is read from the Policy Head as the softmax-weighted mean (temperature 0.2) of the 64 most probable candidates'
poses (`superglance.eval.readout`). In closed loop the executor runs the plan through NAVSIM's PDM simulator from the car's
current state, starting from the steering angle reached under the previous plan unless that angle is more than 0.2 rad
from the yaw-rate estimate (then from the estimate), and returns 40 poses 0.1 s apart (`python -m
superglance.executor.stitched`, JSON lines on stdin and stdout; `--check` tests it).

| paper | script |
|---|---|
| Table 2 (QA benchmarks) | `scripts/eval_qa.sh` |
| Table 4 (NAVSIM navtest) | `scripts/eval_navsim.sh` |
| Table 3, Fig. 6 (AlpaSim closed loop) | `scripts/eval_alpasim.sh` |
| Table 5 (WOD-E2E test) | `scripts/eval_wod.sh` (writes the submission; the server scores it) |
| Table 6 (PAI-AV open loop) | `scripts/eval_pai.sh` |
| Table 7 (Language2Drive) | `scripts/eval_l2d.sh` |

### NAVSIM navtest

```bash
BASE=<stage-2 model> ADAPTER=<adapter> CANDS=<candidates> NAVTEST=<navtest scene export> HIST=<history frames> \
  NAVSIM_PY=<python of the NAVSIM environment> bash scripts/eval_navsim.sh
```

`NAVTEST` is `superglance.builders.pools.navsim_export`'s navtest rows (with each scene's metric cache) and `HIST`
`superglance.builders.pools.history_nav`'s output. The script extracts the features with the past frames (a scene without
all three reads its current frame alone), reads the plan and scores it with the NAVSIM v1.1 PDM scorer; the `mean` line
is the paper's readout, printed beside argmax and medoid.

### PAI-AV open loop

```bash
PAI_ROOT=<PhysicalAI-AV root> GOLD=<1005_7cam_gold_eval_metadb_public.parquet> BASE=<stage-2 model> ADAPTER=<adapter> \
  CANDS=<candidates> bash scripts/eval_pai.sh
```

Our version of a 700-frame leakage-free subset: the 140 test clips of the public gold eval list of the alpamayo recipes,
five frames per clip at 5.1 s + (-2 ... 2) s. `PAI_ROOT` holds `clip_index.parquet`, `camera/camera_front_wide_120fov/`,
`labels/egomotion*/` and `calibration/` as the dataset ships them. Steps: rows (`superglance.eval.pai_rows`: the front wide
frame re-projected to nuPlan's CAM_F0, ego history, velocity, command, the logged future at 10 Hz), past frames
(`pai_history`; `HIST=0` for a model without them), features, ADE (`pai_ade`). `runs/pai/ade_<NAME>_test.txt` has the argmax
(`top1`) and the six most probable candidates (`avg6`, `min6`) at 3 s / 5 s over the 700 frames and the 140 T0 frames, beside
the oracle candidate and constant velocity. Table 6's best-of-six cells are `min6`. Its one-plan cells read the top-64 mean
densified by a cubic spline from the logged velocity, a step not in this release. `SPLIT=val GOLD=-` builds the validation
rows.

### WOD-E2E test

```bash
WOD_TEST=<test TFRecord shards> WOD_LIST=<test_sequence_frames_for_submission_as_list.txt> BASE=<stage-2 model> \
  ADAPTER=<adapter> CANDS=<candidates> bash scripts/eval_wod.sh
```

The WOD-E2E protos come from the `waymo-open-dataset` package, or set `WOD_PROTOS` to the Python modules generated from its
`.proto` files. The script builds the rows of the listed frames (front camera, re-projected to CAM_F0; the test split has
no past frames), extracts the features, reads `mean02` (the paper's readout) and `mbr02`, and writes and checks one
submission package per readout (`runs/wod/sub/`). It uploads nothing: it prints the `wod_submit fill` command for the
account, method and author fields, and the filled `mean02` package is uploaded on the challenge page. Table 5's row is the
server's RFS and ADE of that package.

### AlpaSim closed loop

```bash
BASE=<stage-2 model> ADAPTER=<RL adapter> CANDS=<5,273 candidates> ALPASIM=<AlpaSim checkout> DATA=<AlpaSim data folder> \
  NAVSIM_PYTHON=<python of the NAVSIM environment> ALPASIM_RUN=<command> bash scripts/eval_alpasim.sh
```

Install AlpaSim with the PAI-AV-NuRec 26.01 scenes (suite `public_2601`, 916 scenarios), its data folder and its
`alpasim_grpc` package, following AlpaSim's instructions. AlpaSim calls the driver every 100 ms of simulated time and the
driver plans at every call: the front wide camera re-projected into CAM_F0, the frames 1.5 / 1.0 / 0.5 s back, the training
prompt, the top-64 mean, executed by the PDM simulator (the executor above, a sidecar in the NAVSIM environment); AlpaSim's
controller tracks the 40 executed poses. Four drivers run on `DRIVER_GPUS` (default 0-3), AlpaSim's renderer and physics on
`SIM_GPUS` (4-7). The script calls AlpaSim's wizard with the paper's settings and `wizard.run_method=NONE`; `ALPASIM_RUN`
runs the generated config dir (it gets the run dir as its argument and `GPUS`, `NRE_D`, `A` in the environment). It prints
the cleaned summary, the table row (collisions all / at fault, off-road, progress, km per incident, mean score, pass) and km
per at-fault incident with its 95 % interval; `REF=<another run's results-summary.clean.json>` adds the paired differences.
Checks without a GPU: `python -m superglance.alpasim.driver --selftest`.

| paper | run |
|---|---|
| Table 3, no delay: RL / supervised model | `DELAY_US=0`, with the RL adapter and 5,273 candidates / the last supervised phase and 5,121 |
| Table 3, delay = our GPU time | `DELAY_US=84000` (the simulator rounds it up to 100 ms) |
| Fig. 6, lag ratios | `DELAY_US` = 0, 84000, 300000, 518000, 1200000, 2000000, 5000000; ratios with `REF` = the no-delay run |
| supplement, without the PDM simulator / without past frames | `EXEC=plan` / `HIST=0` |

The 916 scenarios run as one run; the paper's runs split them into four quarters and merged the summaries, which gives the
same metrics per scenario with a fixed configuration.

### Language2Drive (NAVSIM track)

```bash
L2D_SCORER=<the benchmark's scoring folder> bash scripts/eval_l2d.sh final=<package dir> [noinst=<package dir> ...]
```

A package holds one model's two runs on the 38 PCUs in four wordings, as the Language2Drive SDK writes them. The script
scores each run with the benchmark's scorer and prints SemSISR, SemCFPS, SCS and PCS, the equal-weight mean over the wordings
(`runs/l2d/`). Table 7: "SuperGlance" is the final model's `_mean`, "no instruction" its `_mean_noinstr`. Running the SDK on
our model, the step that writes a package, is not part of this release.

### Question answering

```bash
MODEL=<Qwen3.5-9B> OUT=runs/qa/base ROOT=<data root of scripts/build_qa.sh> bash scripts/eval_qa.sh
MODEL=<Qwen3.5-9B> ADAPTER=<stage-1 adapter> OUT=runs/qa/stage1 ROOT=<...> REF=runs/qa/base bash scripts/eval_qa.sh
MODEL=<stage-1 model> ADAPTER=<stage-2 adapter> OUT=runs/qa/stage2 ROOT=<...> bash scripts/eval_qa.sh
MODEL=<stage-2 model> ADAPTER=<final adapter> OUT=runs/qa/stage3 ROOT=<...> bash scripts/eval_qa.sh
```

`ROOT` needs the `evals`, `f` and `s2` steps of `scripts/build_qa.sh` (CV-Bench, BLINK and 3DSRBench manifests, the text
benchmarks, the DriveLM test questions of our scene split). The readout is the option letters' logits after the
non-thinking chat prefix, with no generation. `python -m superglance.eval.qa_table <OUT> ...` prints Table 2's rows for the
given runs.

## Building the training data

`superglance/builders/` holds the code that built the training data: `qa/` the stage-1 and stage-2 question pools, `pools/`
the stage-3 pools. `scripts/build_qa.sh` and `scripts/build_pools.sh` run the builders in the order the paper's data was
built (pass step names to run part of it); their headers list every input. Hugging Face downloads use `HF_ENDPOINT` when it
is set.

```bash
ROOT=<data dir> TALLYQA_URL=<tallyqa.zip of the TallyQA release> DRIVELM_QA=<v1_1_train_nus.json> \
  DRIVELM_IMAGES=<DriveLM-nuScenes train images> bash scripts/build_qa.sh
OUT=<dir> CANDS=<5,121 candidates> CANDS_KP=<5,273 candidates> <inputs> bash scripts/build_pools.sh
```

| `build_qa.sh` step | output | sources |
|---|---|---|
| `sat` | SAT rows: 50,000 train, 2,000 dev | SAT-v2 |
| `evals` | CV-Bench, BLINK and 3DSRBench manifests (Table 2), also the reference for de-duplication | CV-Bench, BLINK, 3DSRBench |
| `supplement`, `sources` | GQA, Visual Genome, TallyQA; SAT train and The Cauldron's AI2D, TallyQA, ChartQA, TextVQA | |
| `perspective`, `adapt` | SAT perspective rows; every source as choice rows, with splits and one image store | the above |
| `flips` | relation-flip minimal pairs | GQA |
| `e` | box-choice and spatial-relation rows, perceptual-hash de-duplication against the evaluation images | OpenImages V5, SpatialSense |
| `f` | 13 text-only sources on a blank image, n-gram de-duplication | named in `prepare_text_f.py` |
| `g` | the stage-1 pool: "directly above" pairs and rows with a "none of the other options" choice | OpenImages |
| `s2` | the stage-2 pool: the stage-1 questions plus DriveLM box-choice, relation and question rows | DriveLM-nuScenes v1.1 |

`build_pools.sh` steps: `export` (NAVSIM v1.1 scene export and metric caches, NAVSIM environment), `score` (every candidate on
every navtrain scene with the PDM scorer), `wod` (WOD-E2E rows), `qa` (QA replay), `dev`, `merge`, `clean` (drops every
training row of an evaluation set: PAI-AV evaluation clips and test split, the AlpaSim 916 scenes, navtest tokens and logs,
WOD-E2E val and test), `history` (front-camera frames 1.5, 1.0 and 0.5 s back), `stage` (the supervised pool), `last` (the
last-phase pool) and `rl` (the 5,273 candidates' reward columns of the RL pool). Targets: NAVSIM scenes softmax((PDMS x DDC -
0.1 ADE) / 0.1); other NAVSIM frames and WOD-E2E softmax(-ADE / 1 m); PAI-AV softmax(-ADE / 0.5 m); last phase, WOD-E2E rows
softmax(R / 0.5 - (ADE - min ADE) / 1 m), R the RFS with the logged future as the only rated trajectory. Some inputs of
`merge` (the navtrain, OpenScene, PAI-AV and instruction shards) come from builders that are not part of this release; the
header of `build_pools.sh` names them.

## Self-checks

```bash
python tests/test_objective.py                   # CPU, about 10 s; needs torch and Pillow
python -m superglance.alpasim.driver --selftest  # geometry, camera re-projection, past frames
python -m superglance.executor.stitched --check  # NAVSIM environment
```

The first checks the stage-1/2 estimator against the exact gradient for 2 to 9 options and every option shift, including
the shifts where reading the shifted view's gold at the wrong position would change the gradient, and the RL loss's
gradient against the expected reward with its KL term.

## Citation

```bibtex
@inproceedings{superglance2027,
  title     = {SuperGlance: Driving at a Glance with a Vision-Language Model},
  author    = {TBD},
  year      = {2027}
}
```

## License

Apache License 2.0 (`LICENSE`). Third-party code and models are listed in `NOTICE`.
