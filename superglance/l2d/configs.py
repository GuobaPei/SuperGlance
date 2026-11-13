"""Model and SDK evaluation configs for one adapter's Language2Drive NAVSIM run on all 38 PCUs, canonical + 3 paraphrases
(304 requests per run), through the SDK's own runner (language2drive evaluate --tracks navsim). Both runs load
superglance/l2d/adapter.py:
  append   the instruction appended after C256_INSTRUCTION (" Driving instruction: <text>"), the instruction rows' format
  none     no instruction (control): both instructions of a PCU get the same model input
  check    prompt_mode none without a trace (a config for checks)
K comes from the library's length. The SDK block is the SDK workspace's evaluation.json (L2D_EVAL: its corrected scorer and
sidecar settings), with the NAVSIM wordings written out and public_dir left at the SDK default (the release's 38 PCUs). The model
configs carry "readout" (READOUT, default mean64 = the temperature-0.2 softmax mean of the head's top 64).
Environment: BASE_MODEL (the stage-2 merged model, required), L2D_EVAL (required); optional DESC (a short model label for the name),
TRAINING (training_regime text), HISTORY (1 = front-camera history, default 1), READOUT.
usage: python -m superglance.l2d.configs RUN_ROOT ADAPTER_DIR LIBRARY.npy TAG PORT_APPEND PORT_NONE"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = os.environ["BASE_MODEL"]
R, AD, CA, TAG, P_APPEND, P_NONE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
K = len(np.load(CA, mmap_mode="r"))
assert np.load(CA, mmap_mode="r").shape == (K, 8, 3), CA
HIST = os.environ.get("HISTORY", "1") == "1"
RO = os.environ.get("READOUT", "mean64"); assert RO in ("mean64", "argmax"), RO
lc = json.loads((AD / "adapter_config.json").read_text())
ck = "/".join(AD.parts[-3:])   # the checkpoint: the adapter path's last three parts
desc = os.environ.get("DESC", TAG)
name = (f"{Path(BASE).name} + LoRA {ck} [{desc}] (choice-head {'top-64 mean at tau 0.2' if RO == 'mean64' else 'argmax'}, K={K}"
        + (", front-camera history)" if HIST else ")"))
training = os.environ.get("TRAINING", (
    "fine-tuned on non-test data only: LoRA SFT on navtrain, PAI-AV train, WOD-E2E train and QA replay, including L2D-style "
    "instruction rows generated from those train splits, then choice-head RL on train splits; no Language2Drive test example; "
    "L2D NAVSIM logs/recordings, RealSim scenes and WOD-E2E val/test excluded"))
inputs = ("CAM_F0 current frame" + (" and the 3 past CAM_F0 frames of the request (t - 1.5 / 1.0 / 0.5 s, shrunk to <= 174,080 px, "
                                     "JPEG q92, as the training rows)" if HIST else "")
          + "; speed |velocity_xy| at the anchor; ego_history x, y at -1.5/-1.0/-0.5 s; longitudinal acceleration "
            "(v_x[-1] - v_x[-2]) / 0.5 s from ego_history velocities (the request has no acceleration); navigation command "
            "fixed to 'go straight' (L2D sends none); the other cameras and LiDAR are not used")
C = R / "config"; C.mkdir(parents=True, exist_ok=True)
base_eval = json.loads(Path(os.environ["L2D_EVAL"]).read_text())
for mode, pm, port in (("append", "append", P_APPEND), ("none", "none", P_NONE), ("check", "none", None)):
    model = {"name": name, "prompt_mode": pm, "instruction_prefix": "Driving instruction: ", "command": "go straight",
             "model": BASE, "adapter": str(AD), "cands": str(CA), "choice_k": K, "max_pixels": 589824, "min_pixels": 65536,
             "readout": RO, "trace": str(R / mode / "adapter_trace.jsonl") if port else None, "synthetic_ok": True,
             "history": HIST}
    (C / f"model_{mode}.json").write_text(json.dumps(model, indent=1))
    if port is None:
        continue
    ev = copy.deepcopy(base_eval)
    ev["endpoint"] = f"http://127.0.0.1:{port}"
    ev["tracks"]["navsim"]["conditions"] = ["canonical", "paraphrase_1", "paraphrase_2", "paraphrase_3"]
    ev["tracks"]["navsim"].pop("public_dir", None)   # SDK default: the release's 38 PCUs
    ev["model"] = {"name": name, "seed": 0,
                   "checkpoint": f"LoRA (PEFT, r {lc['r']} / alpha {lc['lora_alpha']}) + {K:,}-way choice head on {Path(BASE).name}; internal {ck}",
                   "checkpoint_revision": f"{ck} adapter_model.safetensors",
                   "candidates_revision": CA.name,
                   "decoding": (f"softmax-weighted mean (temperature 0.2) of the 8 poses of the choice head's 64 most probable candidates "
                                f"among {K:,} fixed candidate trajectories (8 poses, 0.5-4.0 s), one forward pass, "
                                "no scorer/rules/re-ranking/sampling" if RO == "mean64" else
                                f"choice-head argmax over {K:,} fixed candidate trajectories (8 poses, 0.5-4.0 s), "
                                "no scorer/rules/re-ranking/sampling"),
                   "training_regime": training, "prompt_mode": pm, "adapter_file": "superglance/l2d/adapter.py", "inputs": inputs}
    (C / f"evaluation_{mode}.json").write_text(json.dumps(ev, indent=2))
print(json.dumps({"K": K, "name": name, "configs": sorted(p.name for p in C.iterdir())}))
