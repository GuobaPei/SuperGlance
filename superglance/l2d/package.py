"""Pack one adapter's two SDK output folders, copied exactly as the SDK wrote them, with a short README into
RUN_ROOT/package/l2d_navsim_<tag>_<yyyymmdd>/ and its .zip:
  instructed/              RUN_ROOT/append/official (the submission)
  control_no_instruction/  RUN_ROOT/none/official   (control: the instruction is left out of the model's prompt)
  argmax_from_same_run/    the argmax readout of both runs (superglance.l2d.argmax), when present
  README.md
scripts/eval_l2d.sh scores the unpacked folder. The README is built from the run's configs and inference reports; it states no
file digests and no proxy scores. The paper's runs also checked the adapter's prompts against the training rows and its decode
against the training package (checks/prompt_rows.json, checks/equiv16.json); the README reports them when present.
Optional environment: DESC_LONG (one line saying which model this is), BASE_TEXT (the base model's description).
usage: python -m superglance.l2d.package RUN_ROOT TAG"""
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

R, TAG = Path(sys.argv[1]), sys.argv[2]
day = time.strftime("%Y%m%d", time.gmtime())
name = f"l2d_navsim_{TAG}_{day}"
P = R / "package"; D = P / name; Z = P / f"{name}.zip"
assert not D.exists() and not Z.exists(), f"{D} or {Z} exists"
mc = json.loads((R / "config/model_append.json").read_text())
ev = json.loads((R / "config/evaluation_append.json").read_text())["model"]
inf = {m: json.loads((R / m / "official/navsim/inference.json").read_text()) for m in ("append", "none")}
pr = json.loads((R / "checks/prompt_rows.json").read_text())["summary"] if (R / "checks/prompt_rows.json").exists() else None
prompt_line = (f"- Prompt format: equal to our training rows' format on {pr['equal_with_recorded_acc']}/{pr['rows']} sampled instruction rows when the logged\n"
               f"  acceleration is put back ({pr['differ_only_in_printed_acc'] + pr['equal_with_derived_acc']}/{pr['rows']} equal up to the printed acceleration value with the derived one)."
               if pr else "- Prompt format: not checked against training rows for this run.")
eqf = json.loads((R / "checks/equiv16.json").read_text()) if (R / "checks/equiv16.json").exists() else None
eq = eqf["summary"] if eqf else None
ties = [r["ref_gap"] for r in eqf["rows"] if "stored" in r and r["stored"] != r["adapter"]] if eqf else []
K = mc["choice_k"]; hist = mc["history"]; RO = mc["readout"]
ck = ev["checkpoint"].split("internal ", 1)[-1]
desc = os.environ.get("DESC_LONG", "")
base_txt = os.environ.get("BASE_TEXT", "Qwen3.5-9B (our stage-2 merged base)")
dec_line = (f"- Our own decoding reproduced through this adapter: argmax equal on {eq['argmax_eq_ref']}/{eq['n']} navtest scenes "
            f"(max |dp| {eq['max_dp_ref']:.1e}) against the package's batch-1 decode"
            + (f", and {eq['argmax_eq_stored']}/{eq['stored_rows']} against the checkpoint's stored (batch-8) navtest read"
               + (f"; the other {len(ties)} {'is a near-tie' if len(ties) == 1 else 'are near-ties'} (top-1 / top-2 gap {max(ties):.3f})." if ties else ".")
               if eq.get("stored_rows") else ".")
            if eq else "- Decode-path check: not run for this checkpoint.")
if eq and RO == "mean64":
    dec_line += (f"\n- The top-64 mean through this adapter equals a separately written copy of the formula on the same distributions "
                 f"(max |d| {eq['mean_formula_dmax']:.1e} m); against the checkpoint's stored navtest mean read (head features, batch > 1) "
                 f"the largest point distance is {eq['mean_vs_stored_dxy_median']:.3f} m at the median and {eq['mean_vs_stored_dxy_max']:.3f} m "
                 f"at most over the {eq['mean_vs_stored_rows']} scenes.")
dec_text = ("the softmax-weighted mean (temperature 0.2) of the 8 (x, y) points of the choice head's 64 most probable candidates, at\n"
            "  0.5 ... 4.0 s (rear axle, `ego_at_anchor`): one forward pass. No scorer, rules, re-ranking or sampling; the seed is unused." if RO == "mean64" else
            "the argmax of the choice head. The output is that candidate's 8 (x, y) points at 0.5 ... 4.0 s (rear axle,\n"
            "  `ego_at_anchor`), passed through unchanged. No scorer, rules, re-ranking or sampling; the seed is unused.")
readme = f"""# Language2Drive NAVSIM open-loop track: {TAG}

## Contents
- `instructed/`: the submission. The output folder of `language2drive evaluate --tracks navsim` (SDK 0.1.0), as the SDK wrote it:
  `campaign-binding.json`, `summary.json`, `navsim/canonical.json`, `navsim/rewrites.json`, `navsim/inference.json`,
  `navsim/predictions.ledger.jsonl`, `navsim/binding.json`, `navsim/command-*.log`, `navsim/score.json`.
- `control_no_instruction/`: the same model and SDK run, with the instruction left out of the model's prompt, so both instructions
  of a PCU get the same trajectory. A control only, not a submission.
- `argmax_from_same_run/`: supplementary, not a submission. The argmax readout of the same two runs, computed from each request's
  full head distribution in the adapter trace: `instructed/` and `control_no_instruction/`, each with `canonical.json` and `rewrites.json`
  in the SDK's layout.
- `navsim/score.json` in both folders is not a score. The NAVSIM sidecars are private, so the SDK's scorer had none
  (`n_pcu` 0, null metrics).

## Workload
All 38 PCUs of the public NAVSIM release, canonical wording plus the 3 paraphrases, both instructions: 304 requests per run.
Instructed: {inf['append']['scheduled_requests']} scheduled, {inf['append']['failed_requests']} failed. Control: {inf['none']['scheduled_requests']} scheduled, {inf['none']['failed_requests']} failed.

## Model
- {base_txt} with a LoRA and a {K:,}-way choice head over a fixed library of {K:,} candidate
  trajectories (8 points, 0.5-4.0 s).
- Checkpoint: internal name `{ck}`.{(' ' + desc) if desc else ''}
- Training: {ev['training_regime']}.
- Decoding: {dec_text}

## Adapter notes (what the model sees)
- Images: CAM_F0 now{' and at -1.5 / -1.0 / -0.5 s (past frames resized to at most 174,080 px and JPEG-encoded at quality 92, as in our training rows)' if hist else ''}.
  The other 7 cameras and the LiDAR are not used.
- Ego state: speed = |velocity_xy| at the anchor; past positions = `ego_history` x, y at -1.5 / -1.0 / -0.5 s.
- Acceleration: the request carries none. The adapter derives the longitudinal acceleration as (v_x(0) - v_x(-0.5 s)) / 0.5 s from
  `ego_history` (as L2D's `model_eval/common.acceleration_history`). On the 38 anchors this differs from the logged value by a median
  of 0.10 m/s² (max 0.71).
- Navigation command: fixed to "go straight" in every request, because L2D sends no route command (L2D fixes it the same way for
  Qwen-Drive and ReCogDrive).
- Instruction: appended to our trained prompt as one sentence, "Driving instruction: <text>". The control run leaves it out.
{prompt_line}
{dec_line}
"""
D.mkdir(parents=True)
shutil.copytree(R / "append/official", D / "instructed", symlinks=False)
shutil.copytree(R / "none/official", D / "control_no_instruction", symlinks=False)
for m, sub in (("append", "instructed"), ("none", "control_no_instruction")):
    a = R / m / "argmax_from_trace"
    if a.exists():
        shutil.copytree(a, D / "argmax_from_same_run" / sub, symlinks=False)
(D / "README.md").write_text(readme)
with zipfile.ZipFile(Z, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(D.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(P))
with zipfile.ZipFile(Z) as z:
    assert z.testzip() is None
    names = z.namelist()
print(json.dumps(dict(zip=str(Z), files=len(names), bytes=Z.stat().st_size, folder=str(D))))
