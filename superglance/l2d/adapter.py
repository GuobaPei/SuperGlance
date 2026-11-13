"""Language2Drive SDK adapter for SuperGlance: the stage-2 merged model with a stage-3 adapter (LoRA + Policy Head over a fixed
library of K candidates, 8 poses at 2 Hz, rear-axle ego frame). The SDK loads this file and calls create_adapter(config).

Decoding is one forward pass of the Policy Head, read out as config "readout": "mean64" (default) = the temperature-0.2
softmax-weighted mean of the 8 poses of the head's 64 most probable candidates (the paper's readout, as the AlpaSim driver),
computed from log p, which equals the head's logits up to a constant; "argmax" = the most probable candidate. No scorer, rule or
re-ranking is applied. With config "trace", each request appends one JSON line: the prompt, the full distribution (float32 and
float16), the argmax, the readout and the output poses, so the argmax readout of the same run can be computed afterwards
(superglance.l2d.argmax).

The prompt is the training rows' NAVSIM prompt, superglance.data.scene.scene_head(scene) + C256_INSTRUCTION, built from the
public observation of the request. How the instruction enters it (config prompt_mode):
  append           (default) C256_INSTRUCTION + " " + instruction_prefix + instruction      [one appended sentence]
  replace_command  the navigation-command phrase carries the instruction text instead of the fixed command
  none             no instruction (control): exactly the trained prompt, so both instructions of a scene get the same plan
L2D supplies no route command, so the non-language command is fixed to config.command ("go straight") for every request,
as L2D does for Qwen-Drive and ReCogDrive.

Observation -> prompt fields (per track):
  navsim    CAM_F0 current frame; speed = |velocity_xy| at the anchor; longitudinal acceleration = (v_x[-1] - v_x[-2]) /
            0.5 s (L2D model_eval/common.acceleration_history); past positions = ego_history x, y at -1.5/-1.0/-0.5 s.
  nuscenes  CAM_FRONT current frame; global ego poses -> anchor frame; history positions interpolated at -1.5/-1.0/-0.5 s,
            speed / acceleration by finite differences of those poses (untested: no nuScenes v1.0-test sensors here).
  wod       front camera current frame; past_states (16 at 4 Hz) interpolated at -1.5/-1.0/-0.5 s, speed from vx, vy,
            acceleration from ax (untested: no WOD-E2E data here).
Trajectory -> output grid: prepend the origin at t = 0, linearly interpolate the 8 poses (0.5 ... 4.0 s); past 4.0 s
extend at the constant velocity of the last 0.5 s segment. The pose reference point (rear axle) is passed through unchanged
for rear_axle and ego_pose_origin (nuScenes ego pose = rear-axle midpoint); rig_origin is not converted (closed loop,
out of scope here).

history=true (the paper's models): navsim only. The request's CAM_F0 list is the public input's camera_paths_history, 4
frames oldest first (t - 1.5 / 1.0 / 0.5 s, then now); the 3 past frames go through superglance.model.history exactly as the
training rows: hist_jpeg (shrink to <= 174,080 px, JPEG q92) -> open_past, then the "frame: k" layout with the current frame
last (frame 3), and the question's camera sentence ONE -> MANY.
"""
import base64
import hashlib
import io
import json
import math
import threading
import time

import numpy as np

DEFAULTS = dict(
    model_code=None,   # unused: the model code is this package
    model=None,
    adapter=None, cands=None,
    choice_k=None, min_pixels=65536, max_pixels=589824,
    prompt_mode="append", instruction_prefix="Driving instruction: ", command="go straight",
    trace=None, synthetic_ok=True, history=False, readout="mean64",
)
PROMPT_MODES = ("append", "replace_command", "none")
READOUTS = ("mean64", "argmax")
_STATE = {}
_LOCK = threading.Lock()


def _cfg(config):
    cfg = dict(DEFAULTS)
    cfg.update(config or {})
    if cfg["prompt_mode"] not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}")
    if cfg["readout"] not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}")
    return cfg


def read_mean(C, L):   # the tau 0.2 softmax mean of the head's top 64
    top = np.argsort(-L)[:64]; q = np.exp((L[top] - L[top[0]]) / 0.2); q /= q.sum()
    return (q[:, None, None] * C[top]).sum(0)


def _model(cfg):
    """Load the frozen weights once per process (shared by every session); same calls as stage3_navsim.cmd_infer."""
    key = (cfg["model_code"], cfg["model"], cfg["adapter"], cfg["cands"], cfg["choice_k"], cfg["max_pixels"], cfg["min_pixels"])
    with _LOCK:
        if _STATE.get("key") == key:
            return _STATE
        if not (cfg["adapter"] and cfg["cands"]):
            raise ValueError("config needs 'adapter' (LoRA dir) and 'cands' (K x 8 x 3 library .npy)")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from superglance.data import scene as s3
        from superglance.data.policy import DecisionCollator, candidate_token_ids, readout
        from superglance.model.prompt import render_chat
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(cfg["model"], min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])
        model = AutoModelForImageTextToText.from_pretrained(cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa")
        cands = np.load(cfg["cands"])
        k = cfg["choice_k"] or len(cands)   # K from the library; a configured choice_k must agree
        assert cands.shape == (k, 8, 3), (k, cands.shape)
        model.choice_head = torch.nn.Linear(model.config.text_config.hidden_size, k, dtype=model.dtype)
        model = PeftModel.from_pretrained(model, cfg["adapter"]).cuda().eval()
        DecisionCollator(processor)  # sets the tokenizer's left padding exactly as in training / cmd_infer
        _STATE.clear()
        _STATE.update(key=key, torch=torch, s3=s3, readout=readout, render_chat=render_chat, processor=processor,
                      model=model, cands=cands, token_ids=candidate_token_ids(processor, 16), cache={},
                      load_s=round(time.time() - t0, 1),
                      mf_hist=__import__("superglance.model.history", fromlist=["OFFS"]) if cfg["history"] else None)
        print(f"[choice_adapter] model loaded in {_STATE['load_s']} s", flush=True)
        return _STATE


def _decode(camera):
    from PIL import Image
    if camera.get("encoding") == "png_base64":
        return Image.open(io.BytesIO(base64.b64decode(camera["data"]))).convert("RGB")
    return Image.open(camera["path"]).convert("RGB")


def _interp_xy(times, xy, at):
    return [[float(np.interp(t, times, xy[:, 0])), float(np.interp(t, times, xy[:, 1]))] for t in at]


def _rotate_into(anchor_xy, anchor_yaw, points):
    c, s = math.cos(anchor_yaw), math.sin(anchor_yaw)
    d = np.asarray(points, dtype=np.float64) - np.asarray(anchor_xy, dtype=np.float64)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)


def parse_observation(track, obs, history=False):
    """(front camera payload, scene dict for stage3_navsim.question) from the public observation only; with history, also the
    3 past CAM_F0 payloads (navsim only)."""
    cams = obs.get("cameras") or {}
    if history:
        if track != "navsim" or len(cams["CAM_F0"]) != 4:
            raise ValueError("history mode needs a navsim request with 4 CAM_F0 frames")
        cam, scene = parse_observation(track, obs)
        return cam, scene, [c["image"] for c in cams["CAM_F0"][:3]]
    if track == "navsim":
        hist = obs["ego_history"]
        vel = np.asarray([h["velocity_xy"] for h in hist], dtype=np.float64)
        acc = (vel[-1] - vel[-2]) / 0.5 if len(vel) > 1 else np.zeros(2)
        past = [[float(h["x"]), float(h["y"]), 0.0] for h in hist]
        return cams["CAM_F0"][-1]["image"], dict(velocity=vel[-1].tolist(), acceleration=acc.tolist(), past=past)
    if track == "nuscenes":
        anchor = obs["ego_pose"]
        poses = [*obs.get("ego_history", []), anchor]
        t = np.asarray([(p["timestamp_us"] - anchor["timestamp_us"]) * 1e-6 for p in poses])
        xy = _rotate_into(anchor["xy_m"], anchor["yaw_rad"], [p["xy_m"] for p in poses])
        at = [-1.5, -1.0, -0.5, 0.0]
        past = [[x, y, 0.0] for x, y in _interp_xy(t, xy, at)]
        p = np.asarray(past)[:, :2]
        v = (p[-1] - p[-2]) / 0.5
        acc = (v - (p[-2] - p[-3]) / 0.5) / 0.5
        return cams["CAM_FRONT"][-1]["image"], dict(velocity=v.tolist(), acceleration=acc.tolist(), past=past)
    if track == "wod":
        ps = obs["past_states"]
        x, y = np.asarray(ps["x"], dtype=np.float64), np.asarray(ps["y"], dtype=np.float64)
        n = len(x)
        last_is_now = math.hypot(x[-1], y[-1]) < 0.05
        t = -0.25 * (np.arange(n)[::-1] + (0 if last_is_now else 1))
        tt, xy = (t, np.stack([x, y], -1)) if last_is_now else (np.r_[t, 0.0], np.vstack([np.stack([x, y], -1), [0.0, 0.0]]))
        past = [[a, b, 0.0] for a, b in _interp_xy(tt, xy, [-1.5, -1.0, -0.5, 0.0])]
        vel = [float(ps["vx"][-1]), float(ps["vy"][-1])]
        acc = [float(ps["ax"][-1]), float(ps["ay"][-1])]
        front = next((k for k in cams if "FRONT" in k.upper() and "LEFT" not in k.upper() and "RIGHT" not in k.upper()), None)
        return cams[front][-1]["image"], dict(velocity=vel, acceleration=acc, past=past)
    raise ValueError(f"track {track} is not supported by this adapter (open-loop tracks only)")


def build_prompt(s3, scene, instruction, cfg):
    scene = dict(scene, command=cfg["command"])
    mode = cfg["prompt_mode"]
    if mode == "replace_command":
        scene["command"] = instruction.strip().rstrip(".").strip()
    question = s3.scene_head(scene) + s3.C256_INSTRUCTION
    if mode == "append":
        question += " " + cfg["instruction_prefix"] + instruction.strip()
    return question


def forward(state, question, image, past=None):
    """Choice-head probabilities for one (question, image), as cmd_infer --choice256 with a candidate file; past = the full-size
    past front frames (oldest first) for the SFT-mf layout."""
    torch = state["torch"]
    h = hashlib.sha256(question.encode() + image.tobytes())
    for im in past or []:
        h.update(im.tobytes())
    key = h.hexdigest()
    if key in state["cache"]:
        return state["cache"][key], key, True
    if past:
        mf = state["mf_hist"]
        assert len(past) == len(mf.OFFS)
        messages = [{"role": "user", "content": mf.content(len(past)) + [{"type": "text", "text": mf.question(question)}]}]
        images = [mf.open_past(mf.hist_jpeg(im)) for im in past] + [image]
    else:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        images = [image]
    text = state["render_chat"](state["processor"], messages)
    inputs = state["processor"](text=[text], images=images, padding=True, return_tensors="pt")
    with torch.inference_mode():
        logits = state["readout"](state["model"], inputs.to("cuda"), state["token_ids"],
                                  torch.ones(1, dtype=torch.bool), len(state["cands"]))
    p = logits.softmax(-1)[0].float().cpu().numpy()
    if len(state["cache"]) > 4096:
        state["cache"].clear()
    state["cache"][key] = p
    return p, key, False


def to_grid(poses, timestamps):
    """8 poses at 0.5 ... 4.0 s -> requested grid: linear interpolation from the origin, constant velocity past 4 s."""
    t = np.r_[0.0, 0.5 * np.arange(1, 9)]
    xy = np.vstack([[0.0, 0.0], np.asarray(poses, dtype=np.float64)[:, :2]])
    v = (xy[-1] - xy[-2]) / 0.5
    out = []
    for s in timestamps:
        if s <= t[-1] + 1e-9:
            out.append([float(np.interp(s, t, xy[:, 0])), float(np.interp(s, t, xy[:, 1]))])
        else:
            out.append([float(xy[-1, 0] + v[0] * (s - t[-1])), float(xy[-1, 1] + v[1] * (s - t[-1]))])
    return out


class SuperGlanceAdapter:
    def __init__(self, config):
        self.cfg = _cfg(config)

    def predict(self, request):
        spec = request.output_spec
        obs = request.observation or {}
        if self.cfg["synthetic_ok"] and set(obs) == {"cameras", "speed_m_s"} and not obs["cameras"]:
            # SDK adapter-check only (its synthetic request has no cameras): no model call; a constant-speed 8-pose
            # trajectory goes through the same grid conversion as real predictions.
            v = float(obs["speed_m_s"])
            poses = [[v * 0.5 * k, 0.0, 0.0] for k in range(1, 9)]
            return {"coordinate_frame": spec.coordinate_frame, "reference_point": spec.reference_point,
                    "timestamps_s": list(spec.timestamps_s), "xy_m": to_grid(poses, spec.timestamps_s)}
        if request.track not in ("navsim", "nuscenes", "wod"):
            raise ValueError("closed-loop tracks are not wired for this model (rig_origin / controller not converted)")
        state = _model(self.cfg)
        camera, scene, *past = parse_observation(request.track, obs, self.cfg["history"])
        image = _decode(camera)
        past = [_decode(c) for c in past[0]] if past else None
        question = build_prompt(state["s3"], scene, request.instruction, self.cfg)
        t0 = time.time()
        p, key, cached = forward(state, question, image, past)
        j = int(p.argmax())
        poses = read_mean(state["cands"], np.log(np.maximum(p, 1e-38))) if self.cfg["readout"] == "mean64" else state["cands"][j]
        xy = to_grid(poses, spec.timestamps_s)
        if self.cfg.get("trace"):
            top = np.argsort(p)[::-1][:16]
            row = dict(t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), track=request.track, seed=request.seed,
                       instruction=request.instruction, prompt_mode=self.cfg["prompt_mode"], question=question,
                       past_frames=len(past or []),
                       input_key=key, cached=cached, infer_s=round(time.time() - t0, 3), argmax=j, p_argmax=float(p[j]),
                       entropy=float(-(p * np.log(np.clip(p, 1e-12, None))).sum()),
                       top16=[[int(c), round(float(p[c]), 6)] for c in top], readout=self.cfg["readout"], poses=np.round(poses, 4).tolist(),
                       probs_f16_b64=base64.b64encode(p.astype(np.float16).tobytes()).decode(),
                       probs_f32_b64=base64.b64encode(p.astype(np.float32).tobytes()).decode())
            with _LOCK, open(self.cfg["trace"], "a") as f:
                f.write(json.dumps(row) + "\n")
        return {"coordinate_frame": spec.coordinate_frame, "reference_point": spec.reference_point,
                "timestamps_s": list(spec.timestamps_s), "xy_m": xy}


def create_adapter(config):
    return SuperGlanceAdapter(config)
