"""Scene rows -> the policy prompt: one choice among K candidate trajectories with no option list in the prompt
(the answers t0000 ... are never shown; the choice head reads the last prompt position).
A scene row (load_scenes) has token, cam_f0 (front image path), ego_velocity, ego_acceleration, driving_command (one-hot:
turn left, go straight, turn right, unknown), history_poses (1.5, 1.0, 0.5 s ago and now, ego frame: x ahead, y left) and,
on NAVSIM, metric_cache (the NAVSIM v1.1 metric cache path the PDM scorer reads)."""
import json

import numpy as np

COMMANDS = ["turn left", "go straight", "turn right", "unknown"]  # NAVSIM driving_command one-hot order


def command_text(command):
    if isinstance(command, (list, tuple, np.ndarray)):
        command = int(np.argmax(command))
    return COMMANDS[int(command)] if isinstance(command, (int, np.integer)) else str(command)


C256_INSTRUCTION = "Choose the trajectory for the next 4 seconds."


def scene_head(scene):
    return question(scene, None, [], 1).split("Step 1 of 1:")[0]


def c256_row(scene, target=None, k=256):
    """One-step choice among k candidate trajectories (k = the candidate set's size): no option list in the prompt,
    so the answer labels are never shown."""
    row = {"question": scene_head(scene) + C256_INSTRUCTION, "answers": [f"t{i:0{3 if k <= 1000 else 4}d}" for i in range(k)],
           "no_options": True, "fixed_order": True, "gold_index": 0, "image_file": scene["image"]}
    if target is not None:
        row["target_probs"] = [float(x) for x in target]
        row["gold_index"] = int(np.argmax(target))
    return row


def question(scene, codebook, prefix, levels=3):
    v = float(np.hypot(*scene["velocity"][:2]))
    a = float(scene["acceleration"][0])
    past = ", ".join(f"({p[0]:.1f}, {p[1]:.1f})" for p in scene["past"][:-1])
    head = (f"You are driving the ego vehicle; the image is its front camera. Speed {v:.1f} m/s, "
            f"longitudinal acceleration {a:+.1f} m/s^2, navigation command: {command_text(scene['command'])}. "
            f"Past positions over the last 1.5 s relative to now (x ahead, y left, metres): {past}. ")
    return head + f"Step 1 of {levels}: choose the coarse plan for the next 4 seconds."


def load_scenes(path):
    scenes = {}
    for line in open(path):
        r = json.loads(line)
        scenes[r["token"]] = {"token": r["token"], "image": r["cam_f0"], "velocity": r["ego_velocity"],
                              "acceleration": r["ego_acceleration"], "command": r["driving_command"],
                              "past": r["history_poses"], "subset": r.get("subset"), "metric_cache": r.get("metric_cache")}
    return scenes
