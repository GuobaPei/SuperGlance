"""WOD-E2E scene rows and RFS reads.
inputs: wod_rows.py meta -> the scene rows the feature extraction reads (superglance.data.scene.load_scenes: token, cam_f0,
  ego_velocity, ego_acceleration, driving_command one-hot [left, straight, right, unknown], history_poses at -1.5 / -1.0 / -0.5 / 0 s)
  + a token list.
rfs: official rater_feedback_utils.get_rater_feedback_score. A pick is one of the 4 s candidates (8 poses at 0.5 s); RFS wants 20
  points at 4 Hz over 5 s: linear from the origin through the 8 poses, then 4.25-5 s at the last segment's velocity. References: the
  logged future itself (5 s), the logged future cut to 8 poses and stretched the same way, and the oracle candidate (min ADE to the
  logged future over the 8 poses). Prints the frame mean and the mean over scenario clusters (WOD_CLUSTERS: the WOD-E2E release's
  val_sequence_name_to_scenario_cluster.json).
usage: python -m superglance.builders.pools.wod_rfs inputs <rows dir> <out prefix>
       python -m superglance.builders.pools.wod_rfs rfs <rows dir> <cands.npy> [<name>=<feature dir> ...]"""
import glob, json, os, sys
import numpy as np
from superglance.builders.pools.rater_feedback_utils import get_rater_feedback_score

ONE = {"turn left": [1, 0, 0, 0], "go straight": [0, 1, 0, 0], "turn right": [0, 0, 1, 0], "unknown": [0, 0, 0, 1]}
CLUSTERS = os.environ.get("WOD_CLUSTERS", "")
T8, T20 = np.arange(1, 9) * 0.5, np.arange(1, 21) * 0.25


def rows(d):
    return [json.loads(l) for p in sorted(glob.glob(f"{d}/meta/*.jsonl")) for l in open(p)]


def to20(p8):   # [N, 8, 2] at 0.5-4 s -> [N, 20, 2] at 0.25-5 s
    t = np.concatenate([[0.0], T8]); p = np.concatenate([np.zeros((len(p8), 1, 2)), p8], 1)
    out = np.stack([np.stack([np.interp(T20, t, q[:, k]) for k in (0, 1)], -1) for q in p])
    late = T20 > 4; out[:, late] = p8[:, 7:8] + (p8[:, 7:8] - p8[:, 6:7]) / 0.5 * (T20[late] - 4)[None, :, None]
    return out


def rfs(R, traj):   # traj [N, 20, 2] -> per-frame RFS
    o = get_rater_feedback_score(traj[:, None], np.ones((len(R), 1)), [[np.array(p["traj"]) for p in r["prefs"]] for r in R],
                                 [np.array([p["score"] for p in r["prefs"]]) for r in R], np.array([r["speed"] for r in R]),
                                 frequency=4, length_seconds=5)
    return o["rater_feedback_score"]


def report(name, s, R):
    cl = {k: v["scenario_cluster"] for k, v in json.load(open(CLUSTERS)).items()} if os.path.exists(CLUSTERS) else {}
    by = {}
    for r, x in zip(R, s):
        by.setdefault(cl.get(r["segment"], "?"), []).append(x)
    print(f"{name:28s} RFS {s.mean():.3f} (n {len(s)}); cluster mean {np.mean([np.mean(v) for v in by.values()]):.3f} over {len(by)} clusters")


def main():
    if sys.argv[1] == "inputs":
        d, out = sys.argv[2:4]; R = rows(d)
        with open(f"{out}_inputs.jsonl", "w") as f:
            for r in R:
                p = r["past16"]
                f.write(json.dumps({"token": r["token"], "cam_f0": r["images"][0], "ego_velocity": [r["speed"], 0.0], "ego_acceleration": [r["accel"], 0.0],
                                    "driving_command": ONE[r["command"]], "history_poses": [[*p[i], 0.0] for i in (-7, -5, -3, -1)]}) + "\n")
        open(f"{out}_tokens.txt", "w").write("\n".join(r["token"] for r in R) + "\n"); print(len(R), "rows"); return
    d, cands = sys.argv[2:4]; R = [r for r in rows(d) if r["prefs"]]; C = np.load(cands)[:, :, :2]
    F = np.array([r["future"] for r in R]); F8 = F[:, 1:16:2]
    report("logged future (5 s)", rfs(R, F), R)
    report("logged future, 8 poses -> 5 s", rfs(R, to20(F8)), R)
    ade = np.linalg.norm(C[None] - F8[:, None], axis=-1).mean(-1)
    report("oracle candidate", rfs(R, to20(C[ade.argmin(1)])), R)
    print(f"{'':28s} oracle ADE (4 s, 8 poses) median {np.median(ade.min(1)):.2f} m")
    for arg in sys.argv[4:]:
        name, xd = arg.split("=", 1); pick = {}
        for r in range(64):
            if not os.path.exists(f"{xd}/tok_{r}.txt"):
                continue
            pick.update(zip(open(f"{xd}/tok_{r}.txt").read().split(), np.load(f"{xd}/code_{r}.npy")))
        S = [r for r in R if r["token"] in pick]
        P = C[[int(pick[r["token"]]) for r in S]]
        report(name, rfs(S, to20(P)), S)
        print(f"{'':28s} ADE to the logged future (4 s) median {np.median(np.linalg.norm(P - np.array([r['future'] for r in S])[:, 1:16:2], axis=-1).mean(-1)):.2f} m")


if __name__ == "__main__":
    main()
