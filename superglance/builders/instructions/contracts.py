"""Language2Drive-style instruction rows: shared parts (imitation targets only, yes/no feasibility only, no test data, the
L2D adapter's test-time prompt). Contracts are the L2D verifier's own types (the SDK's navsim_offline/verify.py), evaluated for all
5,121 candidates at once on the positions the model outputs, headings from successive positions (the adapter sends xy only).
  feasible   yes/no: trajectory_quality's |a| <= 3.5 m/s^2 and yaw rate <= 0.8 rad/s on the candidate's own poses, plus the
             source's no-collision / no-off-road flags (navtrain PDM NC = DAC = 1; PAI-AV footprint vs obstacle.offline boxes)
  plain      softmax(-ADE / tau) over all candidates (the pool's target)
  hindsight  softmax(-ADE / tau) over {C_h and not C_c} (h = what the human did), renormalized
  contrast   softmax(-ADE_perp / tau) over S_c = {C_c and not C_h and feasible}; ADE_perp = the distance to the human in the
             dimension the instruction leaves free: arc-length profile (lateral families), distance to the human's path (speed
             families)
tau: navtrain 1 m, PAI-AV 0.5 m, WOD-E2E 1 m. A pair is kept when |S_c| >= 3 and the human satisfies h exclusively.
env: L2D_ROOT (the Language2Drive checkout: ws/ with the
OpenloopTrack workspaces), CANDS (the 5,121-candidate library)."""
import base64, difflib, glob, json, math, re, sys, zlib
import numpy as np
import shapely

import os
L2D = os.environ.get("L2D_ROOT", "")
CANDS = os.environ.get("CANDS", "")
K, KP = 5121, 4097   # candidates; only the first KP have navtrain PDM rows
T = np.arange(1, 9) * 0.5
TAU = {"navtrain": 1.0, "pai": 0.5, "wod": 1.0}
FOOT = np.array([[3.4, -1.0], [3.4, 1.0], [-1.4, 1.0], [-1.4, -1.0], [3.4, -1.0]])   # verify.py's box(-1.4, -1.0, 3.4, 1.0)
LATERAL = {"lane", "junction"}   # ADE_perp = arc-length profile; every other family: distance to the human's path
sys.path[:0] = [f"{L2D}/ws/language2drive.OpenloopTrack.navsim/navsim_offline"]


def rem(a):   # math.remainder(a, 2 pi), elementwise
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def feats(xy):   # [..., 8, 2] -> positions, segment speeds (verify.validate_trajectory), derived headings (normalize_prediction)
    d = np.diff(xy, axis=-2, prepend=np.zeros_like(xy[..., :1, :]))
    return {"xy": xy, "s": np.hypot(d[..., 0], d[..., 1]) / 0.5, "h": np.arctan2(d[..., 1], d[..., 0])}


def kin_ok(P):   # verify.trajectory_quality's kinematically_feasible on the candidates' own (x, y, heading) poses
    s = feats(P[..., :2])["s"]
    return (np.abs(np.diff(s, axis=-1)).max(-1) / 0.5 <= 3.5) & (np.abs(rem(np.diff(P[..., 2], axis=-1))).max(-1) / 0.5 <= 0.8)


def win(c):
    return (T >= c["start_s"]) & (T <= min(c["end_s"], 4.0))


def lane_holds(c, X, w):   # verify_lane_region: the footprint at every window pose inside polygon.buffer(0.05), and x(4 s) >= min_progress
    ok = np.zeros(len(X["s"]), bool); lane = shapely.Polygon(c["polygon_xy"])
    if not lane.is_valid or lane.is_empty:
        return ok
    buf = lane.buffer(0.05); shapely.prepare(buf); js = np.flatnonzero(w)
    idx = np.flatnonzero(X["xy"][:, 7, 0] >= c.get("min_progress_m", 0))
    for j in js:   # the rear axle is inside the footprint, so a covered footprint has its rear axle inside: a cheap prefilter
        idx = idx[shapely.contains_xy(buf, X["xy"][idx, j, 0], X["xy"][idx, j, 1])]
    if len(idx):
        xy, h = X["xy"][idx][:, js], X["h"][idx][:, js]; co, si = np.cos(h)[..., None], np.sin(h)[..., None]
        px, py = co * FOOT[:, 0] - si * FOOT[:, 1] + xy[..., :1], si * FOOT[:, 0] + co * FOOT[:, 1] + xy[..., 1:]
        ok[idx[shapely.covers(buf, shapely.polygons(np.stack([px, py], -1).reshape(-1, 5, 2))).reshape(len(idx), len(js)).all(1)]] = True
    return ok


def holds(c, X):   # verify.verify_contract for every row of X (feats of [N, 8, 2]); a list of contracts = all of them
    if isinstance(c, list):
        return np.logical_and.reduce([holds(x, X) for x in c])
    n, kind = len(X["s"]), c["type"]
    if kind == "goal_region":
        p, h = X["xy"][:, 7], X["h"][:, 7]
        return ((np.hypot(p[:, 0] - c["center_xy"][0], p[:, 1] - c["center_xy"][1]) <= c["radius_m"])
                & (np.abs(rem(h - c["heading_rad"])) <= c["heading_tolerance_rad"]) & (p[:, 0] >= c.get("min_progress_m", 0.0)))
    w = win(c)
    if not w.any():
        return np.zeros(n, bool)
    if kind == "lane_region":
        return lane_holds(c, X, w)
    xy, s = X["xy"][:, w], X["s"][:, w]
    if kind == "speed_band":
        return ((s >= c["min_mps"]) & (s <= c["max_mps"])).all(1)
    if kind == "stop_region":
        d = np.hypot(xy[..., 0] - c["center_xy"][0], xy[..., 1] - c["center_xy"][1])
        return ((d <= c["radius_m"]) & (s <= c["max_mps"])).all(1)
    if kind == "object_speed":
        tg = np.asarray(c["target_positions_xy"][:8], np.float64)
        if len(tg) != 8:
            return np.zeros(n, bool)
        dx, dy = tg[w, 0] - xy[..., 0], tg[w, 1] - xy[..., 1]; g = np.hypot(dx, dy)
        return ((s >= c["min_mps"]) & (s <= c["max_mps"]) & (g >= c["min_gap_m"]) & (g <= c["max_gap_m"])
                & (dx >= c["min_longitudinal_gap_m"])).all(1)
    raise ValueError(kind)


def verify_one(xy, c):   # the SDK's own verifier on one output (positions only, as the adapter sends them)
    import verify
    poses = verify.normalize_prediction({"coordinate_frame": "ego_at_anchor", "reference_point": "rear_axle", "timestamps": list(verify.TIMES),
                                         "positions_xy": [[float(a), float(b)] for a, b in xy]})
    return all(verify.verify_contract(poses, x) for x in (c if isinstance(c, list) else [c]))


# contracts ------------------------------------------------------------------------------------------------------------------
def c_band(lo):   # the spec's speed_band: 2 m/s wide, held over 1.5-4.0 s
    return {"type": "speed_band", "min_mps": lo, "max_mps": lo + 2.0, "start_s": 1.5, "end_s": 4.0}


def c_stop(xy=(0.0, 0.0), r=1000.0):   # stopped (<= 0.5 m/s) from 2.5 s on: centre = the human's stop point (2 m) or anywhere
    return {"type": "stop_region", "center_xy": [float(xy[0]), float(xy[1])], "radius_m": r, "max_mps": 0.5, "start_s": 3.0, "end_s": 4.0}


C_GO = [{"type": "speed_band", "min_mps": 2.0, "max_mps": 1000.0, "start_s": 3.5, "end_s": 4.0},   # WOD P0 GO: v_tail >= 2, x_end >= 8
        {"type": "goal_region", "center_xy": [0.0, 0.0], "radius_m": 1e4, "heading_rad": 0.0, "heading_tolerance_rad": math.pi, "min_progress_m": 8.0}]


def c_follow(lo, tg):   # object_speed: the band plus gaps to the lead's centre (rear axle to centre 7-40 m, at least 5 m ahead)
    return {"type": "object_speed", "min_mps": lo, "max_mps": lo + 2.0, "start_s": 1.5, "end_s": 4.0, "min_gap_m": 7.0, "max_gap_m": 40.0,
            "min_longitudinal_gap_m": 5.0, "target_positions_xy": [[round(float(a), 3), round(float(b), 3)] for a, b in tg]}


def band_of(s6):   # lower edge of the 2 m/s band (0.5 m/s grid) holding the human's 1.5-4.0 s speeds, as centred as the grid allows
    lo_hi, lo_lo = math.floor(s6.min() * 2) / 2, math.ceil((s6.max() - 2.0) * 2) / 2
    if lo_lo > lo_hi:
        return None
    return float(min(max(round(((s6.min() + s6.max()) / 2 - 1.0) * 2) / 2, lo_lo), lo_hi))


def long_pairs(F2, FE, X, H, lead=None):   # A3 stop/go, B1 speed band, B3 follow: [(family, h, c, contract_h, contract_c)]
    out, s = [], H["s"][0]
    stop_h = c_stop(F2[7], 2.0)
    if holds(stop_h, H)[0]:
        out.append(("stopgo", "stop", "go", stop_h, C_GO))
    elif holds(C_GO, H)[0]:
        out.append(("stopgo", "go", "stop", C_GO, c_stop()))
    lo = band_of(s[2:])
    if lo is not None and lo >= 1.0:
        for d, lc in (("slower", lo - 3.0), ("faster", lo + 3.0)):
            if lc >= 0:
                out.append(("speed", f"band:{lo:g}", f"band:{lc:g}:{d}", c_band(lo), c_band(lc)))
                if lead is not None:
                    out.append(("follow", f"band:{lo:g}", f"band:{lc:g}:{d}", c_follow(lo, lead), c_follow(lc, lead)))
    return out


def judge(fam, h, c, ch, cc, X, H, FE):   # -> pair dict or None: the human satisfies h exclusively, |S_c| >= 3
    if not (holds(ch, H)[0] and not holds(cc, H)[0]):
        return None
    Ah, Ac = holds(ch, X), holds(cc, X); E, S = Ah & ~Ac, Ac & ~Ah & FE
    if S.sum() < 3 or not E.any():
        return None
    return {"family": fam, "h": h, "c": c, "contract_h": ch, "contract_c": cc, "nE": int(E.sum()), "nS": int(S.sum()), "E": pack(E), "S": pack(S)}


def pack(m):
    return base64.b64encode(np.packbits(m).tobytes()).decode()


def unpack(s):
    return np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8))[:K].astype(bool)


# targets --------------------------------------------------------------------------------------------------------------------
def ade(C2, F2):   # mean pose distance [K]
    return np.linalg.norm(C2 - F2[None], axis=-1).mean(-1)


def ade_lon(C2, F2):   # arc-length profile distance: mean |s_k(t) - s_h(t)|
    return np.abs(np.cumsum(feats(C2)["s"], -1) - np.cumsum(feats(F2[None])["s"], -1)).mean(-1) * 0.5


def human_path(F2):   # origin + the human's poses (repeats dropped), extended 200 m along the last moving direction (+x if it never moved)
    P = np.vstack([[0.0, 0.0], F2]); d = np.diff(P, axis=0); m = np.hypot(d[:, 0], d[:, 1]) > 0.1
    if not m.any():
        return np.array([[0.0, 0.0], [200.0, 0.0]])
    u = d[np.flatnonzero(m)[-1]]; u = u / np.hypot(*u)
    P = P[np.r_[True, m]]
    return np.vstack([P, P[-1] + 200.0 * u])


def path_dist(Q, path):   # distance of points [N, 2] to a polyline
    A, AB = path[:-1], np.diff(path, axis=0)
    t = np.clip(((Q[:, None] - A[None]) * AB[None]).sum(-1) / (AB ** 2).sum(-1)[None], 0, 1)
    R = Q[:, None] - (A[None] + t[..., None] * AB[None])
    return np.hypot(R[..., 0], R[..., 1]).min(1)


def ade_lat(C2, path):   # mean distance of the candidate's poses to the human's path polyline
    return path_dist(C2.reshape(-1, 2), path).reshape(len(C2), -1).mean(-1)


def softmin(a, tau, mask=None):   # softmax(-a / tau) over the mask (all when None), float64
    q = np.zeros(len(a)); m = np.ones(len(a), bool) if mask is None else mask
    z = a[m]; e = np.exp(-(z - z.min()) / tau); q[m] = e / e.sum()
    return q


# wordings: ours; templates [train..., dev, dev] per instruction kind; {lo}/{hi} in m/s -----------------------------------------
WORDS = {
    "keep": [
        "Keep to your current lane with no lane change.",
        "Hold your lane; do not drift into another one.",
        "Remain in the lane you occupy at the moment.",
        "Do not switch lanes; carry on in this one.",
        "No lane changes, keep driving where you are.",
        "Drive on without leaving your lane.",
        "Maintain your lane and do not cross into another.",
        "Keep the car in its own lane throughout.",
    ],
    "left": [
        "Move one lane over to the left.",
        "Switch into the lane on your left.",
        "Move over a lane, to the left side.",
        "Slide over into the left-hand lane beside you.",
        "Get into the next lane to the left.",
        "Merge leftward by one lane.",
        "Occupy the lane on your left side from here on.",
        "Cross over into the neighbouring left lane.",
    ],
    "right": [
        "Move one lane over to the right.",
        "Switch into the lane on your right.",
        "Move over a lane, to the right side.",
        "Slide over into the right-hand lane beside you.",
        "Get into the next lane to the right.",
        "Merge rightward by one lane.",
        "Occupy the lane on your right side from here on.",
        "Cross over into the neighbouring right lane.",
    ],
    "turn_left": [
        "Turn left at the next intersection.",
        "Make the left turn when the intersection comes.",
        "Make a left when you reach the crossing.",
        "Head left through the coming intersection.",
        "At the crossroads coming up, head off to the left.",
        "Go left at the next crossroads.",
        "Your route bears left at the intersection ahead.",
        "Swing left once you get to the junction.",
    ],
    "straight": [
        "Cross the next intersection without turning.",
        "Carry straight on over the crossing ahead.",
        "Do not turn at the crossing; keep straight on.",
        "Drive straight over the coming intersection.",
        "Pass the intersection on a straight course.",
        "Keep heading straight at the next crossroads.",
        "Your route runs straight on through the intersection.",
        "No turning at the junction, straight across it.",
    ],
    "turn_right": [
        "Turn right at the next intersection.",
        "Make the right turn when the intersection comes.",
        "Make a right when you reach the crossing.",
        "Head right through the coming intersection.",
        "At the crossroads coming up, head off to the right.",
        "Go right at the next crossroads.",
        "Your route bears right at the intersection ahead.",
        "Swing right once you get to the junction.",
    ],
    "stop": [
        "Bring the car to a stop and stay stopped.",
        "Come to a halt and remain stationary.",
        "Brake to a standstill and hold there.",
        "Stop the vehicle and wait without moving.",
        "Pull up to a full stop and keep still.",
        "Halt here and do not move off.",
        "Ease the car to rest and keep it there.",
        "Come to a complete standstill and stay put.",
    ],
    "go": [
        "Keep moving and do not stop.",
        "Carry on driving forward.",
        "Proceed and keep the car rolling.",
        "Do not stop here; keep going ahead.",
        "Keep going forward without stopping.",
        "Drive on rather than stopping.",
        "Press on and keep moving.",
        "Move along without coming to a stop.",
    ],
    "band": [
        "Bring your speed into the {lo}-{hi} m/s range within 1.5 seconds and keep it there.",
        "Settle at a speed between {lo} and {hi} m/s by 1.5 s and hold it until 4 s.",
        "Drive at {lo} to {hi} m/s, reaching it within the first 1.5 seconds and holding it.",
        "Target {lo}-{hi} m/s: get there by 1.5 s and stay in that range.",
        "After 1.5 seconds your speed must sit within {lo}-{hi} m/s.",
        "Within 1.5 seconds, be moving at {lo} to {hi} m/s, and stay at that speed.",
        "From 1.5 s until 4 s your speed should stay between {lo} and {hi} m/s.",
        "Run at {lo}-{hi} m/s, getting there inside 1.5 seconds and keeping it steady.",
    ],
    "follow": [
        "Drive behind the car ahead at {lo}-{hi} m/s.",
        "Stay behind the car in front, moving at {lo} to {hi} m/s.",
        "Trail the vehicle in front while holding {lo}-{hi} m/s.",
        "Stay in line behind the front car at {lo} to {hi} m/s.",
        "Tail the vehicle in front of you at {lo}-{hi} m/s.",
        "Remain behind the leading car and drive at {lo} to {hi} m/s.",
        "Track the car ahead from behind at {lo}-{hi} m/s.",
        "Go after the vehicle in front at {lo} to {hi} m/s, staying behind it.",
    ],
    "normal": [
        "Drive at your usual pace.",
        "Keep a normal driving speed here.",
        "Carry on at an ordinary pace.",
        "Proceed at the speed you normally would.",
        "Keep your regular speed for this stretch.",
        "Go along at a typical pace for this road.",
        "Drive along at an everyday speed.",
        "Hold the pace you would normally keep.",
    ],
    "slow": [
        "Drive noticeably slower than usual.",
        "Ease off and take this stretch at a reduced pace.",
        "Go at a slower pace than normal.",
        "Reduce your speed well below your usual pace.",
        "Take it slow through here.",
        "Drive more slowly than you otherwise would.",
        "Lower your pace clearly below normal.",
        "Proceed at a markedly slower speed.",
    ],
}
NDEV = 2   # the last two templates of each kind are dev-only
NEAR_DUP = 0.65   # similarity() at or above this to any L2D string = a near-duplicate (the templates above are all below it)


def wording(label, fam, split, key):   # (text, template id) for an instruction label; the template drawn by crc32(key)
    kind = "follow" if fam == "follow" else label.split(":")[0]
    ws = WORDS[kind]; pool = range(len(ws) - NDEV) if split == "train" else range(len(ws) - NDEV, len(ws))
    i = list(pool)[zlib.crc32(key.encode()) % len(pool)]
    lo = float(label.split(":")[1]) if ":" in label else None
    text = ws[i].format(lo=f"{lo:g}", hi=f"{lo + 2:g}") if lo is not None else ws[i]
    return text, f"{kind}/{i}"


def norm_text(s):   # dedupe form: lower case, numbers -> #, letters / # / spaces only
    s = re.sub(r"\d+(\.\d+)?", "#", s.lower())
    return " ".join(re.sub(r"[^a-z# ]+", " ", s).split())


def l2d_strings():   # the 304 NAVSIM + 600 nuScenes L2D instruction strings (canonical + paraphrases) in the SDK workspace
    out = []
    for f in sorted(glob.glob(f"{L2D}/ws/language2drive.OpenloopTrack.navsim/data/public/public_inputs/*.json")):
        d = json.load(open(f)); out += d["instructions"] + [p for ps in d["paraphrases"] for p in ps]
    for l in open(f"{L2D}/ws/language2drive.OpenloopTrack.nScenes/releases/nuscenes-test-v1.0.0/public_inputs.jsonl"):
        for a in json.loads(l)["arms"].values():
            out += [a["instruction"]] + a["paraphrases"]
    return out


def similarity(a, b):   # max of character ratio and token Jaccard on the dedupe forms
    x, y = norm_text(a), norm_text(b); tx, ty = set(x.split()), set(y.split())
    return max(difflib.SequenceMatcher(None, x, y).ratio(), len(tx & ty) / max(len(tx | ty), 1))


# prompts: the adapter's own functions ---------------------------------------------------------------------------------------
def prompts():
    from superglance.data import scene as s3
    from superglance.l2d.adapter import build_prompt
    cfg = dict(prompt_mode="append", instruction_prefix="Driving instruction: ", command="go straight")
    return (lambda scene: s3.scene_head(scene) + s3.C256_INSTRUCTION), (lambda scene, text: build_prompt(s3, scene, text, cfg))


def selftest(n=400, seed=0):   # holds() == the SDK verifier on random candidates for every contract type used here
    rng = np.random.default_rng(seed); C = np.load(CANDS)[:, :, :2].astype(np.float64)
    idx = np.union1d(rng.choice(K, n, replace=False), np.flatnonzero(feats(C)["s"][:, 5:].max(1) <= 0.6))   # + every stopping candidate
    X = feats(C[idx]); strip = [[-5, -1.9], [60, -1.9], [60, 1.9], [-5, 1.9]]; bend = [[-5, -1.8], [20, -1.8], [45, 12], [42, 16], [18, 1.8], [-5, 1.8]]
    tg = [[8 + 4.5 * t, 0.2 * t] for t in T]
    cs = [c_band(3.0), c_band(7.5), c_stop(), c_stop((10.0, 0.0), 2.0), C_GO, c_follow(4.0, tg), c_follow(1.0, tg),
          {"type": "lane_region", "polygon_xy": strip, "start_s": 0.5, "end_s": 4.0, "min_progress_m": 5.0},
          {"type": "lane_region", "polygon_xy": bend, "start_s": 3.5, "end_s": 4.0, "min_progress_m": 0.0}]
    for c in cs:
        a = holds(c, X); b = np.array([verify_one(C[i], c) for i in idx])
        assert (a == b).all(), (c if isinstance(c, list) else c["type"], int((a != b).sum()))
    q = softmin(ade(C, C[7]), 1.0, np.arange(K) % 2 == 1)
    assert abs(q.sum() - 1) < 1e-12 and q[::2].max() == 0
    return {c["type"] if isinstance(c, dict) else "go": int(holds(c, X).sum()) for c in cs}


if __name__ == "__main__":
    print(selftest())
