"""AlpaSim driver service for SuperGlance: Qwen3.5-9B + LoRA + the Policy Head over K candidate trajectories, served through
AlpaSim's EgodriverService gRPC API. AlpaSim calls the driver every 100 ms of simulated time, and the policy plans at every call:
- the latest frame of --camera (the front wide 120-degree f-theta camera); with --rectify-f0 re-projected into nuPlan's CAM_F0,
  the camera of the training rows (rotation only, linear interpolation, JPEG quality 92);
- velocity, acceleration and yaw rate from the latest dynamic state (from the poses while the state is zero and the poses move);
- the rig poses 1.5 / 1.0 / 0.5 s ago in the current rig frame (interpolated), as NAVSIM's history_poses;
- the command from the route heading about 4 s ahead (command_from_route; --cmd-rule ego measures the look-ahead from the ego);
- with --hist, the front frames 1.5 / 1.0 / 0.5 s ago, rectified and shrunk as the training rows' past frames (each prepared
  once per session: PastCache and the session's JPEG cache give the values of preparing it at every call);
- the training rows' prompt (superglance.data.scene.c256_row) and one forward; the plan is the softmax-weighted mean
  (temperature 0.2) of the 64 most probable candidates' poses (read_mean); the log's choice is the argmax;
- with --exec navsim, the plan as NAVSIM's PDM simulator executes it from the car's state, 40 poses 0.1 s apart
  (superglance.executor.stitched, a sidecar in the NAVSIM environment; it starts from the steering angle its run of the previous
  plan reaches now); otherwise the plan's 8 poses 0.5 s apart. Composed with the current rig pose, led by that pose at time_now.
Every call is logged to <--frames>/decisions.jsonl.
usage (training environment with AlpaSim's alpasim_grpc; --navsim-python: the NAVSIM environment's interpreter):
  python -m superglance.alpasim.driver --port 6010 --model <stage-2 merged base> --adapter <adapter> --cands <cands.npy>
      --rectify-f0 --cmd-rule ego --hist --exec navsim --navsim-python <python> [--navsim-repo <NAVSIM devkit checkout>]
  python -m superglance.alpasim.driver --selftest   (geometry checks, no model)"""
import argparse, functools, json, logging, os, subprocess, threading, time
from concurrent import futures
import numpy as np

log = logging.getLogger("superglance_driver")
COMMANDS = ["turn left", "go straight", "turn right", "unknown"]   # NAVSIM one-hot order; AlpaSim's DriveCommand differs


# ---- geometry (rig: x forward, y left, z up; origin rear axle on the ground; same as the NAVSIM ego frame) ----
def quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return w, x, y, z


def Rz(h):
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def past_in_rig(times, P, yaws, t_now, offsets_s=(1.5, 1.0, 0.5)):
    """NAVSIM history_poses: [[x, y, heading]] at now - 1.5 / 1.0 / 0.5 s and now, in the rig frame at now. Positions and
    yaws interpolated between the received rig poses (clamped to the oldest one)."""
    i_now = int(np.argmin(np.abs(times - t_now)))
    p0, h0 = P[i_now], yaws[i_now]
    Rt = Rz(h0).T
    uy = np.unwrap(yaws)
    out = []
    for dt in offsets_s:
        t = t_now - dt * 1e6
        p = np.array([np.interp(t, times, P[:, k]) for k in range(2)])
        h = float(np.interp(t, times, uy))
        d = Rt[:2, :2] @ (p - p0[:2])
        out.append([float(d[0]), float(d[1]), float((h - uy[i_now] + np.pi) % (2 * np.pi) - np.pi)])
    return out + [[0.0, 0.0, 0.0]]


def motion_from_poses(times, P, yaws, t_now, dt=0.5):
    """Velocity and acceleration (x forward, y left) in the rig frame at now from the rig poses: backward differences over dt."""
    def at(t):
        return np.array([np.interp(t, times, P[:, k]) for k in range(2)])
    Rt = Rz(yaws[int(np.argmin(np.abs(times - t_now)))]).T[:2, :2]
    v1 = Rt @ (at(t_now) - at(t_now - dt * 1e6)) / dt
    v0 = Rt @ (at(t_now - dt * 1e6) - at(t_now - 2 * dt * 1e6)) / dt
    return tuple(v1), tuple((v1 - v0) / dt)


def command_from_route(waypoints, speed=0.0, horizon_s=4.0, lead_m=10.0, turn_deg=20.0, from_ego=False):
    """NAVSIM-style command from AlpaSim's route (rig-frame waypoints). In navtest a turn command means a real turn (the
    human's heading 4 s later: left median +36 deg, right -33 deg; straight 90% within +-6.5 deg, lane changes included),
    so the route's heading where the ego will be in about horizon_s (arc length speed * horizon_s + lead_m, 15-60 m)
    decides: > turn_deg left, < -turn_deg right, else straight. (AlpaSim's own rule, |y| > 2 m at the first waypoint 5 m
    ahead, calls a lane offset at 15 m/s a right turn, which can drive the car off the road.)
    from_ego (--cmd-rule ego): the arc length runs from the ego, through the straight gap to the route's start, then
    along the route. With the default the gap counts as driven, so a car slower than about 8 m/s reads the heading at the
    route's start, wherever it will be in 4 s. Offline on 1,475 nuPlan rollouts: agreement with the human's
    4 s heading change 78.0% -> 85.3% of drive calls; 8 of the 9 failed rollouts whose command changes get the human's command."""
    pts = np.asarray(waypoints, np.float64).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(1)]   # the nuPlan track pads the route with NaN, which would make every call "straight"
    if len(pts) < (1 if from_ego else 2):
        return [0, 0, 0, 1]
    d = np.hypot(pts[:, 0], pts[:, 1])
    i0 = int(np.argmin(d))   # the route point nearest the ego
    pts = pts[i0:]
    if from_ego:
        pts = np.vstack([[0.0, 0.0], pts])
    if len(pts) < 2:
        return [0, 1, 0, 0]
    seg = np.diff(pts, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    # the route starts about 40 m ahead (route_start_offset_m) on the nuPlan track and in the PAI-AV scenes too: by default
    # that gap counts as route already driven
    L = float(np.clip(speed * horizon_s + lead_m, 15.0, 60.0))
    if not from_ego:
        L = max(L - float(d[i0]), 0.0) if d[i0] > 10.0 else L
    k = min(int(np.searchsorted(s, L)), len(seg)) - 1
    k = max(k, 0)
    h = np.degrees(np.arctan2(seg[k, 1], seg[k, 0]))
    return [1, 0, 0, 0] if h > turn_deg else [0, 0, 1, 0] if h < -turn_deg else [0, 1, 0, 0]


def compose(p_now, R_now, plan):
    """Plan (8 x [x, y, heading], rig frame at now) -> local poses (position, rotation)."""
    return [(p_now + R_now @ np.array([x, y, 0.0]), R_now @ Rz(h)) for x, y, h in plan]


# ---- --rectify-f0: nuPlan CAM_F0, as the PAI-AV training rows ----
K_F0 = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
D_F0 = np.array([-0.3561, 0.1725, -0.0021, 0.0005, -0.0523])   # nuPlan CAM_F0 (k1, k2, p1, p2, k3)
R_F0 = np.array([[0.0031, -0.0253, 0.9997], [-1.0, -0.0088, 0.0029], [0.0088, -0.9996, -0.0253]])   # CAM_F0 optical -> rig
W_F0, H_F0 = 1920, 1080


@functools.lru_cache(1)
def f0_rays():
    """Every CAM_F0 pixel's undistorted ray in the rig frame (row vectors)."""
    import cv2
    u, v = np.meshgrid(np.arange(W_F0, dtype=np.float64), np.arange(H_F0, dtype=np.float64))
    xn = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2), K_F0, D_F0).reshape(-1, 2)
    return np.concatenate([xn, np.ones((len(xn), 1))], 1) @ R_F0.T


def cam_to_rig(q):
    """Camera (x right, y down, z forward) -> rig rotation of an AvailableCamera's rig_to_camera quaternion, which holds
    camera -> rig (the proto's own comment); R or R.T, whichever looks forward with y down, else the conventions differ."""
    R = quat_to_R(q.w, q.x, q.y, q.z)
    for M in (R, R.T):
        if M[0, 2] > 0.8 and M[2, 1] < -0.8:
            return M
    raise ValueError(f"camera axes not forward / down: {np.round(R, 3).tolist()}")


def ftheta_px(rays, spec, w, h):
    """Camera-frame rays -> pixels of a w x h frame of the f-theta CameraSpec and validity; spec
    pixels scaled when the frame's size differs from the spec's resolution."""
    f = spec.ftheta_param
    pp = np.array([f.principal_point_x, f.principal_point_y])
    fw = list(f.angle_to_pixeldist_poly)
    if not fw:   # only the backward polynomial: invert it numerically
        r = np.linspace(0, max(np.hypot(x - pp[0], y - pp[1]) for x in (0, spec.resolution_w) for y in (0, spec.resolution_h)), 500)
        fw = np.polynomial.polynomial.polyfit(np.polynomial.polynomial.polyval(r, list(f.pixeldist_to_angle_poly)), r, 6)
    cde = (f.linear_cde.linear_c, f.linear_cde.linear_d, f.linear_cde.linear_e)
    c, d, e = cde if any(cde) else (1.0, 0.0, 0.0)
    n = np.linalg.norm(rays[:, :2], axis=1)
    theta = np.arctan2(n, rays[:, 2])
    s = np.divide(np.polynomial.polynomial.polyval(theta, fw), n, out=np.zeros_like(n), where=n > 1e-9)
    px = ((rays[:, :2] * s[:, None]) @ np.array([[c, d], [e, 1.0]]).T + pp) * (w / (spec.resolution_w or w), h / (spec.resolution_h or h))
    ok = (rays[:, 2] > 0) & (theta <= (f.max_angle or np.pi) + 1e-6) & (px[:, 0] >= 0) & (px[:, 0] <= w - 1) & (px[:, 1] >= 0) & (px[:, 1] <= h - 1)
    return px, ok


def f0_remap(cam, w, h):
    """cv2.remap maps: for every CAM_F0 pixel the source pixel in the camera's w x h f-theta frame (-1 outside)."""
    px, ok = ftheta_px(f0_rays() @ cam_to_rig(cam.rig_to_camera.quat), cam.intrinsics, w, h)
    px[~ok] = -1
    return px[:, 0].reshape(H_F0, W_F0).astype(np.float32), px[:, 1].reshape(H_F0, W_F0).astype(np.float32)


def hist_pick(fh, image_t, offs=(1.5, 1.0, 0.5), tol=100_000):
    """--hist: the (t_us, jpeg) of fh nearest to image_t - each offset (oldest first), or None unless all are within tol."""
    out = [min(fh, key=lambda x: abs(x[0] - (image_t - o * 1e6)), default=None) for o in offs]
    return out if all(x is not None and abs(x[0] - (image_t - o * 1e6)) <= tol for x, o in zip(out, offs)) else None


def selftest():
    R = quat_to_R(*R_to_quat(Rz(0.7)))
    assert np.allclose(R, Rz(0.7)) and abs(yaw(R) - 0.7) < 1e-9
    # ego drove straight along local +y (heading 90 deg) at 10 m/s: 1.5 s ago it was 15 m behind, i.e. x = -15 in the rig frame
    times = np.arange(0, 3.01, 0.1) * 1e6
    P = np.stack([np.zeros_like(times), times / 1e6 * 10, np.zeros_like(times)], 1)
    hist = past_in_rig(times, P, np.full(len(times), np.pi / 2), times[-1])
    assert np.allclose(np.array(hist)[:, :2], [[-15, 0], [-10, 0], [-5, 0], [0, 0]], atol=1e-6), hist
    assert np.allclose(np.array(past_in_rig(times[:1], P[:1], np.full(1, np.pi / 2), times[0]))[:, :2], 0)   # default: clamped
    v, a = motion_from_poses(times, P, np.full(len(times), np.pi / 2), times[-1])
    assert np.allclose(v, (10, 0), atol=1e-6) and np.allclose(a, (0, 0), atol=1e-6), (v, a)
    lane_change = [(x, -3.5 * min(max((x - 5) / 20, 0), 1)) for x in range(0, 80, 2)]   # 3.5 m to the right over 20 m
    assert command_from_route(lane_change, 15.0) == [0, 1, 0, 0]
    right_turn = [(x, 0.0) for x in range(0, 20, 2)] + [(20 + 8 * np.sin(a), -8 + 8 * np.cos(a)) for a in np.linspace(0.2, np.pi / 2, 8)] + [(28, -8 - d) for d in range(2, 40, 2)]
    assert command_from_route(right_turn, 5.0) == [0, 0, 1, 0] and command_from_route([(x, -y) for x, y in right_turn], 5.0) == [1, 0, 0, 0]
    assert command_from_route([(x, 0.0) for x in range(-10, 80, 2)], 20.0) == [0, 1, 0, 0] and command_from_route([]) == [0, 0, 0, 1]
    # nuPlan track: the route starts 40 m ahead; a left turn 48-56 m ahead is "straight" at 5 m/s (look-ahead 30 m) and "left" at 15 m/s (60 m)
    late_left = [(x, 0.0) for x in range(40, 48, 2)] + [(48 + 8 * np.sin(a), 8 - 8 * np.cos(a)) for a in np.linspace(0.2, np.pi / 2, 8)] + [(56, 8 + d) for d in range(2, 40, 2)]
    assert command_from_route(late_left, 5.0) == [0, 1, 0, 0] and command_from_route(late_left, 15.0) == [1, 0, 0, 0]
    assert command_from_route(late_left + [(np.nan, np.nan)] * 6, 15.0) == [1, 0, 0, 0]   # NaN padding (nuPlan track route)
    assert command_from_route(late_left, 5.0, from_ego=True) == [0, 1, 0, 0] and command_from_route(late_left, 15.0, from_ego=True) == [1, 0, 0, 0]
    # from_ego: a car at 3 m/s (look-ahead 22 m), 40 m before a route that starts bending right, is still going straight
    bend = [(40 + 2 * i, -2.0 * i) for i in range(20)]
    assert command_from_route(bend, 3.0) == [0, 0, 1, 0] and command_from_route(bend, 3.0, from_ego=True) == [0, 1, 0, 0]
    (p, Rw), = compose(np.array([1.0, 2.0, 0.0]), Rz(np.pi / 2), [(5.0, 0.0, 0.1)])
    assert np.allclose(p, [1, 7, 0]) and abs(yaw(Rw) - (np.pi / 2 + 0.1)) < 1e-9
    # --rectify-f0: an equidistant f-theta camera (r = 1000 theta) looking where CAM_F0 looks
    import cv2
    from types import SimpleNamespace as NS
    quat = lambda R: NS(**dict(zip("wxyz", R_to_quat(R))))
    ft = NS(angle_to_pixeldist_poly=[0.0, 1000.0], pixeldist_to_angle_poly=[], principal_point_x=960.0, principal_point_y=540.0,
            linear_cde=NS(linear_c=0.0, linear_d=0.0, linear_e=0.0), max_angle=1.1)
    cam = NS(intrinsics=NS(ftheta_param=ft, resolution_w=1920, resolution_h=1080), rig_to_camera=NS(quat=quat(R_F0)))
    mx, my = f0_remap(cam, 1920, 1080)
    xn = cv2.undistortPoints(np.array([[[1500.0, 560.0]]]), K_F0, D_F0)[0, 0, 0]
    assert abs(mx[560, 960] - 960) < 2 and abs(my[560, 960] - 540) < 2, (mx[560, 960], my[560, 960])   # principal point -> principal point
    assert abs(mx[560, 1500] - (960 + 1000 * np.arctan(xn))) < 2 and abs(my[560, 1500] - 540) < 2, (mx[560, 1500], my[560, 1500])
    assert np.allclose(f0_remap(cam, 960, 540)[0][560, 1500], mx[560, 1500] / 2, atol=1e-3)   # a half-size frame: half the pixels
    assert np.allclose(cam_to_rig(quat(R_F0.T)), cam_to_rig(quat(R_F0)))   # rig -> camera given: transposed back
    ft.pixeldist_to_angle_poly, ft.angle_to_pixeldist_poly = [0.0, 1e-3], []   # backward polynomial only
    assert np.abs(f0_remap(cam, 1920, 1080)[0] - mx).max() < 0.5
    fh = [(t, b"") for t in range(0, 3_000_001, 100_000)]   # --hist: 10 Hz front frames over 3 s
    assert [t for t, _ in hist_pick(fh, 3_000_000)] == [1_500_000, 2_000_000, 2_500_000]
    assert [t for t, _ in hist_pick(fh[::5], 3_040_000)] == [1_500_000, 2_000_000, 2_500_000] and hist_pick(fh[17:], 3_000_000) is None
    print("SELFTEST_OK")


# ---- model: the navtest decode, one scene at a time ----

def read_mean(C, L):   # the plan: the tau 0.2 softmax-weighted mean of the head's top 64 candidates
    top = np.argsort(-L)[:64]; q = np.exp((L[top] - L[top[0]]) / 0.2); q /= q.sum()
    return (q[:, None, None] * C[top]).sum(0)


class PastCache:   # the image processor's output of a past-frame image, by its exact pixels; other images as before
    def __init__(self, ip, n=64):
        import collections
        from superglance.model import history as mf_hist
        self.ip, self.n, self.px, self.c = ip, n, mf_hist.HIST_PIXELS, collections.OrderedDict()

    def __getattr__(self, k):   # every other attribute (merge_size, patch_size, ...) is the processor's own
        if k == "ip":   # not yet set (copy / pickle): no recursion
            raise AttributeError(k)
        return getattr(self.ip, k)

    def __call__(self, images=None, **kw):
        import torch
        from transformers import BatchFeature
        outs = []
        for im in images:
            key = (im.size, im.mode, im.tobytes()) if im.size[0] * im.size[1] <= self.px else None
            o = self.c.get(key) if key is not None else None
            if o is None:
                o = dict(self.ip(images=[im], **kw))
                if key is not None:
                    self.c[key] = o
                    if len(self.c) > self.n:
                        self.c.popitem(last=False)
            else:
                self.c.move_to_end(key)
            outs.append(o)
        return BatchFeature({k: torch.cat([o[k] for o in outs]) for k in outs[0]})


class Planner:
    def __init__(self, model_dir, adapter, cands, max_pixels=589824):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from superglance.data.policy import DecisionCollator, candidate_token_ids, readout
        from peft import PeftModel
        self.torch, self.readout = torch, readout
        self.cands = np.load(cands)
        self.K = len(self.cands)
        processor = AutoProcessor.from_pretrained(model_dir, min_pixels=65536, max_pixels=max_pixels)
        processor.image_processor = PastCache(processor.image_processor)
        model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
        model.choice_head = torch.nn.Linear(model.config.text_config.hidden_size, self.K, dtype=model.dtype)
        self.model = PeftModel.from_pretrained(model, adapter).cuda().eval()
        self.collate = DecisionCollator(processor)
        self.token_ids = candidate_token_ids(processor, 16)

    def plan(self, scene):
        from superglance.data.scene import c256_row
        row_ = c256_row(scene, k=self.K)
        if scene.get("hist_frames"):   # --hist: the past frames, oldest first; the collator lays them out
            row_.update(extra_image_files=scene["hist_frames"], history=True)
        inputs = self.collate([row_])
        self.n_img = len(inputs["image_grid_thw"])   # --hist: the images read (logged)
        question = row_["question"]
        inputs.pop("gold_index"), inputs.pop("candidate_count")
        torch = self.torch
        with torch.inference_mode():
            inputs = inputs.to("cuda")
            lg = self.readout(self.model, inputs, self.token_ids, torch.ones(1, dtype=torch.bool), self.K)
            head = j = int(lg.argmax(-1)[0])
        head, j = int(head), int(j)
        p = lg.softmax(-1)[0].float().cpu().numpy()
        return j, float(p[j]), [[int(c), round(float(p[c]), 5)] for c in np.argsort(p)[::-1][:8]], read_mean(self.cands, lg[0].float().cpu().numpy()), question, p, head


# ---- gRPC service ----
def serve(args):
    import grpc
    from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
    planner = Planner(args.model, args.adapter, args.cands)
    if args.hist:
        from superglance.model import history as mf_hist
        from PIL import Image
    os.makedirs(args.frames, exist_ok=True)
    trace = open(os.path.join(args.frames, "decisions.jsonl"), "a")
    lock = threading.Lock()   # one model, one request at a time; a queue per GPU if rollouts run in parallel
    executor = None
    if args.exec == "navsim":
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo, for superglance.executor
        executor = subprocess.Popen([args.navsim_python, "-m", "superglance.executor.stitched"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                    env={**os.environ, "PYTHONPATH": os.pathsep.join(x for x in (root, args.navsim_repo) if x), "OMP_NUM_THREADS": "1"})

    class Session:
        def __init__(self):
            self.image, self.image_t, self.poses, self.dyn, self.route = None, 0, {}, {}, []
            self.prev_states, self.prev_t = None, 0   # --exec navsim: the last executed plan's tire steering (steer_at, 41 values) and its call time
            self.cam, self.remap = None, None   # --rectify-f0: the session's camera; ((w, h), map x, map y)
            self.fh, self.hist_ms = [], None   # --hist: the front frames of the last 2 s [(frame_end_us, jpeg)]; the last re-plan's offsets
            self.pj = {}   # --hist: frame_end_us -> (the frame's bytes, its past-frame JPEG)

    sessions = {}

    class Driver(egodriver_pb2_grpc.EgodriverServiceServicer):
        def start_session(self, req, ctx):
            sessions[req.session_uuid] = Session()
            cams = [c.logical_id for c in req.rollout_spec.vehicle.available_cameras]
            log.info("start %s scene=%s cameras=%s", req.session_uuid, req.debug_info.scene_id if req.HasField("debug_info") else "?", cams)
            if args.rectify_f0:
                cam = next(c for c in req.rollout_spec.vehicle.available_cameras if c.logical_id == args.camera)
                assert cam.intrinsics.WhichOneof("camera_param") == "ftheta_param", cam.intrinsics.WhichOneof("camera_param")
                f, M = cam.intrinsics.ftheta_param, cam_to_rig(cam.rig_to_camera.quat)
                log.info("rectify %s: %dx%d pp (%.1f, %.1f) fw %s bw %s cde (%.4f, %.4f, %.4f) max_angle %.3f; camera x / y / z in the rig %s",
                         req.session_uuid, cam.intrinsics.resolution_w, cam.intrinsics.resolution_h, f.principal_point_x, f.principal_point_y,
                         np.round(list(f.angle_to_pixeldist_poly), 6).tolist(), len(f.pixeldist_to_angle_poly), f.linear_cde.linear_c,
                         f.linear_cde.linear_d, f.linear_cde.linear_e, f.max_angle, np.round(M.T, 3).tolist())
                sessions[req.session_uuid].cam = cam
            return common_pb2.SessionRequestStatus()

        def close_session(self, req, ctx):
            sessions.pop(req.session_uuid, None)
            return common_pb2.Empty()

        def submit_image_observation(self, req, ctx):
            s = sessions[req.session_uuid]
            c = req.camera_image
            if c.logical_id == args.camera and c.frame_end_us >= s.image_t:
                s.image, s.image_t = c.image_bytes, c.frame_end_us
                if args.hist:
                    s.fh = [x for x in s.fh if x[0] >= c.frame_end_us - 2_000_000] + [(c.frame_end_us, c.image_bytes)]
            return common_pb2.Empty()

        def submit_egomotion_observation(self, req, ctx):
            s = sessions[req.session_uuid]
            dyns = list(req.dynamic_states)
            for i, pose in enumerate(req.trajectory.poses):
                q, v = pose.pose.quat, pose.pose.vec
                s.poses[pose.timestamp_us] = (np.array([v.x, v.y, v.z]), quat_to_R(q.w, q.x, q.y, q.z))
                if i < len(dyns):   # dynamic states come 1:1 with the poses; keep the last known one otherwise
                    d = dyns[i]
                    s.dyn[pose.timestamp_us] = ((d.linear_velocity.x, d.linear_velocity.y), (d.linear_acceleration.x, d.linear_acceleration.y),
                                                d.angular_velocity.z)
            return common_pb2.Empty()

        def submit_route(self, req, ctx):
            sessions[req.session_uuid].route = [(w.x, w.y) for w in req.route.waypoints if np.isfinite(w.x) and np.isfinite(w.y)]
            return common_pb2.Empty()

        def submit_recording_ground_truth(self, req, ctx):
            return common_pb2.Empty()

        def get_version(self, req, ctx):
            from alpasim_grpc import API_VERSION_MESSAGE
            return common_pb2.VersionId(version_id="superglance", git_hash="none", grpc_api_version=API_VERSION_MESSAGE)

        def drive(self, req, ctx):
            s = sessions[req.session_uuid]
            if s.image is None or not s.poses:
                return egodriver_pb2.DriveResponse(trajectory=common_pb2.Trajectory())
            t0 = time.time()
            times = np.array(sorted(s.poses), dtype=np.float64)
            P = np.stack([s.poses[int(t)][0] for t in times])
            yaws = np.array([yaw(s.poses[int(t)][1]) for t in times])
            t_last = int(times[-1])
            p_now, R_now = s.poses[t_last]
            vel, acc, w = s.dyn[max(s.dyn)] if s.dyn else ((0.0, 0.0), (0.0, 0.0), 0.0)
            vfd, afd = motion_from_poses(times, P, yaws, t_last)
            src = "state"
            if np.hypot(*vel) < 0.2 and np.hypot(*vfd) > 1.0:   # zero dynamic states during the force-GT replay
                vel, acc, src = vfd, afd, "poses"
                w = float((np.unwrap(yaws)[-1] - np.unwrap(yaws)[-2]) / ((times[-1] - times[-2]) / 1e6)) if len(times) > 1 else 0.0
            img = os.path.join(args.frames, f"{req.session_uuid}.jpg")
            if args.rectify_f0:   # as the PAI-AV rows: remap with linear interpolation, JPEG quality 92
                import cv2
                im = cv2.imdecode(np.frombuffer(s.image, np.uint8), cv2.IMREAD_COLOR)
                if s.remap is None or s.remap[0] != im.shape[1::-1]:
                    s.remap = (im.shape[1::-1], *f0_remap(s.cam, im.shape[1], im.shape[0]))
                    s.pj = {}   # past-frame JPEGs are made with these maps
                rim = cv2.remap(im, s.remap[1], s.remap[2], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                cv2.imwrite(img, rim, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if args.hist:   # this frame's past-frame JPEG, for the calls 0.5 / 1.0 / 1.5 s on
                    s.pj[s.image_t] = (s.image, mf_hist.hist_jpeg(Image.fromarray(cv2.cvtColor(rim, cv2.COLOR_BGR2RGB))))
            else:
                with open(img, "wb") as f:
                    f.write(s.image)
            scene = {"token": f"{req.session_uuid}_{req.time_now_us}", "image": img, "route": list(s.route), "velocity": list(vel), "acceleration": list(acc),
                     "command": command_from_route(s.route, float(np.hypot(*vel)), from_ego=args.cmd_rule == "ego"),
                     "past": past_in_rig(times, P, yaws, t_last)}
            if args.hist:   # rectified as the current frame above, then the history rows' shrink and JPEG
                def past_jpeg(t, b):   # the session's JPEG of this frame when made from these bytes
                    v = s.pj.get(t)
                    if v is None or v[0] != b:
                        v = s.pj[t] = (b, mf_hist.hist_jpeg(Image.fromarray(cv2.cvtColor(cv2.remap(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR),
                            s.remap[1], s.remap[2], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), cv2.COLOR_BGR2RGB))))
                    return v[1]
                got = hist_pick(s.fh, s.image_t)
                if got:
                    scene["hist_frames"] = [past_jpeg(t, b) for t, b in got]
                s.hist_ms = [round((t - s.image_t) / 1000) for t, _ in got] if got else None
                s.pj = {t: v for t, v in s.pj.items() if t >= s.image_t - 2_000_000}   # as s.fh, the last 2 s
            with lock:
                j, pj, top8, plan, question, p, head = planner.plan(scene)
                nimg_ = planner.n_img
            traj = common_pb2.Trajectory()

            def add(p, R, t):
                w, x, y, z = R_to_quat(R)
                traj.poses.append(common_pb2.PoseAtTime(pose=common_pb2.Pose(vec=common_pb2.Vec3(x=p[0], y=p[1], z=p[2]),
                                                                               quat=common_pb2.Quat(w=w, x=x, y=y, z=z)), timestamp_us=int(t)))
            poses, dt_us = plan, 500_000
            ev, ea, ew = vel, acc, w   # the executor's start state
            if executor is not None:   # the real state, not the prompt's
                k = int(round((req.time_now_us - s.prev_t) / 1e5))   # stitching: the last executed plan's steering at now
                st = s.prev_states[k] if s.prev_states is not None and 1 <= k <= 5 else None
                executor.stdin.write(json.dumps({"plan": np.asarray(plan).tolist(), "v": [float(x) for x in ev], "a": [float(x) for x in ea],
                                                 "yaw_rate": float(ew), "steer": st}) + "\n")
                executor.stdin.flush()
                o = json.loads(executor.stdout.readline())
                poses, dt_us, s.prev_states, s.prev_t = o["poses"], 100_000, o["steer_at"], req.time_now_us
            add(p_now, R_now, req.time_now_us)
            for k, (p, R) in enumerate(compose(p_now, R_now, poses), 1):
                add(p, R, req.time_now_us + dt_us * k)
            rec = dict(session=req.session_uuid, t_us=req.time_now_us, pose_t_us=t_last, image_t_us=s.image_t, question=question,
                       command=COMMANDS[int(np.argmax(scene["command"]))], v_src=src, v=np.round(vel, 3).tolist(), a=np.round(acc, 3).tolist(),
                       v_poses=np.round(vfd, 3).tolist(), route_n=len(s.route), route_far=[round(float(c), 1) for c in s.route[-1]] if s.route else None, choice=j, prob=round(pj, 5), top8=top8,
                       plan=np.round(plan, 3).tolist(), ms=round(1000 * (time.time() - t0)))
            if args.hist:
                rec["hist_ms"], rec["n_img"] = s.hist_ms, nimg_
            if executor is not None:
                rec["exec"] = np.round(np.asarray(poses)[4::5], 3).tolist()   # executed poses at 0.5 ... 4.0 s
            trace.write(json.dumps(rec) + "\n"); trace.flush()
            log.info("drive t=%d cmd=%s v=%.1f choice=%d p=%.3f end=(%.1f, %.1f) %d ms", req.time_now_us, rec["command"],
                     float(np.hypot(*vel)), j, pj, plan[-1][0], plan[-1][1], rec["ms"])
            return egodriver_pb2.DriveResponse(trajectory=traj)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8), options=[("grpc.max_receive_message_length", 64 << 20)])
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(Driver(), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    log.info("driver on %s:%d (K = %d, camera %s)", args.host, args.port, planner.K, args.camera)
    server.wait_for_termination()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--cands")
    ap.add_argument("--camera", default="camera_front_wide_120fov")
    ap.add_argument("--frames", default="/dev/shm/superglance_driver")
    ap.add_argument("--exec", choices=("plan", "navsim"), default="plan",
                    help="navsim: send the plan as NAVSIM's PDM simulator executes it from the current state (superglance.executor.stitched)")
    ap.add_argument("--navsim-python", default="python", help="the NAVSIM environment's interpreter, for the --exec navsim sidecar")
    ap.add_argument("--navsim-repo", default="", help="a NAVSIM devkit checkout put on the sidecar's PYTHONPATH (empty: installed)")
    ap.add_argument("--cmd-rule", choices=("gap", "ego"), default="gap",
                    help="ego: measure the command's look-ahead from the ego through the gap to the route start (command_from_route)")
    ap.add_argument("--hist", action="store_true", help="the front frames 1.5 / 1.0 / 0.5 s ago as the model's history (needs --rectify-f0)")
    ap.add_argument("--rectify-f0", action="store_true", help="re-project the camera's f-theta frame into nuPlan's CAM_F0 (as the PAI SFT rows)")
    a = ap.parse_args()
    assert not a.hist or a.rectify_f0, "--hist rectifies the past frames as the current one: needs --rectify-f0"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    selftest() if a.selftest else serve(a)
