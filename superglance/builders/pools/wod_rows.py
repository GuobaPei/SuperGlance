"""WOD-E2E frames -> rows in the policy's prompt format (no road-users block). Per selected frame:
- image: FRONT, FRONT_LEFT and FRONT_RIGHT re-projected into nuPlan's CAM_F0 (K, distortion and mounting rotation; rotation only);
  each CAM_F0 pixel comes from the camera whose optical axis is nearest; JPEG quality 92;
- prompt: speed |v|, longitudinal acceleration, command from the intent, past positions at -1.5 / -1.0 / -0.5 s (past_states is
  4 Hz in the frame's ego frame, the last point is now);
- future: the 20 future_states points (4 Hz, 0.25-5 s); rated frames also keep the rater trajectories and scores.
--sel rated: frames with rater trajectories (val's RFS set); --sel every:N: frames whose index in the segment is a multiple of N.
Out: <out>/img/<id>.jpg, <out>/meta/<shard>.jsonl (one JSON per frame), then <out>/meta/<shard>.done.
usage (WOD-E2E protos importable, or WOD_PROTOS = their dir): python -m superglance.builders.pools.wod_rows --sel rated --out <dir> <shard>..."""
import argparse, json, os, struct, sys
import cv2, numpy as np
sys.path[:0] = [os.environ["WOD_PROTOS"]] if "WOD_PROTOS" in os.environ else []
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e

K_F0 = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
D_F0 = np.array([-0.3561, 0.1725, -0.0021, 0.0005, -0.0523])   # nuPlan CAM_F0 (k1, k2, p1, p2, k3)
R_F0 = np.array([[0.0031, -0.0253, 0.9997], [-1.0, -0.0088, 0.0029], [0.0088, -0.9996, -0.0253]])   # CAM_F0 optical -> ego
W_F0, H_F0 = 1920, 1080
CMD = {"GO_STRAIGHT": "go straight", "GO_LEFT": "turn left", "GO_RIGHT": "turn right", "UNKNOWN": "unknown"}
CAMS = (dataset_pb2.CameraName.FRONT, dataset_pb2.CameraName.FRONT_LEFT, dataset_pb2.CameraName.FRONT_RIGHT)
FINAL = "Choose the trajectory for the next 4 seconds."

u, v = np.meshgrid(np.arange(W_F0, dtype=np.float64), np.arange(H_F0, dtype=np.float64))
xn = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2), K_F0, D_F0).reshape(-1, 2)
RAYS = np.concatenate([xn, np.ones((len(xn), 1))], 1) @ R_F0.T   # every CAM_F0 pixel's ray in the ego frame (x ahead, y left, z up)
_maps = {}


def records(path):   # TFRecord framing without TensorFlow (length, crc, data, crc); crcs not checked
    with open(path, "rb") as f:
        while len(h := f.read(12)) == 12:
            n = struct.unpack("<Q", h[:8])[0]; d = f.read(n); f.read(4); yield d


def maps(calibs):   # camera -> (map x, map y) for cv2.remap, -1 where another camera (or none) shows the CAM_F0 pixel
    key = tuple((c.name, *np.round(c.intrinsic, 3), *np.round(c.extrinsic.transform, 5)) for c in calibs if c.name in CAMS)
    if key in _maps:
        return _maps[key]
    proj, best = {}, np.full(len(RAYS), np.inf)
    for c in calibs:
        if c.name not in CAMS:
            continue
        p = RAYS @ np.array(c.extrinsic.transform).reshape(4, 4)[:3, :3]   # ego -> camera (camera -> vehicle rotation, transposed)
        z = np.maximum(p[:, 0], 1e-3); x, y = -p[:, 1] / z, -p[:, 2] / z   # Waymo camera frame: x forward, y left, z up
        fu, fv, cu, cv_, k1, k2, p1, p2, k3 = c.intrinsic
        r2 = x * x + y * y; rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        mu = fu * (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cu
        mv = fv * (y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cv_
        ok = (p[:, 0] > 1e-3) & (r2 < 1.0) & (mu >= 0) & (mu <= c.width - 1) & (mv >= 0) & (mv <= c.height - 1)
        r2 = np.where(ok, r2, np.inf); win = r2 < best; best = np.minimum(best, r2)
        for k in proj:
            proj[k][2] &= ~win
        proj[c.name] = [mu, mv, win]
    _maps[key] = {k: (np.where(w, a, -1).reshape(H_F0, W_F0).astype(np.float32), np.where(w, b, -1).reshape(H_F0, W_F0).astype(np.float32))
                  for k, (a, b, w) in proj.items()}
    return _maps[key]


def stitch(fr):
    m = maps(fr.context.camera_calibrations); out = np.zeros((H_F0, W_F0, 3), np.uint16)
    for im in fr.images:
        if im.name in m:
            src = cv2.imdecode(np.frombuffer(im.image, np.uint8), cv2.IMREAD_COLOR)
            out += cv2.remap(src, *m[im.name], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.minimum(out, 255).astype(np.uint8)


def row(x, img):
    ps, fs = x.past_states, x.future_states
    v = float(np.hypot(ps.vel_x[-1], ps.vel_y[-1])); a = float(ps.accel_x[-1])
    past = [(ps.pos_x[i], ps.pos_y[i]) for i in (-7, -5, -3)]   # -1.5, -1.0, -0.5 s at 4 Hz
    cmd = CMD[e2e.EgoIntent.Intent.Name(x.intent)]
    q = (f"You are driving the ego vehicle; the image is its front camera. Speed {v:.1f} m/s, longitudinal acceleration {a:+.1f} m/s^2, "
         f"navigation command: {cmd}. Past positions over the last 1.5 s relative to now (x ahead, y left, metres): "
         + ", ".join(f"({px:.1f}, {py:.1f})" for px, py in past) + f". {FINAL}")
    name = x.frame.context.name
    return {"id": f"wod_{name}", "token": name, "segment": name.rsplit("-", 1)[0], "frame": int(name.rsplit("-", 1)[1]),
            "question": q, "images": [img], "speed": v, "accel": a, "command": cmd,
            "past16": [[float(px), float(py)] for px, py in zip(ps.pos_x, ps.pos_y)],
            "future": [[float(px), float(py)] for px, py in zip(fs.pos_x, fs.pos_y)],
            "prefs": [{"traj": [[float(px), float(py)] for px, py in zip(p.pos_x, p.pos_y)], "score": float(p.preference_score)}
                      for p in x.preference_trajectories if len(p.pos_x)]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sel", required=True); ap.add_argument("--out", required=True); ap.add_argument("shards", nargs="+")
    a = ap.parse_args(); every = int(a.sel.split(":")[1]) if a.sel.startswith("every:") else None
    os.makedirs(f"{a.out}/img", exist_ok=True); os.makedirs(f"{a.out}/meta", exist_ok=True)
    for sh in a.shards:
        base = os.path.basename(sh); n = k = 0
        with open(f"{a.out}/meta/{base}.jsonl.tmp", "w") as f:
            for d in records(sh):
                n += 1; x = e2e.E2EDFrame.FromString(d); name = x.frame.context.name
                keep = any(len(p.pos_x) for p in x.preference_trajectories) if every is None else int(name.rsplit("-", 1)[1]) % every == 0
                if not keep:
                    continue
                img = f"{a.out}/img/wod_{name}.jpg"
                if not os.path.exists(img):
                    ok, buf = cv2.imencode(".jpg", stitch(x.frame), [cv2.IMWRITE_JPEG_QUALITY, 92]); assert ok
                    open(img + ".part", "wb").write(buf.tobytes()); os.replace(img + ".part", img)
                f.write(json.dumps(row(x, img)) + "\n"); k += 1
        os.replace(f"{a.out}/meta/{base}.jsonl.tmp", f"{a.out}/meta/{base}.jsonl")
        open(f"{a.out}/meta/{base}.done", "w").write(f"{n} frames, {k} kept\n")
        print(base, n, "frames", k, "kept", flush=True)


if __name__ == "__main__":
    main()
