"""PhysicalAI-AV (PAI-AV) frames -> the policy's scene-row fields, shared by superglance.eval.pai_rows and pai_past. At clip
time t:
- image: the front wide 120-degree f-theta frame, re-projected into nuPlan's CAM_F0 (pinhole K = [[1545, 0, 960],
  [0, 1545, 560]] with its radial distortion and mounting rotation, as in every NAVSIM log); camera rotation from the clip's
  extrinsics, f-theta projection as AlpaSim's rectification.py (theta -> radius polynomial, [[c, d], [e, 1]] linear term);
- ego state from the offline egomotion (about 10 Hz), in NAVSIM's frame and units: history at -1.5 / -1.0 / -0.5 / 0 s and
  future at 0.5 ... 4.0 s in the ego frame at t; velocity and acceleration by central differences over +-0.2 s; command from
  the heading 4 s ahead (> 20 deg left, < -20 deg right, else straight).
- about 2.5% of the clips have no offline products at all (no egomotion.offline, no offline calibration), only the online
  ones; those clips use the online egomotion and calibration (ego_track, calib_row).
usage: python -m superglance.eval.pai_inputs --selftest"""
import functools, io, json, math, sys, zipfile
import numpy as np

K_F0 = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
D_F0 = np.array([-0.3561, 0.1725, -0.0021, 0.0005, -0.0523])   # nuPlan CAM_F0 (k1, k2, p1, p2, k3)
R_F0 = np.array([[0.0031, -0.0253, 0.9997], [-1.0, -0.0088, 0.0029], [0.0088, -0.9996, -0.0253]])   # CAM_F0 optical -> ego
W, H = 1920, 1080
CAM = "camera_front_wide_120fov"


def quat_R(qx, qy, qz, qw):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                     [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                     [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])


def ftheta_project(rays, p):
    """Camera-frame rays (N, 3; x right, y down, z forward) -> f-theta pixels (N, 2) and validity (rectification.py)."""
    xy, z = rays[:, :2], rays[:, 2]
    n = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(n, z)
    r = np.polynomial.polynomial.polyval(theta, np.asarray(p["angle_to_pixeldist_poly"], np.float64))
    s = np.divide(r, n, out=np.zeros_like(r), where=n > 1e-9)
    c, d, e = (p.get("linear_cde") or [1.0, 0.0, 0.0])[:3]
    px = (xy * s[:, None]) @ np.array([[c, d], [e, 1.0]]).T + np.asarray(p["principal_point"], np.float64)
    ok = (z > 0) & (theta <= p.get("max_angle", np.pi) + 1e-6) & (px[:, 0] >= 0) & (px[:, 0] <= W - 1) & (px[:, 1] >= 0) & (px[:, 1] <= H - 1)
    return px, ok


def remap_grid(p, R_cam):
    """For every CAM_F0 pixel the source pixel in the clip's f-theta image (cv2.remap maps; -1 where outside)."""
    import cv2
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    xn = cv2.undistortPoints(pts, K_F0, D_F0).reshape(-1, 2)
    rays_f0 = np.concatenate([xn, np.ones((len(xn), 1))], 1)
    rays = rays_f0 @ R_F0.T @ R_cam            # CAM_F0 optical -> ego/rig -> clip camera (R_cam: camera -> rig)
    px, ok = ftheta_project(rays, p)
    px[~ok] = -1
    return px[:, 0].reshape(H, W).astype(np.float32), px[:, 1].reshape(H, W).astype(np.float32)


def ego_samples(ts, P, yaw, t):
    """The scene rows' fields at t (seconds): poses in the ego frame at t, velocity / acceleration (x forward, y left)."""
    uy = np.unwrap(yaw)
    def pose(tt):
        return np.array([np.interp(tt, ts, P[:, 0]), np.interp(tt, ts, P[:, 1]), np.interp(tt, ts, uy)])
    p0 = pose(t)
    c, s = math.cos(p0[2]), math.sin(p0[2])
    def local(q, ref=p0, cs=(c, s)):
        dx, dy = q[0] - ref[0], q[1] - ref[1]
        return [cs[0] * dx + cs[1] * dy, -cs[1] * dx + cs[0] * dy, (q[2] - ref[2] + math.pi) % (2 * math.pi) - math.pi]
    def vel(tt):   # in the ego frame at tt
        q = pose(tt); cq, sq = math.cos(q[2]), math.sin(q[2])
        d = (pose(tt + 0.2)[:2] - pose(tt - 0.2)[:2]) / 0.4
        return np.array([cq * d[0] + sq * d[1], -sq * d[0] + cq * d[1]])
    def acc(tt):
        return (vel(tt + 0.2) - vel(tt - 0.2)) / 0.4
    hist_t = [t - 1.5, t - 1.0, t - 0.5, t]
    fut = [local(pose(t + 0.5 * k)) for k in range(1, 9)]
    h = math.degrees(fut[-1][2])
    cmd = [1, 0, 0, 0] if h > 20 else [0, 0, 1, 0] if h < -20 else [0, 1, 0, 0]
    return dict(history_poses=[local(pose(x)) for x in hist_t], future_poses=fut, ego_velocity=vel(t).tolist(), ego_acceleration=acc(t).tolist(),
                history_velocity=[vel(x).tolist() for x in hist_t], history_acceleration=[acc(x).tolist() for x in hist_t], driving_command=cmd)


def ego_track(root, chunk, clip, sources=("offline", "online")):
    """The clip's ego track: timestamps (s), xy (n, 2), yaw (n,) and which egomotion it came from: the offline one (about
    10 Hz) when the clip has it, else the online one (20-100 Hz; sorted, repeated timestamps dropped)."""
    import pandas as pd
    for src in sources:
        lab = "egomotion.offline" if src == "offline" else "egomotion"
        z = zipfile.ZipFile(f"{root}/labels/{lab}/{lab}.chunk_{chunk:04d}.zip")
        try:
            eg = pd.read_parquet(io.BytesIO(z.read(f"{clip}.{lab}.parquet")))
        except KeyError:
            continue
        if src == "online":
            eg = eg.sort_values("timestamp").drop_duplicates("timestamp")
        ts = eg["timestamp"].to_numpy(np.float64) / 1e6
        P = eg[["x", "y"]].to_numpy(np.float64)
        yaw = np.array([math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) for x, y, z, w in eg[["qx", "qy", "qz", "qw"]].to_numpy()])
        return ts, P, yaw, src
    raise KeyError(f"{clip}: no egomotion in {sources}")


@functools.lru_cache(8)
def calib_table(root, chunk, kind):
    import pandas as pd
    return pd.read_parquet(f"{root}/calibration/{kind}/{kind}.chunk_{chunk:04d}.parquet")


def ftheta_params(intr):
    """f-theta parameters of an intrinsics row. Offline tables: JSON in model_parameters. Online tables: columns cx, cy,
    fw_poly_0..4 (angle -> pixel distance), bw_poly_0..4 (pixel distance -> angle), converted to the same dict; max_angle is the
    backward polynomial at the farthest image corner, which is how the offline tables' value comes out (check_online)."""
    if "model_parameters" in intr.index:
        assert intr["model_type"] == "ftheta", intr["model_type"]
        return json.loads(intr["model_parameters"])
    w, h, cx, cy = (float(intr[k]) for k in ("width", "height", "cx", "cy"))
    bw = [float(intr[f"bw_poly_{i}"]) for i in range(5)]
    return dict(resolution=[int(w), int(h)], principal_point=[cx, cy], pixeldist_to_angle_poly=bw,
                angle_to_pixeldist_poly=[float(intr[f"fw_poly_{i}"]) for i in range(5)], max_angle=max_angle(bw, cx, cy, w, h))


def max_angle(bw, cx, cy, w, h):
    r = max(math.hypot(x - cx, y - cy) for x in (0.0, w - 1) for y in (0.0, h - 1))
    return float(np.polynomial.polynomial.polyval(r, np.asarray(bw, np.float64)))


def calib_row(root, chunk, clip, cam, kind, sources=("offline", "online")):
    """Row (clip, cam) of the offline calibration table <kind> (camera_intrinsics / sensor_extrinsics), else the online one."""
    for src in sources:
        t = calib_table(root, chunk, f"{kind}.offline" if src == "offline" else kind)
        if (clip, cam) in t.index:
            return t.loc[(clip, cam)]
    raise KeyError((clip, cam, kind))


def selftest():
    p = dict(angle_to_pixeldist_poly=[0.0, 925.47, -4.3969, -15.318, 1.2548, 0.0], principal_point=[960.8, 544.6], linear_cde=[1.0016, 0.0, 0.0], max_angle=1.228)
    px, ok = ftheta_project(np.array([[0.0, 0.0, 1.0], [math.sin(0.3), 0.0, math.cos(0.3)]]), p)
    assert ok.all() and np.allclose(px[0], [960.8, 544.6]) and abs(px[1, 0] - 960.8 - 1.0016 * np.polynomial.polynomial.polyval(0.3, p["angle_to_pixeldist_poly"])) < 1e-6
    # a car driving straight at 10 m/s: velocity (10, 0), acceleration 0, history -15 / -10 / -5 m, future 5 ... 40 m, straight
    ts = np.arange(0, 20.01, 0.1); P = np.stack([10 * ts, 0 * ts], 1)
    s = ego_samples(ts, P, np.zeros_like(ts), 8.0)
    assert np.allclose(s["ego_velocity"], [10, 0]) and np.allclose(s["ego_acceleration"], [0, 0], atol=1e-9)
    assert np.allclose(np.array(s["history_poses"])[:, 0], [-15, -10, -5, 0]) and np.allclose(np.array(s["future_poses"])[:, 0], 5 * np.arange(1, 9))
    assert s["driving_command"] == [0, 1, 0, 0]
    print("SELFTEST_OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest(); sys.exit()
