"""At-fault collisions of an AlpaSim run by kind, read from its summary and rollout logs (no re-simulation). Per scenario
with an at-fault collision: AlpaSim's own side flags (collision_front / _lateral / _rear) and, for front hits, the actor hit:
the first actor centre in the 5.2 x 4 m box ahead of the ego, else in a 10 x 5.2 m box around the event time; its gap and
lateral offset, its speed (0.5 s differences) and the ego's speed / a_x over the 6 s before contact. stopped = the actor's
speed 2 s before contact <= 0.5 m/s; decel = it was moving 2 s before and lost >= 1 m/s since; its braking onset = its last
speed peak in the 6 s (onset_s = onset to contact); ego_brake_s = the first ego a_x <= -1 m/s^2 after that onset, to contact
(None: no braking before contact). The closing line has the appendix's counts: at-fault scenarios = the rows, front hits on
a stopped actor (stopped true) and on a moving one (stopped false, decelerating ones included).
usage (AlpaSim's environment): python -m superglance.alpasim.collisions <results-summary.clean.json>
  "<run dir>/rollouts/{c}/*/rollout.asl" <out.jsonl>   ({c} = the scenario's clipgt_id)
"""
import asyncio, glob, json, sys
import numpy as np
from alpasim_utils.logs import async_read_pb_log


def yaw(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def speeds(track):
    ts = np.array(sorted(track)); P = np.array([track[t][:2] for t in ts]); v = np.full(len(ts), np.nan)
    for i, t in enumerate(ts):
        j, k = np.searchsorted(ts, t - 250_000), min(np.searchsorted(ts, t + 250_000), len(ts) - 1)
        if ts[k] > ts[j]: v[i] = np.hypot(*(P[k] - P[j])) / ((ts[k] - ts[j]) / 1e6)
    return ts, v


async def one(asl, dur):
    ego, act = {}, {}
    async for e in async_read_pb_log(asl):
        k = e.WhichOneof("log_entry")
        if k == "controller_return":
            for s in e.controller_return.states:
                p = s.pose_local_to_rig; ego[s.timestamp_us] = (p.vec.x, p.vec.y, yaw(p.quat), s.dynamic_state.linear_velocity.x, s.dynamic_state.linear_acceleration.x)
        elif k == "actor_poses":
            for a in e.actor_poses.actor_poses:
                if a.actor_id != "EGO":
                    act.setdefault(a.actor_id, {})[e.actor_poses.timestamp_us] = (a.actor_pose.vec.x, a.actor_pose.vec.y)
    te = np.array(sorted(ego)); E = np.array([ego[t] for t in te]); at = lambda t: E[int(np.argmin(np.abs(te - t)))]
    A = {a: (np.array(sorted(tr)), tr) for a, tr in act.items()}
    def rel(t, a):
        ts, tr = A[a]; i = int(np.argmin(np.abs(ts - t)))
        if abs(ts[i] - t) > 150_000: return None
        x, y = tr[ts[i]]; ex, ey, eh = at(t)[:3]; dx, dy = x - ex, y - ey
        return np.cos(eh) * dx + np.sin(eh) * dy, -np.sin(eh) * dx + np.cos(eh) * dy
    def first(times, bx, by):
        for t in times:
            for a in A:
                r = rel(t, a)
                if r is not None and 0 < r[0] < bx and abs(r[1]) < by: return a, int(t)
        return None, None
    lead, tc = first(te, 5.2, 2.0)
    if lead is None:
        tq = te[0] + (dur * 20 + 1.7) * 1e6; lead, tc = first(te[(te > tq - 1.5e6) & (te < tq + 1.5e6)], 10, 2.6)
    if lead is None: return {"lead": None}
    ts, v = speeds(A[lead][1]); vl = lambda t: None if np.min(np.abs(ts - t)) > 300_000 else float(v[int(np.argmin(np.abs(ts - t)))])
    w = (ts >= tc - 6e6) & (ts <= tc); out = {"lead": lead, "t_coll": round((tc - te[0]) / 1e6, 1)}
    v0, v2 = vl(tc), vl(tc - 2e6)
    out.update(v_lead=v0, v_lead_2s=v2, decel=bool(v2 is not None and v0 is not None and v2 > 0.5 and v2 - v0 >= 1.0),
               stopped=bool(v2 is not None and v2 <= 0.5))
    if w.any() and np.isfinite(v[w]).any():
        i = int(np.nanargmax(np.where(w, v, -np.inf))); tp = ts[i]
        out.update(onset_s=round((tc - tp) / 1e6, 1), drop=round(float(v[i] - (v0 or 0)), 1))
        b = [t for t in te if tp <= t <= tc and at(t)[4] <= -1.0]
        out["ego_brake_s"] = round((tc - b[0]) / 1e6, 1) if b else None
    h = []
    for back in (6, 4, 3, 2, 1, 0.5, 0):
        t = tc - int(back * 1e6); r = rel(t, lead); e_ = at(t)
        h.append(dict(dt=-back, ahead=None if r is None else round(float(r[0]), 1), lat=None if r is None else round(float(r[1]), 1),
                      v_lead=None if vl(t) is None else round(vl(t), 1), v_ego=round(float(e_[3]), 1), a_ego=round(float(e_[4]), 2)))
    out["hist"] = h
    return out


async def main(summary, pat, out):
    R = json.load(open(summary))["rollouts"]
    Z = [(r["clipgt_id"], r["metrics"]) for r in R if (r["metrics"].get("collision_at_fault") or 0) > 0]
    with open(out, "w") as fo:
        for c, m in Z:
            side = {k: bool((m.get(f"collision_{k}") or 0) > 0) for k in ("front", "lateral", "rear")}
            fs = sorted(glob.glob(pat.format(c=c)))
            r = (await one(fs[0], m.get("duration_frac_20s") or 0)) if fs and side["front"] else {} if fs else {"err": "no asl"}
            fo.write(json.dumps({"c": c, **side, **r}) + "\n"); fo.flush()
    K = [json.loads(line) for line in open(out)]; f = [r for r in K if r.get("front")]
    print(f"at-fault scenarios {len(K)} | front {len(f)}: stopped {sum(r.get('stopped') is True for r in f)}, moving "
          f"{sum(r.get('stopped') is False for r in f)}, no actor {sum('stopped' not in r for r in f)} | no log {sum('err' in r for r in K)}")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:4]))
