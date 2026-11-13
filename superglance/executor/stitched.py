"""The executor the AlpaSim driver starts: superglance/executor/sidecar.py with trajectory stitching.
sidecar.py starts every replan's PDM simulation from the car's tire steering as estimated from its yaw rate
(steer_from_yaw_rate), so each executed plan reaches a turn's curvature about 1 s after the call. With a fresh plan every
100 ms the tightening is put off again and again, and the car runs wide in turns.
Stitching, as AV stacks continue a new trajectory from the previous one: a request may carry "steer", the previous executed
plan's tire steering at now (the driver keeps it per session). The simulation starts from it unless it is more than
STITCH_MAX from the yaw-rate estimate (then from the estimate, as sidecar.py); position, heading, velocity and acceleration
still come from the car. The reply adds "steer_at" (the executed tire steering at 0.0 ... 4.0 s, for the next call) and
"stitched". Without "steer" the poses are sidecar.py's.
usage: started by the driver; python -m superglance.executor.stitched --check   (navsim env)"""
import json, sys
import numpy as np
from superglance.executor import sidecar as ne   # its import sends print() to stderr and keeps the pipe as ne.OUT
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex as S

STITCH_MAX = 0.2   # rad of tire steering; a reset guard only, not tuned (turn-entry gaps are ~0.01 rad)


def run(plan, v, a, steer, t_us=10 ** 12):
    """sidecar.execute's simulation, all 41 PDM states (0.0 ... 4.0 s; the first is the start state)."""
    ego = ne.EgoState.build_from_rear_axle(rear_axle_pose=ne.StateSE2(0.0, 0.0, 0.0), rear_axle_velocity_2d=ne.StateVector2D(*v),
                                           rear_axle_acceleration_2d=ne.StateVector2D(*a), tire_steering_angle=steer,
                                           time_point=ne.TimePoint(t_us), vehicle_parameters=ne.VP)
    return ne.pb._G["sim"].simulate_proposals(ne.pb.to_states(plan, ego)[None], ego)[0]


def answer(r):
    s0 = ne.steer_from_yaw_rate(r["v"], r.get("yaw_rate", 0.0))
    st = r.get("steer")
    ok = st is not None and abs(st - s0) <= STITCH_MAX
    x = run(r["plan"], r["v"], r["a"], st if ok else s0)
    return {"poses": np.round(x[1:, [S.X, S.Y, S.HEADING]], 5).tolist(), "steer_at": np.round(x[:, S.STEERING_ANGLE], 6).tolist(),
            "stitched": ok}


def check():
    plan = [[1.4 * k, 0.05 * k * k, 0.02 * k] for k in range(1, 9)]
    r = {"plan": plan, "v": [7.0, 0.0], "a": [0.3, 0.0], "yaw_rate": 0.05}
    s0 = ne.steer_from_yaw_rate(r["v"], r["yaw_rate"])
    o = answer(r)
    assert not o["stitched"] and np.abs(np.asarray(o["poses"]) - ne.execute(plan, r["v"], r["a"], s0)).max() < 1e-4   # no "steer": sidecar.py
    assert abs(o["steer_at"][0] - s0) < 1e-6 and len(o["steer_at"]) == 41
    o2 = answer({**r, "steer": o["steer_at"][1]})
    assert o2["stitched"] and abs(o2["steer_at"][0] - o["steer_at"][1]) < 1e-6   # starts where the last plan was at +0.1 s
    assert not answer({**r, "steer": s0 + 0.5})["stitched"]   # past the guard: the estimate
    print("CHECK_OK", round(s0, 5), round(o["steer_at"][1], 5), round(o["steer_at"][10], 5))


if __name__ == "__main__":
    ne.pb.init()
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check()
        sys.exit()
    for line in sys.stdin:
        ne.OUT.write(json.dumps(answer(json.loads(line))) + "\n")
        ne.OUT.flush()
