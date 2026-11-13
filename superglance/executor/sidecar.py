"""NAVSIM's plan execution for the AlpaSim driver. The model is trained and scored on plans executed by NAVSIM's PDM
simulator (LQR on speed with a 1 s horizon + kinematic bicycle, 40 x 0.1 s, from the logged ego state); AlpaSim's
controller chases the plan's positions 1-2 s ahead instead and drives the same plan harder. This sidecar runs the PDM
simulator on the chosen plan from the car's current state, so the driver can hand AlpaSim the trajectory NAVSIM would
have driven. superglance/executor/stitched.py, the process the driver starts, adds stitching on top of it.
JSON lines on stdin: {"plan": 8 x [x, y, heading] (rig frame at now), "v": [vx, vy], "a": [ax, ay], "yaw_rate": w};
on stdout: {"poses": 40 x [x, y, heading] at 0.1 ... 4.0 s, rig frame at now}.
usage: (navsim env) python -m superglance.executor.sidecar                         (server)
       python -m superglance.executor.sidecar --selftest <metric_cache.pkl> [...]  (equals the simulation from the cached ego state)"""
import json, sys
OUT, sys.stdout = sys.stdout, sys.stderr   # only results on the pipe
import numpy as np

from superglance.executor import pdm as pb
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex as S

VP = get_pacifica_parameters()   # what every NAVSIM metric cache uses


def execute(plan, v, a, steer, t_us=10 ** 12):
    """The PDM simulation of plan from an ego at the origin with this rear-axle velocity / acceleration / tire steering."""
    ego = EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(0.0, 0.0, 0.0), rear_axle_velocity_2d=StateVector2D(*v),
                                        rear_axle_acceleration_2d=StateVector2D(*a), tire_steering_angle=steer,
                                        time_point=TimePoint(t_us), vehicle_parameters=VP)
    return pb._G["sim"].simulate_proposals(pb.to_states(plan, ego)[None], ego)[0][1:, [S.X, S.Y, S.HEADING]]



def steer_from_yaw_rate(v, w):
    sp = float(np.hypot(*v))
    return float(np.arctan(VP.wheel_base * w / sp)) if sp > 0.5 else 0.0


def selftest(paths):
    for p in paths:
        mc = pb.load_cache(p)
        ego = mc.ego_state
        d = ego.dynamic_car_state
        plan = np.array([[1.4 * k, 0.05 * k * k, 0.02 * k] for k in range(1, 9)])
        g = pb._G["sim"].simulate_proposals(pb.to_states(plan, ego)[None], ego)[0][1:]
        x0, y0, h0 = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        c, s = np.cos(h0), np.sin(h0)
        ref = np.stack([c * (g[:, S.X] - x0) + s * (g[:, S.Y] - y0), -s * (g[:, S.X] - x0) + c * (g[:, S.Y] - y0),
                        (g[:, S.HEADING] - h0 + np.pi) % (2 * np.pi) - np.pi], 1)
        got = execute(plan, (d.rear_axle_velocity_2d.x, d.rear_axle_velocity_2d.y), (d.rear_axle_acceleration_2d.x, d.rear_axle_acceleration_2d.y),
                      ego.tire_steering_angle)
        err = np.abs(got - ref).max()
        assert err < 1e-4, (p, err)   # metres / radians; the cached run works in global coordinates (~1e5 m)
    print("SELFTEST_OK", len(paths), "scenes")


if __name__ == "__main__":
    pb.init()
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest(sys.argv[2:])
        sys.exit()
    for line in sys.stdin:
        r = json.loads(line)
        out = execute(r["plan"], r["v"], r["a"], steer_from_yaw_rate(r["v"], r.get("yaw_rate", 0.0)))
        OUT.write(json.dumps({"poses": np.round(out, 5).tolist()}) + "\n")
        OUT.flush()
