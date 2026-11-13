"""Batched NAVSIM v1.1 PDMS with the exact semantics of navsim evaluate/pdm_score.pdm_score().

Each candidate is scored as the pair (PDM-Closed, candidate): PDMSimulator (LQR + kinematic bicycle, 40 x 0.1 s) +
v1.1 PDMScorer (default_scoring_parameters: EP 5, TTC 5, C 2, DDC 0; progress threshold 5 m). All sub-metrics are
per proposal; the only cross-proposal term, EP normalisation, is done per pair:
    p = raw_progress * NC * DAC;  EP = p_c / max(p_pdm, p_c) if max > 5 m else (1 if NC*DAC > 0 else 0)
    PDMS = NC * DAC * (5 EP + 5 TTC + 2 C) / 12
Sub-score order everywhere: [NC, DAC, EP, TTC, C, DDC, PDMS]."""
import numpy as np

SCORE_NAMES = ["NC", "DAC", "EP", "TTC", "C", "DDC", "PDMS"]
_G = {}


def init():
    if _G:
        return
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    _G.update(ps=ps, sim=PDMSimulator(ps), cfg=PDMScorerConfig(), scorer=PDMScorer(ps, PDMScorerConfig()),
              vsim=PDMSimulator(ps), vscorer=PDMScorer(ps, PDMScorerConfig()))


def load_cache(path):
    import lzma, pickle
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def to_states(poses, ego):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    return get_trajectory_as_array(transform_trajectory(Trajectory(np.asarray(poses, dtype=np.float64)), ego), _G["ps"], ego.time_point)


def score_pairs(mc, cand, chunk=1100):
    """cand (K, 8, 3) ego-frame poses at 2 Hz -> (K, 7) scores, number of non-finite simulated proposals."""
    init()
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    ps, sim, sc, cfg = _G["ps"], _G["sim"], _G["scorer"], _G["cfg"]
    ego = mc.ego_state
    pdm_st = get_trajectory_as_array(mc.trajectory, ps, ego.time_point)
    w = cfg.weighted_metrics_array                    # [EP, TTC, C, DDC]
    out = np.zeros((len(cand), 7)); nbad = 0
    for s in range(0, len(cand), chunk):
        st = np.stack([pdm_st] + [to_states(p, ego) for p in cand[s:s + chunk]])
        simst = sim.simulate_proposals(st, ego)
        bad = ~np.isfinite(simst).reshape(len(simst), -1).all(1)
        if bad.any():
            simst = np.nan_to_num(simst, nan=0.0, posinf=1e4, neginf=-1e4); nbad += int(bad[1:].sum())
        sc.score_proposals(simst, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
        m = sc._multi_metrics.prod(0)
        raw = sc._progress_raw * m
        mx = np.maximum(raw[1:], raw[0])
        big = mx > cfg.progress_distance_threshold
        ep = np.where(big, raw[1:] / np.where(big, mx, 1.0), np.where(m[1:] == 0.0, 0.0, 1.0))
        wm = sc._weighted_metrics[:, 1:].copy(); wm[0] = ep
        pdms = m[1:] * (wm * w[:, None]).sum(0) / w.sum()
        out[s:s + chunk] = np.stack([sc._multi_metrics[0, 1:], sc._multi_metrics[1, 1:], ep, wm[1], wm[2], wm[3], pdms], 1)
    return out, nbad


def verify(mc, cand, out, idxs):
    """max |stock pdm_score() - batched| over the given candidate indices and all 7 sub-scores."""
    init()
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.common.dataclasses import Trajectory
    md = 0.0
    for j in idxs:
        r = pdm_score(mc, Trajectory(np.asarray(cand[j], dtype=np.float64)), _G["ps"], _G["vsim"], _G["vscorer"])
        ref = np.array([r.no_at_fault_collisions, r.drivable_area_compliance, r.ego_progress, r.time_to_collision_within_bound,
                        r.comfort, r.driving_direction_compliance, r.score], dtype=np.float64)
        md = max(md, float(np.abs(ref - out[j]).max()))
    return md
