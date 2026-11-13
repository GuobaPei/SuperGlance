"""OpenScene shard: openscene_export's osx_inputs.jsonl -> navtrain_os/train.jsonl + targets, the navtrain shard's recipe (the
policy prompt, no block, no dropout; target softmax(-ADE / 1 m) to the logged 4-s future). Rows keep source "navtrain", so they join
that source's mixture weight; origin "openscene_extra" marks them. Asserts, all rows:
  control  the 200 control tokens re-exported by openscene_export equal navtrain_inputs.jsonl on every field below;
  leaks    no osx token is a navtrain scene (fit or heldout), no osx log is a held-out log;
  prompts  0 "Road users ahead"; every question's template (digits masked) is one of the navtrain rows' templates (frames with the
           "unknown" command, which navtrain never has, are dropped first);
  images   every CAM_F0 jpg exists and is non-empty, under the img/navtrain/<log> links the navtrain shard made.
With <index> (osall, not in the paper's pools): every 2nd frame of each log, source "openscene", images from camera tars.
usage: NAV_INPUTS=<navtrain_inputs.jsonl> python -m superglance.builders.shards.openscene <osx_inputs.jsonl> [<index>]"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from superglance.builders.shards.common import O, cands, prompt, row, write   # noqa: E402

NAV = os.environ["NAV_INPUTS"]
FIELDS = ("log_name", "map", "timestamp_us", "cam_f0", "ego_velocity", "ego_acceleration", "driving_command", "history_poses",
          "history_velocity", "history_acceleration", "future_poses")


def tmpl(q):
    return re.sub(r"[-+]?\d+(\.\d+)?", "#", q)


def main(osx, index=None):
    D = f"{O}/navtrain_os{'all' if index else ''}"; os.makedirs(D)   # never overwrite
    nav = {}
    for l in open(NAV):
        r = json.loads(l); nav[r["token"]] = r
    held = {r["log_name"] for r in nav.values() if r["subset"] == "heldout"}
    rows = [json.loads(l) for l in open(osx)]
    ctrl, ext = [r for r in rows if r["subset"] == "control"], [r for r in rows if r["subset"] in ("osx", "osall")]
    bad = [(r["token"], k) for r in ctrl for k in FIELDS if r[k] != nav[r["token"]][k]]
    assert len(ctrl) == 200 and not bad, (len(ctrl), bad[:5])
    assert not any(r["token"] in nav for r in ext), "osx token is a navtrain scene"
    assert not any(r["log_name"] in held for r in ext), "osx frame from a heldout log"
    if index:   # osall reaches logs outside navtrain: none may be a navtest log
        nt = set(re.findall(r"\d{4}(?:\.\d\d){5}_veh-\d+_\d+_\d+", open(f"{os.environ['NAVSIM_DEVKIT_ROOT']}/planning/script/config/common/train_test_split/scene_filter/navtest.yaml").read()))
        assert len(nt) == 136 and not {r["log_name"] for r in ext} & nt, "osall frame from a navtest log"
    unk = sum(r["driving_command"] == [0, 0, 0, 1] for r in ext)   # navtrain has no "unknown" command; 331 osx frames do, dropped
    ext = [r for r in ext if r["driving_command"] != [0, 0, 0, 1]]
    n_all = len(ext)
    if index:   # §86q: frame k of a log = (t - its first exported frame's t) / 0.5 s; even k only
        t0 = {}
        for r in ext:
            t0[r["log_name"]] = min(t0.get(r["log_name"], r["timestamp_us"]), r["timestamp_us"])
        ext = [r for r in ext if round((r["timestamp_us"] - t0[r["log_name"]]) / 5e5) % 2 == 0]
    rel = {}
    for r in ext:
        log, cam, f = r["cam_f0"].split("/")[-3:]
        assert cam == "CAM_F0" and log == r["log_name"], r["cam_f0"]
        assert index or os.path.islink(f"{O}/img/navtrain/{log}"), log
        rel[r["token"]] = f"navtrain/{log}/{f}"
    if index:   # sensor_blobs/trainval/<log>/CAM_F0/<file> in tar k -> the source "cam_<k>.tar:<member>" and its size
        idx = {}
        for l in open(index):
            k, m, n = l.rstrip("\n").split("\t"); idx[m.split("/", 2)[2]] = (f"cam_{k}.tar:{m}", int(n))
        src = [idx["/".join(r["cam_f0"].split("/")[-3:])] for r in ext]; size = [n for _, n in src]
    else:
        with ThreadPoolExecutor(32) as ex:
            size = list(ex.map(lambda r: os.path.getsize(r["cam_f0"]), ext))
        src = [(r["cam_f0"], n) for r, n in zip(ext, size)]
    assert min(size) > 0
    known = {tmpl(prompt(r, False)) for r in list(nav.values())[::20]}
    out = []
    for r in ext:
        q = prompt(r, False)
        assert "Road users ahead" not in q and tmpl(q) in known, q[-300:]
        out.append(row("openscene" if index else "navtrain", r["token"], q, rel[r["token"]], label_source="softmax_minus_ade_1m", origin="openscene_extra"))
    imgs = {rel[r["token"]]: sn for r, sn in zip(ext, src)}
    F = np.asarray([r["future_poses"] for r in ext], np.float64)   # [N, 8, 3]: x, y, heading at 0.5-4 s, ego frame
    seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(F), 1, 2)), F[:, :, :2]], 1), axis=1), axis=-1)
    dv = np.abs(seg[:, -1] / 0.5 - np.asarray([r["ego_velocity"][0] for r in ext]))
    cut = {"path_gt_2m": int((seg.sum(1) > 2).sum()), "yaw_gt_5deg": int((np.abs(F[:, -1, 2]) > np.radians(5)).sum()), "dspeed_gt_1": int((dv > 1).sum())}
    cut["any"] = int(((seg.sum(1) > 2) | (np.abs(F[:, -1, 2]) > np.radians(5)) | (dv > 1)).sum())
    rep = {"from": osx, "index": index, "frames_before_every_2nd": n_all, "frames": len(ext), "nontrivial_cut_report_only": cut, "control": {"rows": len(ctrl), "fields_equal": len(ctrl) * len(FIELDS)}, "dropped_unknown_command": unk,
           "logs": len({r["log_name"] for r in ext}),
           "navtrain_templates": len(known), "train": write(D, "train", out, [np.asarray(r["future_poses"], np.float32)[:, :2] for r in ext], 1.0, imgs, cands())}
    json.dump(rep, open(f"{D}/report.json", "w"), indent=1); print(json.dumps(rep))


if __name__ == "__main__":
    main(*sys.argv[1:3])
