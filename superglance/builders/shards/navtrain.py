"""navtrain shard: every scene of the navtrain export as a policy row (the prompt of superglance.data.scene.c256_row, no
road-users block, no dropout), target softmax(-ADE / 1 m) to the logged 4-s future (the imitation target).
  train  the 92,339 "fit" scenes;   val  the 10,949 "heldout" scenes (never trained on);
  dev    the 200 navtrain dev rows of the earlier pool (<old pool>/dev.jsonl, its source "navtrain" rows), a subset of val.
Images: navtrain/<log>/<file>.jpg = <OPENSCENE_DATA_ROOT>/sensor_blobs/trainval/<log>/CAM_F0/<file>.jpg (<SHARDS>/img/navtrain/<log>
links to that CAM_F0). superglance.builders.shards.navtrain_pdms swaps the targets for the PDMS soft targets.
usage: python -m superglance.builders.shards.navtrain <navtrain_inputs.jsonl> <old pool dir>"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from superglance.builders.shards.common import O, cands, link, prompt, row, write   # noqa: E402


def main(nav, old):
    D = f"{O}/navtrain"; os.makedirs(D)   # never overwrite
    C, rows = cands(), [json.loads(l) for l in open(nav)]
    rel = {}
    for r in rows:
        log, cam, f = r["cam_f0"].split("/")[-3:]
        assert cam == "CAM_F0" and log == r["log_name"], r["cam_f0"]
        rel[r["token"]] = f"navtrain/{log}/{f}"
    with ThreadPoolExecutor(32) as ex:
        size = list(ex.map(lambda r: os.path.getsize(r["cam_f0"]), rows))
    imgs = {rel[r["token"]]: (r["cam_f0"], n) for r, n in zip(rows, size)}
    link(D, "navtrain")
    for log, d in {r["log_name"]: os.path.dirname(r["cam_f0"]) for r in rows}.items():
        if not os.path.islink(f"{O}/img/navtrain/{log}"):
            os.symlink(d, f"{O}/img/navtrain/{log}")
    by = {r["token"]: r for r in rows}
    dev = [json.loads(l)["token"] for l in open(f"{old}/dev.jsonl") if '"source": "navtrain"' in l]
    rep = {"from": nav, "tau_m": 1.0, "target": "e2e_pool.soft, softmax(-ADE / 1 m) to future_poses", "subsets": {s: sum(r["subset"] == s for r in rows) for s in ("fit", "heldout")}}
    for split, sel in (("train", [r for r in rows if r["subset"] == "fit"]), ("val", [r for r in rows if r["subset"] == "heldout"]), ("dev", [by[t] for t in dev])):
        out = [row("navtrain", r["token"], prompt(r, False), rel[r["token"]], label_source="softmax_minus_ade_1m") for r in sel]
        rep[split] = write(D, split, out, [np.asarray(r["future_poses"], np.float32)[:, :2] for r in sel], 1.0, imgs, C)
    rep["dev_in_val"] = len(set(dev) & {r["token"] for r in rows if r["subset"] == "heldout"})
    json.dump(rep, open(f"{D}/report.json", "w"), indent=1); print(json.dumps(rep))


if __name__ == "__main__":
    main(*sys.argv[1:3])
