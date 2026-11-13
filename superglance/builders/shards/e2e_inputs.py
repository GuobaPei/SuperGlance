"""e2e_inputs.npz: the labels the rollout and RL pools read, in one file. pai_tok / pai_F: the PAI-AV human rows' tokens and
logged 4-s futures (from the PAI-AV fit rows); dag_tok / dag_E / dag_scene and held_tok / held_E: the rollout rows' tokens, recovery
labels (x, y of the 8 poses) and scenes, training and held-out, from the recovery-label build's meta files (pai_meta.npz,
dag_meta.npz, dagheld_meta.npz, labels "labels_v4c"); s916: the 916 AlpaSim scenes.
usage: RR_META=<dir of the meta files> PAI_FIT_ROWS=<file> S916=<list> python -m superglance.builders.shards.e2e_inputs <out.npz>"""
import json, os, sys
import numpy as np


def export(out):
    S = os.environ["RR_META"]
    ptok = [str(t) for t in np.load(f"{S}/pai_meta.npz")["tok"]]; want, F = set(ptok), {}
    for l in open(os.environ["PAI_FIT_ROWS"]):   # the PAI-AV fit rows
        r = json.loads(l)
        if r["token"] in want:
            F[r["token"]] = np.asarray(r["future_poses"], np.float32)[:, :2]
    assert len(F) == len(ptok)
    dm, hm = np.load(f"{S}/dag_meta.npz"), np.load(f"{S}/dagheld_meta.npz")
    assert str(dm["lab"]) == str(hm["lab"]) == "labels_v4c", (dm["lab"], hm["lab"])
    s916 = sorted({l.strip() for l in open(os.environ["S916"]) if l.strip()}); assert len(s916) == 916
    np.savez(out, pai_tok=np.array(ptok), pai_F=np.stack([F[t] for t in ptok]), dag_tok=dm["tok"], dag_E=dm["E"][:, :, :2],
             dag_scene=dm["scene"], held_tok=hm["tok"], held_E=hm["E"][:, :, :2], s916=np.array(s916))
    sc = set(map(str, dm["scene"]))
    print(json.dumps({"pai": len(ptok), "dag": len(dm["tok"]), "held": len(hm["tok"]), "dag_scenes": len(sc), "dag_scenes_in_916": len(sc & set(s916)),
                      "dag_rows_in_916": int(np.isin(dm["scene"], s916).sum())}))


if __name__ == "__main__":
    export(sys.argv[1])
