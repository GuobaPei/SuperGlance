"""PAI-AV human rows, step 2: pai_inputs' rows -> the pai_human shard, in 16 parts of the sorted chunk list, each built as
soon as its chunks are done.
  prompt  the policy prompt with the dropout on training rows (common.prompt);
  target  softmax(-ADE / 0.5 m) to the human's 4-s future (soft); bucket "event" / "plain" (common.bucket);
  splits  train = the train clips, val = the held-out clips (HELD2; never trained on, no dropout);
  fields  + ego_source (offline / online egomotion) + old_row (the token is a pai_human row of the earlier pool);
  images  pai/<clip>/<ms>.jpg (<SHARDS>/img/pai; <part>/pai links there).
Checks per part (report.json): on the earlier pool's tokens, the question equals strip(its question) byte for byte, and the input
fields equal the earlier fit rows'. OLD_ROWS (the earlier pool's train.jsonl) and OLD_FIT (its PAI-AV fit rows) feed only these
checks and old_row; an empty file (/dev/null) skips them.
Out: <SHARDS>/pai_human/pNN/{train,val}.jsonl, *_targets.npy, *_images.tsv, report.json; <SHARDS>/pai_human/report.json at the end.
usage: OLD_ROWS=<file> OLD_FIT=<file> python -m superglance.builders.shards.pai_rows [procs=2]   (waits for step 1; restarts skip
       built parts)"""
import json, os, shutil, sys, time
from multiprocessing import Pool

import numpy as np

from superglance.builders.shards.common import O, bucket, cands, dropped, prompt, row, strip, write   # noqa: E402

NP, IN = 16, f"{O}/pai_in"
FIT = ("timestamp_us", "history_poses", "future_poses", "ego_velocity", "ego_acceleration", "history_velocity", "history_acceleration",
       "driving_command", "frame_offset_ms")
OLDQ, OLDF = {}, {}


def parts():
    ch = sorted({int(l.split()[1]) for l in open(f"{IN}/clips.tsv")})
    return [ch[len(ch) * p // NP: len(ch) * (p + 1) // NP] for p in range(NP)]


def build(job):
    p, chunks = job
    d = f"{O}/pai_human/p{p:02d}"; tmp = d + ".tmp"; shutil.rmtree(tmp, ignore_errors=True)
    rows = [json.loads(l) for c in chunks for l in open(f"{IN}/inputs/{c:04d}.jsonl")]
    rep, C = {"chunks": [chunks[0], chunks[-1], len(chunks)], "clips": len({r["log_name"] for r in rows})}, cands()
    for split in ("train", "val"):
        sel = [r for r in rows if r["part"] == split]
        rel = {r["token"]: f"pai/{r['log_name']}/{r['token'].rsplit('_', 1)[1]}.jpg" for r in sel}
        out = [row("pai_human", r["token"], prompt(r, split == "train"), rel[r["token"]], sampling_bucket=bucket(r), label_source="e2e_softmax_minus_ade_0p5m",
                   dropout=bool(dropped(r["token"])) and split == "train", ego_source=r["ego_source"], old_row=r["token"] in OLDQ) for r in sel]
        imgs = {rel[r["token"]]: (r["cam_f0"], r["img_bytes"]) for r in sel}
        rep[split] = write(tmp, split, out, [np.asarray(r["future_poses"], np.float32)[:, :2] for r in sel], 0.5, imgs, C)
        old = [(r, x) for r, x in zip(sel, out) if r["token"] in OLDQ]
        fit = [r for r in sel if r["token"] in OLDF]
        rep[split].update(clips=len({r["log_name"] for r in sel}), online_rows=sum(r["ego_source"] == "online" for r in sel),
                          event_rows=sum(x["sampling_bucket"] == "event" for x in out), dropout_rows=sum(x["dropout"] for x in out),
                          old_rows=len(old), old_question_equal=sum(x["question"] == OLDQ[r["token"]] for r, x in old),
                          old_fit_rows=len(fit), old_fit_fields_equal=sum(all(r[k] == OLDF[r["token"]][k] for k in FIT) for r in fit))
    os.makedirs(tmp, exist_ok=True)
    os.symlink(os.path.relpath(f"{O}/img/pai", tmp), f"{tmp}/pai")
    json.dump(rep, open(f"{tmp}/report.json", "w"), indent=1)
    os.rename(tmp, d)
    return p, rep


def main(procs=2):
    for l in open(os.environ["OLD_ROWS"]):   # the earlier pool's pai_human rows' prompts, their block cut out
        if '"source": "pai_human"' in l:
            r = json.loads(l); OLDQ[r["token"]] = strip(r["question"])
    for l in open(os.environ["OLD_FIT"]):
        r = json.loads(l); OLDF[r["token"]] = {k: r[k] for k in FIT}
    P = parts(); os.makedirs(f"{O}/pai_human", exist_ok=True); done = {}
    print(f"{len(OLDQ)} old pai_human rows, {len(OLDF)} old fit rows; {NP} parts of {[len(x) for x in P]} chunks", flush=True)
    with Pool(int(procs)) as pool:
        run = {}
        while len(done) < NP:
            for p, ch in enumerate(P):
                if os.path.exists(f"{O}/pai_human/p{p:02d}/report.json"):
                    done.setdefault(p, json.load(open(f"{O}/pai_human/p{p:02d}/report.json")))
                elif p not in run and all(os.path.exists(f"{IN}/inputs/{c:04d}.jsonl") for c in ch):
                    run[p] = pool.apply_async(build, ((p, ch),))
            for p in [p for p, a in run.items() if a.ready()]:
                done[p] = run.pop(p).get()[1]
                print(f"{time.strftime('%H:%M:%S')} part {p:02d}: {json.dumps(done[p])}", flush=True)
            time.sleep(30)
    tot = {s: {k: sum(done[p][s][k] for p in done) for k in done[0][s] if k != "target_sum_abs_err_max"} for s in ("train", "val")}
    for s in tot:
        tot[s]["target_sum_abs_err_max"] = max(done[p][s]["target_sum_abs_err_max"] for p in done)
    rep = {"parts": NP, "tau_m": 0.5, "block": False, "clips": json.load(open(f"{IN}/clips.json")), **tot}
    json.dump(rep, open(f"{O}/pai_human/report.json", "w"), indent=1); print("DONE " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
