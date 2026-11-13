"""Stage-3 pool with the front-camera history: the pool's rows with past frames added; nothing else changes.
A row gets history when its frame has all three past frames:
  navtrain (fit, held-out dev and the OpenScene rows) and L2D if_navtrain_*: by scene token, history_nav.py's index.jsonl;
  pai_human and L2D if_pai_*: by token <clip>_<ms>, the PAI history tars (members pai/<clip>/<ms>_<k>.jpg, k = 0, 1, 2), and only
  for clips with crc32(clip) % 2 == 0.
Such a row's images become [current, hist/<past 1.5 s>, hist/<1.0 s>, hist/<0.5 s>] and it gets history: true (the collator then
lays it out with frame tags). Every other row is written byte for byte as it was (asserted on the output). Row order is kept, so
the pool's targets (compact row indices) are reused: <out>/<split>_targets.npy are symlinks to the pool's.
<hist_nav>_fill, when there, gives the tokens hist_nav lacks (same rule, JPEG and layout): used only where hist_nav has no frames,
never replacing one (asserted), and every frame path in both tar sets must have the same bytes (asserted); report.json counts the
rows whose history came from the fill per source (history_from_fill).
Out: <out>/{train,dev}.jsonl, the target links, report.json (per source: rows, with history, and why not).
usage: python -m superglance.builders.pools.history_rows <pool dir> <hist_nav dir> <hist_pai dir> <out dir>"""
import json, os, sys, tarfile, zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

NAV = ("navtrain", "if_navtrain_")
PAI = ("pai_human", "if_pai_")


SEL = lambda clip: zlib.crc32(clip.encode()) % 2 == 0


def kind(src):
    return "nav" if src == NAV[0] or src.startswith(NAV[1]) else "pai" if src == PAI[0] or src.startswith(PAI[1]) else None


def pai_index(d):
    """{token: [3 member names, oldest first]} for the tokens with all three past frames in the per-chunk tars."""
    def names(f):
        with tarfile.open(f"{d}/{f}") as t:
            return t.getnames()
    got = defaultdict(dict)
    with ThreadPoolExecutor(32) as ex:   # <chunk>.tar (the all-clip first run) and <chunk>.sel.tar; empty ones are placeholders
        for ns in ex.map(names, sorted(f for f in os.listdir(d) if f.endswith(".tar") and os.path.getsize(f"{d}/{f}") > 0)):
            for n in ns:
                _, clip, f = n.split("/"); ms, k = f[:-4].split("_")
                got[f"{clip}_{ms}"][int(k)] = n
    return {t: [v[k] for k in range(3)] for t, v in got.items() if len(v) == 3}


def same_bytes(hnav, fill):
    """Frame paths in both a fill tar and hist_nav's tar of the same log: the count; asserts every one has the same bytes."""
    def cmp(f):
        if not os.path.exists(f"{hnav}/trainval/{f}"):
            return 0, 0
        with tarfile.open(f"{fill}/trainval/{f}") as ta, tarfile.open(f"{hnav}/trainval/{f}") as tb:
            ma, mb = {m.name: m for m in ta}, {m.name: m for m in tb}
            both = sorted(ma.keys() & mb.keys())
            return len(both), sum(ta.extractfile(ma[n]).read() != tb.extractfile(mb[n]).read() for n in both)
    with ThreadPoolExecutor(16) as ex:
        r = list(ex.map(cmp, sorted(f for f in os.listdir(f"{fill}/trainval") if f.endswith(".tar"))))
    n, d = sum(x for x, _ in r), sum(y for _, y in r)
    assert d == 0, f"{d} of {n} frames in both hist_nav and its fill differ"
    return n


def main(pool, hnav, hpai, out):
    os.makedirs(out)   # never overwrite
    nav = {}
    for l in open(f"{hnav}/index.jsonl"):
        r = json.loads(l)
        if r["split"] == "trainval":
            nav[r["token"]] = r.get("past") or r["none"]
    fill, from_fill = f"{hnav}_fill", set()
    if os.path.exists(f"{fill}/index.jsonl"):
        for l in open(f"{fill}/index.jsonl"):
            r = json.loads(l)
            if r["split"] == "trainval" and r.get("past"):
                old = nav.get(r["token"])
                assert not isinstance(old, list) or old == r["past"], r["token"]   # never another frame for a token hist_nav has
                if not isinstance(old, list):
                    nav[r["token"]] = r["past"]; from_fill.add(r["token"])
    pai = pai_index(hpai)
    rep = {"nav_index": len(nav), "nav_from_fill": len(from_fill), "fill_frames_shared_byte_equal": same_bytes(hnav, fill) if from_fill else 0,
           "pai_index": len(pai), "pai_index_selected": sum(SEL(t.rsplit("_", 1)[0]) for t in pai)}
    for split in ("train", "dev"):
        st = defaultdict(Counter)
        with open(f"{pool}/{split}.jsonl") as fi, open(f"{out}/{split}.jsonl", "w") as fo:
            for line in fi:
                r = json.loads(line); k = kind(r["source"]); src = r["source"] if k is None else f"{k}:{r['source'] if not r['source'].startswith('if_') else 'l2d'}"
                p = (nav.get(r["token"], "not_in_index") if k == "nav" else "clip_not_selected" if not SEL(r["token"].rsplit("_", 1)[0])
                     else pai.get(r["token"], "not_decoded")) if k else None
                st[src]["rows"] += 1
                if isinstance(p, list):
                    assert len(r["images"]) == 1 and "history" not in r, r["token"]
                    r["images"] = r["images"] + [f"hist/{x}" for x in p]; r["history"] = True
                    fo.write(json.dumps(r) + "\n"); st[src]["history"] += 1
                    if k == "nav" and r["token"] in from_fill:
                        st[src]["history_from_fill"] += 1
                else:
                    fo.write(line if line.endswith("\n") else line + "\n")
                    if p:
                        st[src][p] += 1
        os.symlink(os.path.abspath(f"{pool}/{split}_targets.npy"), f"{out}/{split}_targets.npy")
        rep[split] = {s: dict(c) for s, c in sorted(st.items())}
    for split in ("train", "dev"):   # the rows without history are the pool's lines byte for byte, in order
        with open(f"{pool}/{split}.jsonl") as a, open(f"{out}/{split}.jsonl") as b:
            n = 0
            for la, lb in zip(a, b):
                n += 1
                if la != lb:
                    ra, rb = json.loads(la), json.loads(lb)
                    assert rb.pop("history") and rb["images"][0] == ra["images"][0] and len(rb.pop("images")) == 4 and ra.pop("images"), n
                    assert ra == rb, n
            assert next(a, None) is None and next(b, None) is None, split
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print(json.dumps(rep))


if __name__ == "__main__":
    main(*sys.argv[1:5])
