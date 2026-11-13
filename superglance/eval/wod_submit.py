"""WOD-E2E test submission with the official end_to_end_driving_submission.proto: serialized E2EDChallengeSubmission shards
(on disk <pkg>/part0, part1, ... as the official tutorial writes them), packed into <pkg>.tar.gz at the archive root as
mysubmission.binproto-<i>-of-<n> (waymo-open-dataset issue #936; the tutorial's directory of part<i> files gets "No shards
found"); the 1,505 test predictions fit in one shard.
build: a features or readout dir's tok_<r>.txt / code_<r>.npy (one candidate per frame) -> candidate (8 poses at 0.5-4 s in the
       frame's ego frame, x ahead, y left, metres) -> wod_rfs.to20 (20 points at 4 Hz, 0.25-5 s; 4-5 s at the last segment's
       velocity, as the val RFS) -> one FrameTrajectoryPredictions per listed frame, in the list's order. Metadata FILL_ME; the
       pretraining flag stays unset.
fill:  sets the metadata from the arguments in every shard, rewrites the parts and the .tar.gz; refuses while a FILL_ME is left.
check: parses the .tar.gz back (and compares it with the parts on disk); names = the list, once each; 20 finite points each;
       implied speeds against the ego speed of the rows' meta (all frames, plus 20 printed).
usage: python -m superglance.eval.wod_submit build <name> <features dir> <cands.npy> <out dir> <list.txt>
       python -m superglance.eval.wod_submit fill <pkg dir> --account-name ACCOUNT --method-name NAME --authors A [B ...]
              --affiliation AFF --description TEXT --method-link URL --uses-public-model-pretraining true|false
              --public-model-names [M ...] --num-model-parameters 9B
       python -m superglance.eval.wod_submit check <pkg dir> <list.txt> [<rows dir>]
Protos: the waymo-open-dataset package, or WOD_PROTOS = a directory with the modules generated from its .proto files."""
import argparse, glob, json, os, re, sys, tarfile

sys.path[:0] = [os.environ["WOD_PROTOS"]] if "WOD_PROTOS" in os.environ else []
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as sub   # noqa: E402

S = sub.E2EDChallengeSubmission
META = ("account_name", "unique_method_name", "authors", "affiliation", "description", "method_link", "uses_public_model_pretraining",
        "public_model_names", "num_model_parameters")
PLACE = dict(account_name="FILL_ME", unique_method_name="FILL_ME", authors=["FILL_ME"], affiliation="FILL_ME", description="FILL_ME",
             method_link="FILL_ME", public_model_names=["FILL_ME"], num_model_parameters="FILL_ME")


def parts(pkg):
    return sorted(glob.glob(f"{pkg}/part*"), key=lambda p: int(p.rsplit("part", 1)[1]))


def write(pkg, shards):
    os.makedirs(pkg, exist_ok=True)
    for i, s in enumerate(shards):
        open(f"{pkg}/part{i}.tmp", "wb").write(s.SerializeToString()); os.replace(f"{pkg}/part{i}.tmp", f"{pkg}/part{i}")
    # The tutorial's <pkg>/part<i> layout gets "No shards found" on the server. Waymo's fix (waymo-open-dataset issue #936):
    # shards named mysubmission.binproto-<i>-of-<n> at the archive root. GNU format, as a Linux tar would write it.
    with tarfile.open(f"{pkg}.tar.gz", "w:gz", format=tarfile.GNU_FORMAT) as t:
        for i, p in enumerate(parts(pkg)):
            ti = t.gettarinfo(p, arcname=f"mysubmission.binproto-{i:05d}-of-{len(shards):05d}"); ti.mtime = int(ti.mtime)
            t.addfile(ti, open(p, "rb"))


def meta(s):
    return {k: (list(getattr(s, k)) if k in ("authors", "public_model_names") else getattr(s, k) if s.HasField(k) else None) for k in META}


def build(a):
    import numpy as np
    from superglance.builders.pools.wod_rfs import to20
    names = [l.strip() for l in open(a.list) if l.strip()]; pick = {}
    for r in range(64):
        if os.path.exists(f"{a.xh}/tok_{r}.txt"):
            pick.update(zip(open(f"{a.xh}/tok_{r}.txt").read().split(), np.load(f"{a.xh}/code_{r}.npy")))
    miss = [n for n in names if n not in pick]
    if miss:
        sys.exit(f"STOP: {len(miss)} listed frames have no pick, e.g. {miss[:3]}")
    C = np.load(a.cands)[:, :, :2]; T = to20(C[[int(pick[n]) for n in names]]).astype(np.float32)
    s = S(submission_type=S.SubmissionType.Value("E2ED_SUBMISSION"), **PLACE)
    for n, t in zip(names, T):
        s.predictions.add(frame_name=n, trajectory=sub.TrajectoryPrediction(pos_x=t[:, 0].tolist(), pos_y=t[:, 1].tolist()))
    pkg = f"{a.out}/wodtest_{a.name}"; write(pkg, [s])
    print(f"built {pkg}/part0 + {pkg}.tar.gz: {len(s.predictions)} predictions from {len(pick)} picks ({len(set(pick.values()))} distinct "
          f"candidates), {os.path.getsize(pkg + '.tar.gz')} bytes")


def fill(a):
    if not re.fullmatch(r"[0-9]+[KMBTkmbt]", a.num_model_parameters):
        sys.exit("STOP: --num-model-parameters must be an integer with a K / M / B / T suffix, e.g. 9B")
    v = dict(account_name=a.account_name, unique_method_name=a.method_name, authors=a.authors, affiliation=a.affiliation,
             description=a.description, method_link=a.method_link, uses_public_model_pretraining=a.uses_public_model_pretraining == "true",
             public_model_names=a.public_model_names, num_model_parameters=a.num_model_parameters)
    if "FILL_ME" in json.dumps(v):
        sys.exit("STOP: a value is still FILL_ME")
    pkg = a.pkg.rstrip("/"); shards = [S.FromString(open(p, "rb").read()) for p in parts(pkg)]
    for s in shards:
        for k, x in v.items():
            if isinstance(x, list):
                del getattr(s, k)[:]; getattr(s, k).extend(x)
            else:
                setattr(s, k, x)
    write(pkg, shards)
    print(f"filled {len(shards)} part(s), {sum(len(s.predictions) for s in shards)} predictions -> {pkg}.tar.gz (upload this file)")
    print(json.dumps(meta(shards[0]), indent=1))


def check(a):
    import numpy as np
    pkg = a.pkg.rstrip("/"); names = [l.strip() for l in open(a.list) if l.strip()]
    with tarfile.open(f"{pkg}.tar.gz") as t:
        mem = [m for m in t.getmembers() if m.isfile()]; blobs = [t.extractfile(m).read() for m in mem]
    print("tar.gz members:", [m.name for m in mem], "| equal to the parts on disk:",
          sorted(blobs) == sorted(open(p, "rb").read() for p in parts(pkg)))
    shards = [S.FromString(b) for b in blobs]
    for m, s in zip(mem, shards):
        print(f"{m.name}: type {S.SubmissionType.Name(s.submission_type)}, {len(s.predictions)} predictions; metadata {json.dumps(meta(s))}")
    P = [p for s in shards for p in s.predictions]; got = [p.frame_name for p in P]
    print(f"predictions {len(P)}, distinct names {len(set(got))}, listed {len(names)}, names == list {sorted(got) == sorted(names)}, "
          f"same order as the list {got == names}")
    n = sorted({(len(p.trajectory.pos_x), len(p.trajectory.pos_y)) for p in P})
    X = np.array([np.stack([p.trajectory.pos_x, p.trajectory.pos_y], -1) for p in P], dtype=np.float64)
    print(f"points per prediction (x, y): {n}; all finite {bool(np.isfinite(X).all())}")
    rows = a.rows or ""
    if not os.path.isdir(f"{rows}/meta"):
        return
    R = {r["token"]: r for p in glob.glob(f"{rows}/meta/*.jsonl") for r in map(json.loads, open(p))}
    v0 = np.array([R[g]["speed"] for g in got]); seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(X), 1, 2)), X], 1), axis=1), axis=-1) / 0.25
    v05 = np.linalg.norm(X[:, 1], axis=-1) / 0.5; dv = v05 - v0   # mean speed over 0-0.5 s (the candidate's first pose) vs now
    print(f"ego speed now: median {np.median(v0):.2f}, max {v0.max():.2f} m/s; implied 0-0.5 s speed minus ego speed: median {np.median(dv):+.2f}, "
          f"|.| p95 {np.percentile(abs(dv), 95):.2f}, max {abs(dv).max():.2f} m/s; frames with |.| > 3 m/s: {int((abs(dv) > 3).sum())}")
    print(f"max 0.25 s step speed {seg.max():.1f} m/s; 5 s endpoint x [{X[:, -1, 0].min():.1f}, {X[:, -1, 0].max():.1f}], "
          f"y [{X[:, -1, 1].min():.1f}, {X[:, -1, 1].max():.1f}] m; going backwards (x < -0.5 at 5 s): {int((X[:, -1, 0] < -0.5).sum())}")
    print(f"{'frame':38s} {'cmd':12s} {'v now':>6s} {'v 0-0.5s':>8s} {'v mean 5s':>9s} {'end x':>6s} {'end y':>6s}")
    for i in np.linspace(0, len(P) - 1, 20).astype(int):
        print(f"{got[i]:38s} {R[got[i]]['command']:12s} {v0[i]:6.2f} {v05[i]:8.2f} {seg[i].mean():9.2f} {X[i, -1, 0]:6.1f} {X[i, -1, 1]:6.1f}")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); [b.add_argument(k) for k in ("name", "xh", "cands", "out", "list")]
    f = sp.add_parser("fill"); f.add_argument("pkg")
    for k in ("--account-name", "--method-name", "--affiliation", "--description", "--method-link", "--num-model-parameters"):
        f.add_argument(k, required=True)
    f.add_argument("--authors", nargs="+", required=True); f.add_argument("--public-model-names", nargs="*", required=True)
    f.add_argument("--uses-public-model-pretraining", choices=("true", "false"), required=True)
    c = sp.add_parser("check"); c.add_argument("pkg"); c.add_argument("list"); c.add_argument("rows", nargs="?")
    a = ap.parse_args(); {"build": build, "fill": fill, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
