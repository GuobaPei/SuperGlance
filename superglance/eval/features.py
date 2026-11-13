"""Frozen features: h = the final hidden state at the last prompt position, which the Policy Head reads, for every scene
of a token list, with the prompt, processor pixels (589,824) and model of the evaluation; also the head's own argmax.
--index (jsonl: image = the current frame's path, past = its three past frames, oldest first, or null) gives each listed
prompt its past frames, as the training rows have them; a prompt without them reads the current frame alone.
Shard r of n writes <out>/h_<r>.npy (float16 [N, d]), tok_<r>.txt, code_<r>.npy.
usage: python -m superglance.eval.features --inputs a.jsonl[,b.jsonl] --tokens t.txt --model M --adapter A --out DIR
       --k K [--index past.jsonl] [--rank r --world n] [--bs 8]"""
import argparse, json, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True); ap.add_argument("--tokens", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--k", type=int, default=4097); ap.add_argument("--max-pixels", type=int, default=589824)
    ap.add_argument("--index", default=None)   # past frames per current image (see above)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("EXTRACT_THREADS", 8)))   # torch's default (every core per process) times 8
    # processes oversubscribes the CPUs; threads change no number.
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    from superglance.data.policy import DecisionCollator, choice_linear
    from superglance.data.scene import c256_row, load_scenes
    scenes = {}
    for p in a.inputs.split(","):
        scenes.update(load_scenes(p))
    toks = [t.strip() for t in open(a.tokens) if t.strip()]
    toks = [t for t in toks if t in scenes][a.rank::a.world]
    os.makedirs(a.out, exist_ok=True)
    processor = AutoProcessor.from_pretrained(a.model, min_pixels=65536, max_pixels=a.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.choice_head = torch.nn.Linear(model.config.text_config.hidden_size, a.k, dtype=model.dtype)
    model = PeftModel.from_pretrained(model, a.adapter).cuda().eval()
    collate = DecisionCollator(processor)
    past = {r["image"]: r["past"] for r in map(json.loads, open(a.index)) if r["past"]} if a.index else {}
    n = [0, 0]   # prompts read without, with past frames
    head = model.get_output_embeddings()
    lin = choice_linear(model)
    H, C, seen = [], [], []
    hook = head.register_forward_hook(lambda module, args, output: seen.append(args[0][:, -1].float()))
    t0 = time.time()
    for s in range(0, len(toks), a.bs):
        rows = [c256_row(scenes[t], k=a.k) for t in toks[s:s + a.bs]]
        for r in rows:
            p = past.get(r["image_file"]); n[bool(p)] += 1
            if p:
                r.update(extra_image_files=p, history=True)
        inputs = collate(rows)
        for key in ("gold_index", "candidate_count"):
            inputs.pop(key, None)
        with torch.inference_mode():
            model(**inputs.to("cuda"), use_cache=False, logits_to_keep=1)
        h = seen.pop()
        H.append(h.half().cpu().numpy()); C.append((h @ lin.weight.float().T + lin.bias.float()).argmax(-1).cpu().numpy())
        if (s // a.bs) % 200 == 0:
            print(f"{s + len(rows)}/{len(toks)} {time.time() - t0:.0f} s", flush=True)
    hook.remove()
    np.save(f"{a.out}/h_{a.rank}.npy", np.concatenate(H)); np.save(f"{a.out}/code_{a.rank}.npy", np.concatenate(C))
    open(f"{a.out}/tok_{a.rank}.txt", "w").write("\n".join(toks) + "\n")
    print(f"DONE {len(toks)} scenes in {time.time() - t0:.0f} s", flush=True)
    print(f"features: {n[1]} prompts with past frames, {n[0]} without", flush=True)


if __name__ == "__main__":
    main()
