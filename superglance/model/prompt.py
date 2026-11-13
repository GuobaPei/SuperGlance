"""Multiple-choice prompt and option-letter readout at the answer position.

Each question, with all its options, is one sequence in a batch. The answer distribution is the softmax over the option
letters' logits at the position after the rendered chat prefix (thinking off), so no answer is generated. The readout
follows core/engine_torch.py of Qwen-2.5-1B-RLCD (Apache-2.0, see NOTICE).
"""

import argparse
import json
import string
from pathlib import Path


def choice_labels(count):
    """Option letters A, B, ... for 2 to 26 options, each one token."""
    if not 2 <= count <= 26:
        raise ValueError("The pilot supports 2–26 options per question")
    return list(string.ascii_uppercase[:count])


def build_choice_prompt(question, answers):
    labels = choice_labels(len(answers))
    return str(question) + "\n" + "\n".join(
        f"{label}. {answer}" for label, answer in zip(labels, answers)
    ) + "\nAnswer with only the letter of the correct option."


def chat_template_kwargs(processor):
    """Chat templates with a thinking switch (Qwen3.5) are rendered with
    enable_thinking=False, which appends the empty '<think>\\n\\n</think>\\n\\n'
    block, so the next position is the answer. Templates without the switch
    are rendered unchanged."""
    template = getattr(processor, "chat_template", None) or getattr(processor.tokenizer, "chat_template", None) or ""
    if isinstance(template, dict):
        template = " ".join(str(value) for value in template.values())
    return {"enable_thinking": False} if "enable_thinking" in template else {}


def render_chat(processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs(processor)
    )
    if "<think>" in text.rsplit("<|im_start|>assistant", 1)[-1] and not text.endswith("</think>\n\n"):
        raise ValueError("Chat prefix would enter thinking mode; the answer letter must be the next token")
    return text


def candidate_token_ids(processor, max_options):
    """Use the checkpoint tokenizer, never hard-code vocabulary IDs."""
    tokenizer = processor.tokenizer
    labels = choice_labels(max_options)
    encoded = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError("Candidate labels must each encode as exactly one token")
    ids = [row[0] for row in encoded]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate labels must have distinct token IDs")
    return ids


def validate_answer_boundary(processor, max_options):
    """Check real chat-prefix tokenization, not only isolated label strings."""
    labels = choice_labels(max_options)
    messages = [{"role": "user", "content": [{
        "type": "text", "text": build_choice_prompt("Select an option.", labels)
    }]}]
    text = render_chat(processor, messages)
    tokenizer = processor.tokenizer
    prefix = tokenizer.encode(text, add_special_tokens=False)
    token_ids = candidate_token_ids(processor, max_options)
    for label, token_id in zip(labels, token_ids):
        actual = tokenizer.encode(text + label, add_special_tokens=False)
        if actual != prefix + [token_id]:
            raise ValueError(f"Label {label} is not one token after the chat prefix")
    return dict(zip(labels, token_ids))


def native_candidate_logits(model, inputs, token_ids, fp32=False):
    """Differentiable native LM-head readout; inputs must use LEFT padding.

    logits_to_keep=1 avoids materializing vocabulary logits at every image/text
    position. The model still executes its native processor/M-RoPE/DeepStack.
    fp32=True recomputes the candidate logits from the final hidden state in
    float32: the bf16 LM head rounds logits near 25 to steps of 0.125, which
    ties the top two of 16 options in 12-16% of stage-3 rows.
    """
    if not fp32:
        outputs = model(**inputs, use_cache=False, logits_to_keep=1)
        return outputs.logits[:, -1, token_ids].float()
    head = getattr(model, "module", model).get_output_embeddings()
    assert head.bias is None
    seen = []
    hook = head.register_forward_hook(lambda module, args, output: seen.append(args[0]))
    try:
        model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        hook.remove()
    return seen[-1][:, -1].float() @ head.weight[token_ids].float().T


def constrained_probabilities(logits, candidate_counts, temperature=1.0):
    """Normalize per row over real options only; padding options get zero mass."""
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    counts = torch.as_tensor(candidate_counts, device=logits.device)
    valid = torch.arange(logits.shape[-1], device=logits.device)[None, :] < counts[:, None]
    return (logits.float() / temperature).masked_fill(~valid, -torch.inf).softmax(-1)


def format_decisions(rows, probabilities):
    """Keep raw probabilities; softmax alone is not evidence of calibration."""
    results = []
    for row, values in zip(rows, probabilities):
        answers = row["answers"]
        probs = [float(value) for value in values[:len(answers)]]
        winner = max(range(len(answers)), key=probs.__getitem__)
        result = {
            "id": row.get("id"),
            "pattern": row.get("pattern", "Choice"),
            "predicted_index": winner,
            "label": choice_labels(len(answers))[winner],
            "value": answers[winner],
            "probabilities": probs,
            "max_probability": probs[winner],
        }
        if "yes_index" in row:
            result["p_yes"] = probs[row["yes_index"]]
        results.append(result)
    return results


class VLPCDEngine:
    """Independent multimodal decisions batched in one model forward pass."""

    def __init__(self, model, processor):
        self.model = model.eval()
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"
        self.validated_options = 0

    @classmethod
    def from_pretrained(cls, checkpoint, max_pixels=262144, device_map="auto"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(
            checkpoint, min_pixels=65536, max_pixels=max_pixels
        )
        # q35 port: Auto class resolves Qwen3_5ForConditionalGeneration (transformers >= 5).
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, dtype=torch.bfloat16, device_map=device_map,
            attn_implementation="sdpa"
        )
        return cls(model, processor)

    def prepare(self, rows):
        from PIL import Image

        texts, images = [], []
        for row in rows:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": build_choice_prompt(row["question"], row["answers"])}
            ]}]
            texts.append(render_chat(self.processor, messages))
            with Image.open(row["image_file"]) as image:
                images.append(image.convert("RGB"))
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt")

    def score(self, rows, temperature=1.0):
        import torch

        max_options = max(len(row["answers"]) for row in rows)
        if max_options > self.validated_options:
            validate_answer_boundary(self.processor, max_options)
            self.validated_options = max_options
        inputs = self.prepare(rows).to(self.model.device)
        ids = candidate_token_ids(self.processor, max_options)
        with torch.inference_mode():
            logits = native_candidate_logits(self.model, inputs, ids)
            probabilities = constrained_probabilities(
                logits, [len(row["answers"]) for row in rows], temperature
            ).cpu().tolist()
        return format_decisions(rows, probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=262144)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    for row in rows:
        row["image_file"] = str(args.manifest.parent / row["images"][0])
    engine = VLPCDEngine.from_pretrained(args.model, max_pixels=args.max_pixels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            for result in engine.score(batch, args.temperature):
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Scored {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
