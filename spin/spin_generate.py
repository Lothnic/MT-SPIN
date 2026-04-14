"""
MT-SPIN Stage 1: Candidate Generation

Takes the LoRA SFT adapter (or iteration N adapter) and generates
diverse candidate translations for each Hindi sentence in the training set.

Output: HuggingFace dataset saved to disk with columns:
  prompt        : source sentence (Hindi)
  chosen        : gold reference translation (Kangri)
  rejected      : worst model output (scored by chrF)
  chrf_chosen   : always 1.0 (gold)
  chrf_rejected : chrF score of the rejected candidate
"""

import argparse
import os
import torch
import sacrebleu
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset, Dataset

# ── Defaults ───────────────────────────────────────────────────────────────
BASE_MODEL      = "facebook/nllb-200-distilled-600M"
SRC_LANG        = "hin_Deva"
TGT_LANG        = "hin_Deva"   # proxy for Kangri (kng_Deva not in NLLB vocab)
NUM_CANDIDATES  = 4
BATCH_SIZE      = 8
MAX_SRC_LEN     = 128
MAX_TGT_LEN     = 128
# ───────────────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(adapter_path: str):
    """Load base NLLB + LoRA adapter in bf16 for inference."""
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, src_lang=SRC_LANG, tgt_lang=TGT_LANG
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_candidates(model, tokenizer, source: str, num_cands: int):
    """
    Generate num_cands diverse translations for a single source sentence.
    Uses temperature sampling with top-p for diversity (memory-safe on 8GB GPU).
    Returns List[str] — num_cands candidate translations.
    """
    lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    inputs = tokenizer(
        source, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_SRC_LEN
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=lang_id,
            max_new_tokens=MAX_TGT_LEN,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=num_cands,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


def score_chrf(hypothesis: str, reference: str) -> float:
    """Returns chrF++ score normalised to [0, 1]."""
    result = sacrebleu.corpus_chrf([hypothesis], [[reference]])
    return result.score / 100.0


def select_rejected(candidates: list[str], reference: str, curriculum: str = "easy"):
    """
    Pick a candidate as the 'rejected' example.
    - 'easy' (iter 0): pick the WORST candidate → largest chrF gap → strong signal
    - 'hard' (iter 1+): pick the BEST candidate → smallest gap → harder negatives
    """
    scored = [(c, score_chrf(c, reference)) for c in candidates]
    if curriculum == "easy":
        scored.sort(key=lambda x: x[1])        # ascending — worst first
    else:
        scored.sort(key=lambda x: x[1], reverse=True)  # descending — best first
    return scored[0][0], scored[0][1]


def main():
    parser = argparse.ArgumentParser(description="MT-SPIN Stage 1: Generate preference pairs")
    parser.add_argument("--adapter", type=str, default="models/nllb-200-kangri-lora/final_adapter",
                        help="Path to the LoRA adapter checkpoint")
    parser.add_argument("--data", type=str, default="data/processed_dataset/train_combined.parquet",
                        help="Path to the training parquet file")
    parser.add_argument("--output", type=str, default="spin/spin_data/iteration_0",
                        help="Output directory for the preference dataset")
    parser.add_argument("--curriculum", type=str, default="easy", choices=["easy", "hard"],
                        help="'easy' = worst candidate (iter 0), 'hard' = best candidate (iter 1+)")
    parser.add_argument("--num_candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Load data
    print(f"Loading dataset from {args.data}...")
    ds = load_dataset("parquet", data_files={"train": args.data})["train"]
    print(f"Loaded {len(ds)} training examples")

    # Extract parallel sentences
    sources = [row["translation"]["hin_Deva"] for row in ds]
    targets = [row["translation"]["kng_Deva"] for row in ds]

    # Load model
    print(f"Loading model from {args.adapter}...")
    model, tokenizer = load_model_and_tokenizer(args.adapter)

    # Generate preference pairs
    records = []
    for i in tqdm(range(len(sources)), desc="Generating candidates"):
        src = sources[i]
        gold = targets[i]

        candidates = generate_candidates(model, tokenizer, src, args.num_candidates)
        rejected, chrf_rej = select_rejected(candidates, gold, args.curriculum)
        records.append({
            "prompt":        src,
            "chosen":        gold,
            "rejected":      rejected,
            "chrf_chosen":   1.0,
            "chrf_rejected": chrf_rej,
        })

    # Save
    os.makedirs(args.output, exist_ok=True)
    dataset = Dataset.from_list(records)
    dataset.save_to_disk(args.output)
    print(f"\nSaved {len(records)} preference pairs → {args.output}")

    # Sanity stats
    avg_gap = sum(r["chrf_chosen"] - r["chrf_rejected"] for r in records) / len(records)
    avg_rej = sum(r["chrf_rejected"] for r in records) / len(records)
    print(f"Average chrF gap (chosen - rejected): {avg_gap:.4f}")
    print(f"Average rejected chrF score: {avg_rej:.4f}")

    # Print a few examples
    print("\nSample preference pairs:")
    for i in range(min(3, len(records))):
        r = records[i]
        print(f"  HIN:      {r['prompt'][:80]}")
        print(f"  GOLD KNG: {r['chosen'][:80]}")
        print(f"  REJ  KNG: {r['rejected'][:80]} (chrF: {r['chrf_rejected']:.3f})")
        print()


if __name__ == "__main__":
    main()