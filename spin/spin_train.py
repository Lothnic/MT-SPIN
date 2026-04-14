"""
MT-SPIN Stage 2: DPO Training with chrF-Weighted Loss (No TRL dependency)

Implements DPO loss from scratch for Seq2Seq models since TRL 1.x
dropped encoder-decoder support and is incompatible with our
transformers version.

Uses Hugging Face Trainer as the training backbone.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_from_disk

# ── Defaults ───────────────────────────────────────────────────────────────
BASE_MODEL     = "facebook/nllb-200-distilled-600M"
SRC_LANG       = "hin_Deva"
TGT_LANG       = "hin_Deva"

LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05

BETA           = 0.3      # DPO temperature (higher = stronger KL penalty)
LAMBDA_REWARD  = 0.5      # weight of chrF reward term
NUM_EPOCHS     = 1        # DPO converges fast; extra epochs cause collapse
BATCH_SIZE     = 4
GRAD_ACCUM     = 4         # effective batch = 16
LR             = 5e-5      # conservative LR for DPO stability
MAX_SEQ_LEN    = 256
MAX_PROMPT_LEN = 128
# ───────────────────────────────────────────────────────────────────────────


def get_seq2seq_logps(model, input_ids, attention_mask, labels):
    """
    Compute per-sequence sum of log probabilities for a Seq2Seq model.
    
    Args:
        model: encoder-decoder model
        input_ids: encoder input [batch, src_len]
        attention_mask: encoder mask [batch, src_len]
        labels: decoder target ids [batch, tgt_len]
    
    Returns:
        Tensor of shape [batch] with sum-of-logp per sequence.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    logits = outputs.logits  # [batch, tgt_len, vocab]

    # Per-token negative log-likelihoods (mask out -100 positions)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token_nll = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )
    per_token_nll = per_token_nll.view(labels.size(0), -1)

    # Sum of log-probs = negative sum of NLL
    return -per_token_nll.sum(dim=1)


class DPOSeq2SeqTrainer(Trainer):
    """
    DPO Trainer for Encoder-Decoder models.
    
    Uses a frozen reference model and computes the standard DPO loss
    with an optional chrF-weighted reward gap scaling.
    """

    def __init__(self, ref_model, beta=0.1, lambda_reward=0.5, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.lambda_reward = lambda_reward
        self._dpo_step = 0

        # Move ref_model to the same device as the policy
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.args.device)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override so the eval loop doesn't call model(**inputs) with our custom
        DPO batch keys — NLLB's encoder would raise ValueError about missing
        input_ids.  We compute the DPO loss directly instead.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract our custom fields
        prompt_ids = inputs["prompt_input_ids"]
        prompt_mask = inputs["prompt_attention_mask"]
        chosen_labels = inputs["chosen_labels"]
        rejected_labels = inputs["rejected_labels"]
        chrf_chosen = inputs["chrf_chosen"]
        chrf_rejected = inputs["chrf_rejected"]

        # Policy log-probs
        policy_chosen_logps = get_seq2seq_logps(model, prompt_ids, prompt_mask, chosen_labels)
        policy_rejected_logps = get_seq2seq_logps(model, prompt_ids, prompt_mask, rejected_labels)

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_chosen_logps = get_seq2seq_logps(self.ref_model, prompt_ids, prompt_mask, chosen_labels)
            ref_rejected_logps = get_seq2seq_logps(self.ref_model, prompt_ids, prompt_mask, rejected_labels)

        # DPO loss: -log σ(β * ((π_θ(chosen) - π_ref(chosen)) - (π_θ(rejected) - π_ref(rejected))))
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        loss = -F.logsigmoid(self.beta * logits).mean()

        # chrF reward weighting
        reward_gap = (chrf_chosen - chrf_rejected).to(loss.device)
        scale = 1.0 + self.lambda_reward * reward_gap.mean()
        loss = loss * scale

        # ── DPO Monitoring ──
        self._dpo_step += 1
        if self._dpo_step % 25 == 1:  # log every 25 steps
            with torch.no_grad():
                acc = (logits > 0).float().mean().item()
                chosen_reward = (policy_chosen_logps - ref_chosen_logps).mean().item()
                rejected_reward = (policy_rejected_logps - ref_rejected_logps).mean().item()
                margin = (chosen_reward - rejected_reward)
                print(
                    f"  [DPO step {self._dpo_step}] "
                    f"acc={acc:.3f} | "
                    f"chosen_reward={chosen_reward:.3f} | "
                    f"rejected_reward={rejected_reward:.3f} | "
                    f"margin={margin:.3f} | "
                    f"pi_chosen_logp={policy_chosen_logps.mean().item():.1f} | "
                    f"pi_rejected_logp={policy_rejected_logps.mean().item():.1f} | "
                    f"loss={loss.item():.4f}"
                )

        return (loss, None) if return_outputs else loss


class MTSpinDataCollator:
    """Pads Seq2Seq DPO batches and preserves chrF scores."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        def pad_seq(seqs, pad_val):
            max_len = max(len(s) for s in seqs)
            return [s + [pad_val] * (max_len - len(s)) for s in seqs]

        def to_tensor(seqs, dtype=torch.long):
            return torch.tensor(seqs, dtype=dtype)

        batch = {}
        for key in ["prompt_input_ids", "prompt_attention_mask",
                     "chosen_labels", "rejected_labels"]:
            if key not in features[0]:
                continue
            seqs = [f[key] for f in features]
            if "mask" in key:
                pad_val = 0
            elif "labels" in key:
                pad_val = -100
            else:
                pad_val = self.pad_id
            batch[key] = to_tensor(pad_seq(seqs, pad_val))

        batch["chrf_chosen"] = torch.tensor([f["chrf_chosen"] for f in features])
        batch["chrf_rejected"] = torch.tensor([f["chrf_rejected"] for f in features])
        return batch


def prepare_dataset(dataset, tokenizer):
    """Tokenize the preference dataset for our Seq2Seq DPO trainer."""
    tokenizer.src_lang = SRC_LANG

    def tokenize(example):
        src_enc = tokenizer(
            example["prompt"],
            max_length=MAX_PROMPT_LEN,
            truncation=True,
            padding=False,
        )
        chosen_enc = tokenizer(
            text_target=example["chosen"],
            max_length=MAX_SEQ_LEN - MAX_PROMPT_LEN,
            truncation=True,
            padding=False,
        )
        rejected_enc = tokenizer(
            text_target=example["rejected"],
            max_length=MAX_SEQ_LEN - MAX_PROMPT_LEN,
            truncation=True,
            padding=False,
        )
        return {
            "prompt_input_ids":      src_enc["input_ids"],
            "prompt_attention_mask": src_enc["attention_mask"],
            "chosen_labels":         chosen_enc["input_ids"],
            "rejected_labels":       rejected_enc["input_ids"],
            "chrf_chosen":           float(example["chrf_chosen"]),
            "chrf_rejected":         float(example["chrf_rejected"]),
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names, num_proc=4)


def main():
    parser = argparse.ArgumentParser(description="MT-SPIN Stage 2: DPO Training")
    parser.add_argument("--ref_adapter", type=str,
                        default="models/nllb-200-kangri-lora/final_adapter")
    parser.add_argument("--spin_data", type=str,
                        default="spin/spin_data/iteration_0")
    parser.add_argument("--output", type=str,
                        default="models/spin_iter_1")
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--lambda_reward", type=float, default=LAMBDA_REWARD)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    # ── Reference Model (frozen) ──
    print(f"Loading reference model from {args.ref_adapter}...")
    ref_base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    ref_model = PeftModel.from_pretrained(ref_base, args.ref_adapter)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Policy Model (initialized from same adapter as reference) ──
    print(f"Building policy model from {args.ref_adapter} (same init as reference)...")
    policy_base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    policy_model = PeftModel.from_pretrained(policy_base, args.ref_adapter, is_trainable=True)
    policy_model.print_trainable_parameters()

    # ── Load and tokenize preference dataset ──
    print(f"Loading preference dataset from {args.spin_data}...")
    raw_dataset = load_from_disk(args.spin_data)

    raw_dataset = raw_dataset.filter(
        lambda x: (x["chrf_chosen"] - x["chrf_rejected"]) > 0.05
    )
    print(f"After filtering: {len(raw_dataset)} preference pairs")

    train_val = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = prepare_dataset(train_val["train"], tokenizer)
    eval_ds = prepare_dataset(train_val["test"], tokenizer)

    # ── Training Config ──
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        max_grad_norm=1.0,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    collator = MTSpinDataCollator(tokenizer)

    print("Starting MT-SPIN DPO training...")
    trainer = DPOSeq2SeqTrainer(
        ref_model=ref_model,
        beta=args.beta,
        lambda_reward=args.lambda_reward,
        model=policy_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output)
    print(f"\nSaved new adapter → {args.output}")


if __name__ == "__main__":
    main()
