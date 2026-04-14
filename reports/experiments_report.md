# Experiment Report: Fine-Tuning NLLB-200 for Kangri Translation

**Date**: 2026-04-14  
**Base Model**: `facebook/nllb-200-distilled-600M`  
**Target Language**: Kangri (Kng)  
**Source Language**: Hindi (Hin)

---

## 1. Executive Summary
The primary challenge of this project was severe unalignment and language-swapping in the initial raw text datasets. Through the implementation of a **Neural Semantic Aligner (LaBSE)**, we successfully filtered 26k lines of noisy data into a high-quality "Golden + Silver" tiered dataset of ~13,800 pairs. This resulted in a **+14.66 BLEU** improvement over the failed baseline.

---

## 2. Experimental Timeline

### Experiment 1: Raw Baseline (Sequential Alignment)
*   **Strategy**: Zip-pairing `Kr_4_Hindi.txt` and `Kr_4_kangri.txt` sequentially.
*   **Result**: **Critical Failure**.
*   **Observations**: 
    *   BLEU: 0.03 | chrF: 5.11
    *   The model produced hallucinations and repetitive patterns (e.g., "मेरी ता इनां...").
    *   Diagnosis: Raw files contained arbitrary duplicate blocks and dropped sentences, causing indices to drift by hundreds of lines.

### Experiment 2: Neural Semantic Aligner (Golden Set)
*   **Strategy**: Used `sentence-transformers/LaBSE` to embed each sentence and used Hungarian Matching to find pairs with similarity > 0.78.
*   **Dataset Size**: 5,731 pairs.
*   **Result**: **Significant Success**.
*   **Metrics**: BLEU: 12.13 | chrF: 41.01
*   **Observations**: The model began producing meaningful translation attempts for the first time.

### Experiment 3: Tiered Expansion (Golden + Silver Set) - 5 Epochs
*   **Strategy**: Expanded the similarity threshold to include "Silver" data (> 0.60).
*   **Dataset Size**: 13,831 pairs.
*   **Metrics**: BLEU: 14.69 | chrF: 44.80 | NSSS (Ref): 0.8722 | NSSS (Src): 0.9081

### Experiment 4: Extended Fine-Tuning - 10 Epochs
*   **Strategy**: Same as Exp 3, but increased training duration to 10 epochs.
*   **Metrics**: **BLEU: 16.24** | **chrF: 46.20** | **NSSS (Ref): 0.8767** | **NSSS (Src): 0.8908**
*   **Observations**: 
    *   **Success**: Highest overlap with reference text (BLEU +1.55).
    *   **Tradeoff**: NSSS (Src) decreased. This suggests the model is becoming less of a "literal/Hindi-like" translator and is adapting more deeply to the specific Kangri dialectal patterns in the reference set. The model is increasingly "speaking Kangri" rather than "translating Hindi words".

---

## 6. Stage 2: MT-SPIN Self-Play Fine-Tuning

### SPIN Overview
Self-Play Fine-Tuning (SPIN) uses DPO to iteratively align the model's translations against gold references. Each iteration generates candidate translations, selects the worst (easy curriculum) or best (hard curriculum) as rejected examples, and trains the model to prefer gold over its own outputs.

**Pipeline**: Generate preference pairs → DPO Training → Evaluate → Repeat

### Experiment 5: SPIN Iteration 1 — Initial Attempt (FAILED)
*   **Strategy**: DPO training with policy initialized from a **fresh random LoRA adapter**, reference model loaded from SFT adapter (Exp 4).
*   **Hyperparameters**: β=0.1, LR=2e-4, 3 epochs, batch=4, grad_accum=4
*   **Preference Data**: 12,694 pairs (after filtering chrF gap > 0.05 from 13,831)
*   **Result**: **Critical Failure — Complete Model Collapse**
*   **Metrics**: BLEU: 0.10 | chrF: 1.57 | NSSS (Ref): 0.0702 | NSSS (Src): 0.0593
*   **Observations**:
    *   Model output degenerated to repetitive garbage: `. . . . . . . . .` or `यांयांयांयां...`
    *   Every single test example produced nonsensical output
    *   Training loss dropped to near-zero prematurely — classic DPO over-optimization

#### Root Cause Analysis
Weight comparison between SFT and collapsed SPIN adapter revealed:
*   **Relative weight drift = 1.05x** — weights moved more than their entire original norm
*   `lora_B` weights collapsed to near-zero (0.08 vs 3.1 in SFT original)
*   No NaN/Inf — pure degeneration from **incorrect initialization**

**Root Cause**: The policy model was initialized with **random LoRA weights** over bare NLLB base (`get_peft_model()`), while the reference model loaded the trained SFT adapter. In DPO, the policy must start from the **same checkpoint** as the reference — otherwise the log-ratio signal is meaningless and the policy quickly diverges.

**Additional contributing factors**:
*   β=0.1 too low (weak KL penalty → excessive drift from reference)
*   LR=2e-4 too high for DPO (causes over-optimization)
*   3 epochs excessive (DPO converges fast; extra epochs → collapse)

#### Fixes Applied
1.  **Policy initialization**: Changed to `PeftModel.from_pretrained(..., is_trainable=True)` — loads SFT adapter weights as starting point
2.  **DPO monitoring**: Added per-step logging of accuracy, chosen/rejected rewards, margins, and log-probabilities
3.  **Eval decoding**: Added `repetition_penalty=1.2` and `no_repeat_ngram_size=3`
4.  **Prediction step override**: Custom `prediction_step()` prevents `Trainer` from passing DPO batch keys to the encoder during eval

---

### Experiment 6: SPIN Iteration 1 v2 — Fixed Init, Aggressive LR
*   **Strategy**: Same as Exp 5 but with corrected policy initialization (loaded from SFT adapter).
*   **Hyperparameters**: β=0.3, LR=5e-5, 1 epoch
*   **Result**: **Partial Success — No Collapse, But Significant Regression**
*   **Metrics**: BLEU: 8.26 | chrF: 35.01 | NSSS (Ref): 0.7298 | NSSS (Src): 0.6591
*   **Training Stats**: train_loss=0.4914, eval_loss=0.3421
*   **Observations**:
    *   Model produces real Kangri (not garbage) — initialization fix confirmed as critical
    *   Short sentences excellent (e.g., `कुत्थु चलणा ?`)
    *   Longer sentences still showed repetition loops before eval fix
    *   LR=5e-5 still too aggressive — policy drifted significantly from SFT baseline

---

### Experiment 7: SPIN Iteration 1 v3 — Conservative Parameters ✓
*   **Strategy**: More conservative DPO hyperparameters to minimize deviation from SFT.
*   **Hyperparameters**: β=0.5, LR=1e-5, 1 epoch, max_grad_norm=1.0
*   **Result**: **Best SPIN Result — Near Baseline Performance**
*   **Metrics**: **BLEU: 15.50** | **chrF: 43.27** | **NSSS (Ref): 0.8417** | **NSSS (Src): 0.8009**
*   **Training Stats**: train_loss=0.7482, eval_loss=0.3981
*   **Observations**:
    *   Only -1.3 BLEU from SFT baseline — model preserved most translation quality
    *   Coherent Kangri output with no repetition loops
    *   Higher train loss (0.75 vs 0.49 in v2) indicates less overfitting — the policy stayed closer to the reference
    *   DPO accuracy fluctuated 0.5–1.0 throughout, suggesting the model learned preference signals without over-optimizing

---

## 7. Comparative Metrics Table (Updated)

| Metric | Exp 1 (Raw) | Exp 2 (Gold) | Exp 3 (5 Ep) | Exp 4 (10 Ep) | Exp 5 (SPIN v1) | Exp 6 (SPIN v2) | **Exp 7 (SPIN v3)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Pairs** | 10k+ (Noisy) | 5,731 | 13,831 | 13,831 | 12,694 pref | 12,694 pref | 12,694 pref |
| **BLEU Score** | 0.03 | 12.13 | 14.69 | 16.79 | 0.10 | 8.26 | **15.50** |
| **chrF Score** | 5.11 | 41.01 | 44.80 | 46.84 | 1.57 | 35.01 | **43.27** |
| **NSSS (Ref)** | - | - | 0.8722 | 0.8791 | 0.07 | 0.7298 | **0.8417** |
| **NSSS (Src)** | - | - | 0.9081 | 0.8911 | 0.06 | 0.6591 | **0.8009** |

*Note: Exp 4+ metrics use `repetition_penalty=1.2` and `no_repeat_ngram_size=3` in decoding. Earlier experiments did not use these.*

---

## 8. Key Technical Innovations
1.  **Semantic Similarity Filtering**: Solved text-level drift using LaBSE embeddings instead of sequential indices.
2.  **Tiered Data Strategy**: Demonstrated that "Silver" data (0.60-0.75 similarity) is beneficial for NMT when the Golden set is small.
3.  **Language-Agnostic Baseline**: Leveraged NLLB-200's zero-shot capability by using `hin_Deva` tokens for the low-resource Kangri script.
4.  **DPO Initialization Fix**: Identified that DPO policy must be initialized from the same adapter checkpoint as the reference model.
5.  **DPO Hyperparameter Sensitivity**: Demonstrated that conservative parameters (β=0.5, LR=1e-5, 1 epoch) are critical for stable DPO training on small datasets.

---

## 9. Sample Output (SPIN v3 — Current Best SPIN)
*   **Source**: कहाँ चलना ?
*   **Reference**: कुथू चलणा ?
*   **Prediction**: कुथू चलणा ? ✅ (Exact match)

*   **Source**: शिलालेख / राजाओं के लिखे पत्र...
*   **Reference**: शिलालेख/ राजेयां दियां लिखियां चिट्ठियां...
*   **Prediction**: शिलालेख / राजयां दे लिखेया पत्र... (Coherent Kangri)

*   **Source**: लेकिन जल गई फूकने वाली मन्दबुद्धि...
*   **Reference**: पर जली अई फूकणी बूसर मत...
*   **Prediction**: पर जली गई फक्की ने बणदी मन्दबुद्धि... (Partial match)

---

## 10. Next Steps
*   **SPIN Iteration 2**: Generate hard-curriculum preference pairs using v3 adapter, then DPO train iteration 2.
*   **SPIN Iteration 3**: Repeat with iteration 2 adapter.
*   **Final Evaluation**: Compare iteration 2/3 against SFT baseline to quantify cumulative SPIN gains.

