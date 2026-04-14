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

## 3. Comparative Metrics Table

| Metric | Exp 1 (Raw) | Exp 2 (Gold) | Exp 3 (5 Ep) | Exp 4 (10 Ep) |
| :--- | :---: | :---: | :---: | :---: |
| **Training Pairs** | 10,000+ (Noisy) | 5,731 | 13,831 | 13,831 |
| **BLEU Score** | 0.03 | 12.13 | 14.69 | **16.24** |
| **chrF Score** | 5.11 | 41.01 | 44.80 | **46.20** |
| **NSSS (Ref)** | - | - | 0.8722 | **0.8767** |
| **NSSS (Src)** | - | - | **0.9081** | 0.8908 |

*NSSS: Neural Semantic Similarity Score (measured against Reference).*

---

## 4. Key Technical Innovations
1.  **Semantic Similarity Filtering**: Solved text-level drift using LaBSE embeddings instead of sequential indices.
2.  **Tiered Data Strategy**: Demonstrated that "Silver" data (0.60-0.75 similarity) is beneficial for NMT when the Golden set is small.
3.  **Language-Agnostic Baseline**: Leveraged NLLB-200's zero-shot capability by using `hin_Deva` tokens for the low-resource Kangri script.

---

## 5. Sample Output (Final Model)
*   **Source**: कहाँ चलना ?
*   **Reference**: कुथू चलणा ?
*   **Prediction**: कुथू चलना ? (High Accuracy)

*   **Source**: बोलना तो छोड़ो...
*   **Reference**: बोलणा तां छड्डा...
*   **Prediction**: बोलणा तां छोड़ी... (Grammatically correct)

---

## 6. Next Steps: SPIN
The model is now ready for **Stage 2: Self-Play Fine-Tuning**. Would start with SPIN ASAP.
