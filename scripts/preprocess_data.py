import os
import torch
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

def clean_lines(filepath):
    """ Read lines, remove empty, apply simple deduplication """
    with open(filepath, 'r', encoding='utf-8') as f:
        # Some simple cleaning
        # Remove headers like "कांगडी - 6"
        lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "कांगडी" in line or "कांगड़ी" in line:
                continue 
            lines.append(line)
            
    # We don't use strict set deduplication because order matters in texts.
    # We will dedupe sequentially (sliding window dedupe)
    deduped = []
    seen = set()
    for line in lines:
        if line not in seen:
            deduped.append(line)
            seen.add(line)
        else:
            # If we've seen it before, just skip it to remove the massive noise repeats
            pass
            
    return deduped

def extract_tiered_pairs(hin_lines, kng_lines, model, chunk_size=1000, gold_threshold=0.78, silver_threshold=0.6):
    """
    Chunks datasets, embeds via LaBSE, and uses Hungarian matching to extract Gold and Silver tiers.
    """
    gold_pairs = []
    silver_pairs = []
    
    num_chunks = max(len(hin_lines), len(kng_lines)) // chunk_size + 1
    
    for i in tqdm(range(num_chunks), desc="Processing Chunks via LaBSE"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size + 200, len(hin_lines))
        h_chunk = hin_lines[start_idx:end_idx]
        
        end_idx_kng = min((i + 1) * chunk_size + 200, len(kng_lines))
        k_chunk = kng_lines[start_idx:end_idx_kng]
        
        if not h_chunk or not k_chunk:
            continue
            
        h_emb = model.encode(h_chunk, convert_to_tensor=True, show_progress_bar=False)
        k_emb = model.encode(k_chunk, convert_to_tensor=True, show_progress_bar=False)
        
        cos_scores = util.cos_sim(h_emb, k_emb).cpu().numpy()
        
        cost_matrix = -cos_scores
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            score = cos_scores[r, c]
            pair = {
                "translation": {
                    "hin_Deva": h_chunk[r],
                    "kng_Deva": k_chunk[c]
                },
                "score": float(score)
            }
            if score >= gold_threshold:
                gold_pairs.append(pair)
            elif score >= silver_threshold:
                silver_pairs.append(pair)
                
    return gold_pairs, silver_pairs

def main():
    base_dir = "."
    os.makedirs(os.path.join(base_dir, "data", "processed"), exist_ok=True)
    
    train_kng = os.path.join(base_dir, "data", "train dataset", "Kr_4_kangri.txt")
    train_hin = os.path.join(base_dir, "data", "train dataset", "Kr_4_Hindi.txt")
    test_kng = os.path.join(base_dir, "data", "test dataset", "Kr_4_Kangri.txt")
    test_hin = os.path.join(base_dir, "data", "test dataset", "Kr_4_Hindi.txt")
    
    print("Loading LaBSE Model...")
    model = SentenceTransformer("sentence-transformers/LaBSE").to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nProcessing Train Dataset...")
    tr_h = clean_lines(train_hin)
    tr_k = clean_lines(train_kng)
    print(f"Cleaned lines - Hindi: {len(tr_h)}, Kangri: {len(tr_k)}")
    
    gold_train, silver_train = extract_tiered_pairs(tr_h, tr_k, model, gold_threshold=0.78, silver_threshold=0.6)
    
    print("\nProcessing Test Dataset...")
    te_h = clean_lines(test_hin)
    te_k = clean_lines(test_kng)
    # We only take Gold pairs for test to ensure evaluation is on high-quality data
    gold_test, _ = extract_tiered_pairs(te_h, te_k, model, gold_threshold=0.78, silver_threshold=0.78)
    
    # Check for strict leakage!
    train_hin_set = set([p["translation"]["hin_Deva"] for p in gold_train + silver_train])
    train_kng_set = set([p["translation"]["kng_Deva"] for p in gold_train + silver_train])
    
    clean_test_pairs = []
    for tp in gold_test:
        if tp["translation"]["hin_Deva"] not in train_hin_set and tp["translation"]["kng_Deva"] not in train_kng_set:
            clean_test_pairs.append(tp)
            
    print(f"\nFinal Train Golden Pairs: {len(gold_train)}")
    print(f"Final Train Silver Pairs: {len(silver_train)}")
    print(f"Final Test Golden Pairs: {len(clean_test_pairs)}")

    # Prepare final datasets
    # We'll create a combined training set for the user to try
    combined_train = gold_train + silver_train
    
    # Remove scores for final parquet compatibility if needed, though they can be useful
    def remove_score(dataset):
        return [{"translation": p["translation"]} for p in dataset]

    train_out = os.path.join(base_dir, "data", "processed", "train_combined.parquet")
    test_out = os.path.join(base_dir, "data", "processed", "test.parquet")
    
    Dataset.from_list(remove_score(combined_train)).to_parquet(train_out)
    Dataset.from_list(remove_score(clean_test_pairs)).to_parquet(test_out)
    print(f"Expanded parquets successfully written to {train_out}!")

if __name__ == "__main__":
    main()
