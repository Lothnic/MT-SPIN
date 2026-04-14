import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
import evaluate
import numpy as np

def main():
    print("Loading test dataset...")
    # Load our leakage-free test set (128 examples)
    test_df = load_dataset("parquet", data_files={"test": "data/processed_dataset/test.parquet"})["test"]
    
    print("Loading base model and tokenizer in bfloat16...")
    base_model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        src_lang="hin_Deva"
    )
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        device_map="auto",
        dtype=torch.bfloat16
    )
    
    # Load LoRA adapter
    adapter_path = "models/nllb-200-kangri-lora/final_adapter"
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print("Loading metrics...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    # Use LaBSE for neural semantic scoring
    sem_model = SentenceTransformer("sentence-transformers/LaBSE").to(device)
    
    preds = []
    refs = []
    sources = []
    
    print(f"Generating translations for {len(test_df)} examples...")
    for idx, row in enumerate(test_df):
        text_hin = row["translation"]["hin_Deva"]
        text_kng = row["translation"]["kng_Deva"]
        
        inputs = tokenizer(text_hin, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"),
                max_length=128
            )
            
        decoded_pred = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        preds.append(decoded_pred)
        refs.append(text_kng)
        sources.append(text_hin)
        
        # Verbose output for monitoring
        if idx % 20 == 0 or idx == len(test_df) - 1:
            print(f"\n[{idx+1}/{len(test_df)}]")
            print(f"Source (Hindi):  {text_hin}")
            print(f"Target (Kangri): {text_kng}")
            print(f"Predicted:       {decoded_pred}")
            
    print("\nCalculating metrics...")
    # BLEU/chrF expect references as a list of lists of strings
    bleu_refs = [[r] for r in refs]
    bleu_result = sacrebleu.compute(predictions=preds, references=bleu_refs)
    chrf_result = chrf.compute(predictions=preds, references=bleu_refs)
    
    # Calculate Neural Semantic Similarity (NSSS) via LaBSE
    print("Calculating LaBSE Semantic Scores...")
    src_emb = sem_model.encode(sources, convert_to_tensor=True, show_progress_bar=False)
    ref_emb = sem_model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
    pred_emb = sem_model.encode(preds, convert_to_tensor=True, show_progress_bar=False)
    
    # Similarity between Prediction and Reference (Main Semantic Score)
    ref_sim = util.cos_sim(pred_emb, ref_emb).diagonal().mean().item()
    # Similarity between Prediction and Source (Cross-lingual alignment)
    src_sim = util.cos_sim(pred_emb, src_emb).diagonal().mean().item()
    
    print("\n" + "="*30)
    print("      Evaluation Results     ")
    print("="*30)
    print(f"BLEU score:  {bleu_result['score']:.2f}")
    print(f"chrF score:  {chrf_result['score']:.2f}")
    print(f"NSSS (Ref):  {ref_sim:.4f}")
    print(f"NSSS (Src):  {src_sim:.4f}")
    print("="*30)
    
if __name__ == "__main__":
    main()
