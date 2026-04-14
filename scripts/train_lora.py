import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

def main():
    base_model_name = "facebook/nllb-200-distilled-600M"
    # Using hin_Deva for both as a proxy, per prior discussions
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, src_lang="hin_Deva", tgt_lang="hin_Deva")

    print("Loading datasets...")
    train_file = "data/processed_dataset/train_combined.parquet"
    test_file = "data/processed_dataset/test.parquet"
    dataset = load_dataset("parquet", data_files={"train": train_file, "test": test_file})

    def preprocess_function(examples):
        inputs = [ex["hin_Deva"] for ex in examples["translation"]]
        targets = [ex["kng_Deva"] for ex in examples["translation"]]
        
        model_inputs = tokenizer(text=inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["translation"])

    print("Loading base model in bfloat16...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    output_dir = "models/nllb-200-kangri-lora"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        bf16=True,
        push_to_hub=False,
        logging_steps=100,
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving adapter model...")
    trainer.model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))

if __name__ == "__main__":
    main()
