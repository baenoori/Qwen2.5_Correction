"""
LoRA Fine-tuning for Qwen2.5-0.5B-Instruct
오류 탐지 및 수정 태스크
"""

import torch
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


ERROR_TYPES = {
    1: "Arithmetic/Total Inconsistency",
    2: "Unit/Measurement Mismatch",
    3: "Temporal Order/Duration Inconsistency",
    4: "Spatial Relation Inconsistency",
    5: "Pronoun/Referent Ambiguity",
    6: "Action-Agent/Object Mismatch",
}


def create_prompt(sentence: str) -> str:
    return f"""You are an expert at detecting logical and semantic errors in text.
The following sentence contains exactly ONE error. Identify the error type and correct it.

## Error Types:
1. Arithmetic/Total Inconsistency - Numbers don't add up correctly
2. Unit/Measurement Mismatch - Wrong or inappropriate units for the context
3. Temporal Order/Duration Inconsistency - Events in impossible chronological order
4. Spatial Relation Inconsistency - Contradictory location/position statements
5. Pronoun/Referent Ambiguity - Pronouns that create logical contradictions or confusion
6. Action-Agent/Object Mismatch - Inanimate objects performing human/animate actions

## Input Sentence (contains ONE error):
"{sentence}"

## Task:
Identify the error type (1-6) and provide the corrected sentence. Respond ONLY with a valid JSON object:

```json
{{
  "has_error": true,
  "error_type": 1-6,
  "corrected_sentence": "corrected version with minimal edits"
}}
```"""


def create_response(error_type: int, corrected_sentence: str) -> str:
    return f'''```json
{{
  "has_error": true,
  "error_type": {error_type},
  "corrected_sentence": "{corrected_sentence}"
}}
```'''


def prepare_dataset(data_path: str) -> Dataset:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        prompt = create_prompt(item['sentence'])
        response = create_response(item['error_type'], item['ground_truth'])

        samples.append({
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": response}]
        })

    return Dataset.from_list(samples)


def main():
    # 경로 설정
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'train_data' / 'train_dataset.json'
    output_dir = project_root / 'lora_Qwen2.5' / 'checkpoints'
    merged_dir = project_root / 'lora_Qwen2.5' / 'merged_model'

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Model loaded. Device: {next(model.parameters()).device}")

    # 데이터셋 준비
    print(f"Loading dataset: {data_path}")
    train_dataset = prepare_dataset(str(data_path))
    train_dataset = train_dataset.shuffle(seed=42)
    print(f"Dataset size: {len(train_dataset)}")

    # LoRA 설정
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 학습 설정
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=25,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=50,
        bf16=True,
        gradient_checkpointing=False,
        completion_only_loss=True,
        packing=False,
        report_to="wandb",
        run_name="qwen2.5-0.5b-lora-v2",
    )

    # Trainer 생성 및 학습
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
    )
    trainer.train()

    # 모델 병합 및 저장
    print("Merging and saving model...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    print(f"Training complete!")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
