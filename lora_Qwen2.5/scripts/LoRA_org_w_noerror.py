"""
LoRA Fine-tuned Qwen2.5-0.5B Evaluation (Normal/Long Data)
With No Error option
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


ERROR_TYPES = {
    0: "No Error",
    1: "Arithmetic/Total Inconsistency",
    2: "Unit/Measurement Mismatch",
    3: "Temporal Order/Duration Inconsistency",
    4: "Spatial Relation Inconsistency",
    5: "Pronoun/Referent Ambiguity",
    6: "Action-Agent/Object Mismatch",
}


@dataclass
class DetectionResult:
    sentence: str
    has_error: bool
    error_type: int
    explanation: str
    corrected_sentence: Optional[str]


class ContradictionDetector:
    def __init__(self, model_path: str, device: str = "auto"):
        print(f"Loading model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        device_map = "auto" if device == "auto" else {"": 0 if device == "cuda" else "cpu"}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded. Device: {self.device}")

    def _create_prompt(self, sentence: str) -> str:
        return f"""You are an expert at detecting logical and semantic errors in text.

## Error Types:
0. No Error - The sentence is logically correct

1. Arithmetic/Total Inconsistency - Numbers don't add up correctly
   Example:
   - "The report lists 100 items: 60 were approved and 50 were rejected." (60+50=110 ≠ 100)

2. Unit/Measurement Mismatch - Wrong or inappropriate units for the context
   Example:
   - "She is 165 liters tall." (height should use cm/meters, not liters)

3. Temporal Order/Duration Inconsistency - Events in impossible chronological order
   Example:
   - "He retired in 2016 and started his first job in 2019." (retired before starting work)

4. Spatial Relation Inconsistency - Contradictory location/position statements
   Example:
   - "The statue is in front of the museum, but it is also behind the museum."

5. Pronoun/Referent Ambiguity - Pronouns that don't match the gender or identity of the person
   Example:
   - "Jessica blamed himself for the mistake." (should be "herself" - Jessica is female)

6. Action-Agent/Object Mismatch - Inanimate objects performing human/animate actions
   Example:
   - "The lamp negotiated a new salary." (lamps cannot negotiate)

## Input Sentence:
"{sentence}"

## Task:
Analyze the sentence and respond ONLY with a valid JSON object:

```json
{{
  "has_error": true or false,
  "error_type": 1-6 if has_error is true, null if has_error is false,
  "explanation": "brief explanation in English",
  "corrected_sentence": "corrected version with minimal edits" or null if no error
}}
```"""

    def detect(self, sentence: str, max_new_tokens: int = 256, temperature: float = 0.1) -> DetectionResult:
        prompt = self._create_prompt(sentence)
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return self._parse_response(response, sentence)

    def _parse_response(self, response: str, sentence: str) -> DetectionResult:
        has_error = False
        error_type = 0
        explanation = ""
        corrected_sentence = None

        try:
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)

            if matches:
                json_str = max(matches, key=len)
                json_str = re.sub(r'\s+', ' ', json_str)
                parsed = json.loads(json_str)

                has_error = bool(parsed.get("has_error", False))
                raw_error_type = parsed.get("error_type")
                explanation = str(parsed.get("explanation", ""))
                corrected_sentence = parsed.get("corrected_sentence")

                if has_error:
                    if raw_error_type is not None:
                        error_type = int(raw_error_type)
                        if error_type < 1 or error_type > 6:
                            error_type = 1
                    else:
                        error_type = 1
                else:
                    error_type = 0

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            explanation = f"Parse error: {str(e)}"

        return DetectionResult(
            sentence=sentence,
            has_error=has_error,
            error_type=error_type,
            explanation=explanation,
            corrected_sentence=corrected_sentence
        )


def main():
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'lora_Qwen2.5' / 'merged_model'
    dataset_path = project_root / 'data' / 'org_data' / 'eval_normal_data.json'
    output_path = project_root / 'data' / 'response_data' / 'lora' / 'lora_normal_w_noerror.json'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    detector = ContradictionDetector(str(model_path))

    results = []
    for item in dataset:
        result = detector.detect(item['sentence'])
        results.append({
            'id': item['id'],
            'pred_has_error': result.has_error,
            'pred_error_type': result.error_type,
            'pred_error_type_name': ERROR_TYPES.get(result.error_type, "Unknown"),
            'sentence': item['sentence'],
            'pred_correction': result.corrected_sentence,
            'explanation': result.explanation
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
