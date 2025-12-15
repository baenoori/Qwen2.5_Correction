"""
Contradiction Detection for Hard Data (TWO errors per sentence)
Using Qwen2.5-0.5B-Instruct with No Error option
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from dataclasses import dataclass
from typing import Optional, List
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
    error_type_1: Optional[int]
    error_type_2: Optional[int]
    explanation: str
    corrected_sentence: Optional[str]


class ContradictionDetector:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto"):
        print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        device_map = "auto" if device == "auto" else {"": 0 if device == "cuda" else "cpu"}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded. Device: {self.device}")

    def _create_prompt(self, sentence: str) -> str:
        return f"""You are an expert at detecting logical and semantic errors in text.
The following sentence may contain exactly TWO errors. Identify both error types and correct them.

## Error Types:
0. No Error - The sentence is logically correct
1. Arithmetic/Total Inconsistency - Numbers don't add up correctly
2. Unit/Measurement Mismatch - Wrong or inappropriate units for the context
3. Temporal Order/Duration Inconsistency - Events in impossible chronological order
4. Spatial Relation Inconsistency - Contradictory location/position statements
5. Pronoun/Referent Ambiguity - Pronouns that don't match the gender or identity of the person
6. Action-Agent/Object Mismatch - Inanimate objects performing human/animate actions

## Example 1 (Arithmetic + Unit):
Input: "The warehouse report says inventory totals 500 items: 280 in section A and 250 in section B. It also notes the storage temperature is maintained at 15 kilometers."
Output: {{"has_error": true, "error_type_1": 1, "error_type_2": 2, "explanation": "280+250=530 not 500, temperature should be in Celsius not kilometers", "corrected_sentence": "The warehouse report says inventory totals 530 items: 280 in section A and 250 in section B. It also notes the storage temperature is maintained at 15 degrees Celsius."}}

## Example 2 (Temporal + Spatial):
Input: "The announcement says the store opened in 2020 and was founded in 2022. It also says the store is located inside the mall but also across the street from the mall."
Output: {{"has_error": true, "error_type_1": 3, "error_type_2": 4, "explanation": "Store cannot open before being founded, and cannot be both inside and across from the mall", "corrected_sentence": "The announcement says the store was founded in 2020 and opened in 2022. It also says the store is located inside the mall."}}

## Example 3 (Pronoun + Action-Agent):
Input: "Emily finished her project early. He then submitted it to the manager. The desk reviewed the submission and approved it immediately."
Output: {{"has_error": true, "error_type_1": 5, "error_type_2": 6, "explanation": "Emily is female so should use 'she' not 'he', and a desk cannot review submissions", "corrected_sentence": "Emily finished her project early. She then submitted it to the manager. The manager reviewed the submission and approved it immediately."}}

## Input Sentence (may contain TWO errors):
"{sentence}"

## Task:
Determine if the sentence has errors. If it does, identify both error types (1-6) and provide the corrected sentence. If no errors exist, set has_error to false. Respond ONLY with a valid JSON object:

```json
{{
  "has_error": true or false,
  "error_type_1": 0-6 (first error type, 0 if no error),
  "error_type_2": 0-6 (second error type, 0 if no error),
  "explanation": "brief explanation of errors or why no error exists",
  "corrected_sentence": "corrected version or null if no error"
}}
```"""

    def detect(self, sentence: str, max_new_tokens: int = 512, temperature: float = 0.1) -> DetectionResult:
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
        error_type_1 = None
        error_type_2 = None
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
                explanation = str(parsed.get("explanation", ""))
                corrected_sentence = parsed.get("corrected_sentence")

                if has_error:
                    raw_type_1 = parsed.get("error_type_1")
                    raw_type_2 = parsed.get("error_type_2")

                    if raw_type_1 is not None:
                        error_type_1 = int(raw_type_1)
                        if error_type_1 < 1 or error_type_1 > 6:
                            error_type_1 = 1

                    if raw_type_2 is not None:
                        error_type_2 = int(raw_type_2)
                        if error_type_2 < 1 or error_type_2 > 6:
                            error_type_2 = 1

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            explanation = f"Parse error: {str(e)}"

        return DetectionResult(
            sentence=sentence,
            has_error=has_error,
            error_type_1=error_type_1,
            error_type_2=error_type_2,
            explanation=explanation,
            corrected_sentence=corrected_sentence
        )


def main():
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / 'data' / 'org_data' / 'eval_hard_data.json'
    output_path = project_root / 'data' / 'response_data' / 'vanilla7B' / 'vanilla_7B_hard_w_noerror.json'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    detector = ContradictionDetector()

    results = []
    for item in dataset:
        result = detector.detect(item['sentence'])
        results.append({
            'id': item['id'],
            'pred_has_error': result.has_error,
            'pred_error_type_1': result.error_type_1,
            'pred_error_type_1_name': ERROR_TYPES.get(result.error_type_1, "Unknown") if result.error_type_1 else None,
            'pred_error_type_2': result.error_type_2,
            'pred_error_type_2_name': ERROR_TYPES.get(result.error_type_2, "Unknown") if result.error_type_2 else None,
            'sentence': item['sentence'],
            'pred_correction': result.corrected_sentence,
            'explanation': result.explanation
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
