import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from bert_score import score

# 평가자 모델 - gpt 4.1-mini
EVALUATOR_MODEL = "gpt-4.1-mini"

# API KEY
try:
    openai_client = OpenAI(api_key="")
except TypeError:
    print("Error: OPENAI_API_KEY environment variable not set or invalid.")
    exit()

# path settings
CWD = Path(__file__).resolve().parent
PROJECT_ROOT = CWD.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESPONSE_DATA_DIR = DATA_DIR / "response_data"
EVAL_RESULTS_DIR = DATA_DIR / "eval_results"

# 사용할 정답 데이터 매핑
GROUND_TRUTH_MAP = {
    "normal": DATA_DIR / "org_data" / "eval_normal_data.json",
    "long": DATA_DIR / "org_data" / "eval_long_data.json",
}

# Evaluator prompt
EVALUATION_SYSTEM_PROMPT = """
You are an expert LLM evaluator (LLM Judge) known for your **strict and critical eye**. Your task is to provide a *quantitative* evaluation of an AI's text correction.
You will receive the Error_text (original text), the ground truth (GT) error type, and the AI's corrected text.

Evaluate the AI's "Corrected Text" based ONLY on the following three criteria.

**[CRITICAL INSTRUCTION - READ CAREFULLY]:**
1. **Global Consistency Check:** Do NOT just check the sentence where the error occurred. You MUST verify if the correction aligns with the *rest of the text*.
2. **The "Partial Fix" Trap:** A common failure is when the AI fixes a total sum in the last sentence but fails to update the individual numbers in the first sentence (or vice versa).
   - *Example:* Sentence 1 says "Total 20". Sentence 2 lists items summing to 25. Sentence 3 says "Total 25". -> **This is a CONTRADICTION.**
   - If this happens, **Consistency_Score must be 2 or lower.**
3. **No Generosity:** If the AI leaves a logical contradiction, do NOT give a perfect score.

---

### Scoring Criteria

1.  **Error_Removal_Score (Score 0-4):**
    * **Evaluation Focus:** Strictly measures whether the specific error defined in `[Ground Truth Error Type]` has been eliminated *without a trace*.
    * **4 (Excellent - Flawless):** The specified error is **absolutely and completely removed**. There is zero trace of the original issue.
    * **3 (Good - Stylistic):** The specified error is technically removed, but the resulting sentence structure is slightly unnatural (though grammatically correct).
    * **2 (Partial Fix / Incomplete):** The **major part** of the error is fixed, BUT the correction is **incomplete** or creates a mismatch with the immediate context. (e.g., Fixed the verb but ignored the subject-verb agreement implications).
    * **1 (Poor):** A weak attempt was made, but the error is still clearly present.
    * **0 (Failed):** The error was not fixed at all or was made worse.

2.  **Consistency_Score (Score 0-4):**
    * **Evaluation Focus:** Measures whether the `[AI's Corrected Text]` is free of *new* logical contradictions, numerical mismatches, or grammatical errors across the WHOLE text.
    * **4 (Excellent):** The text is perfectly logical throughout. All numbers, dates, and facts are internally consistent across all sentences.
    * **3 (Good):** The text remains logical, but introduces a single, very minor new grammatical slip or typo that does *not* affect the meaning.
    * **2 (Contradictory / Partial):** **[IMPORTANT]** The text contains **internal contradictions**. (e.g., Sentence 1 says "20 accidents" but Sentence 3 implies "25 accidents"). Or introduces a noticeable new grammatical error.
    * **1 (Poor):** The correction introduces a significant new logical contradiction making the text confusing, or a major grammatical error.
    * **0 (Failed):** The corrected text is nonsensical, factually incorrect, or contains critical new errors.

3.  **Minimality_Score (Score 0-4):**
    * **Evaluation Focus:** Measures whether *only* the necessary changes were made.
    * **4 (Excellent):** Modified *only* the specific words/parts necessary to fix the error. All other valid sentences are 100% untouched.
    * **3 (Good):** Modified *only* the sentence containing the error. Changes are minimal but slightly more than absolutely necessary.
    * **2 (Partial):** Rewrote the error-containing sentence excessively, OR made small unnecessary changes to *other* correct sentences.
    * **1 (Poor):** Made significant, unnecessary changes to multiple valid parts of the text.
    * **0 (Failed):** Completely rewrote large portions of the text unnecessarily.

4.  **Analysis & Reasoning:**
    * Before scoring, perform a **Fact Check**: List all numbers/facts in the text and check if they match.

---

Provide your evaluation in a strict JSON format.

INPUT:
[Error_text]: {error_text}
[Ground Truth Error Type]: {ground_truth_type}
[AI's Detected Error Type]: {detected_error_type}
[AI's Corrected Text]: {corrected_text}

OUTPUT FORMAT:
{
  "analysis_trace": "Briefly list the facts/numbers in the text to verify consistency (e.g., 'Sentence 1 says X, Sentence 2 implies Y...'). State if they match.",
  "qualitative_scores": {
    "error_removal_score": (Integer 0-4),
    "consistency_score": (Integer 0-4),
    "minimality_score": (Integer 0-4)
  },
  "judge_reasoning": "(Explain the scores based on the analysis_trace. Explicitly mention any contradictions found.)"
}
"""

def load_json_file(filename: Path) -> List[Dict[str, Any]]:
    """Loads JSON data from a file path."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file {filename} is not a valid JSON file.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading {filename}: {e}")
        return []


def build_ground_truth_lookup(dataset: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Build lookup table for GT info."""
    lookup: Dict[int, Dict[str, Any]] = {}
    for entry in dataset:
        entry_id = entry.get("id") or entry.get("ID")
        if entry_id is None:
            continue
        gt_text = entry.get("GT") or entry.get("ground_truth")
        lookup[entry_id] = {
            "sentence": entry.get("sentence") or entry.get("error_text"),
            "error_type": entry.get("error_type"),
            "error_type_name": entry.get("error_type_name") or str(entry.get("error_type")),
            "ground_truth_text": gt_text,
        }
    print(f"Ground-truth lookup built for {len(lookup)} items.")
    return lookup


def normalize_prediction_item(item: Dict[str, Any]) -> Tuple[Optional[int], str, Optional[str], Optional[str], Optional[bool], Optional[int]]:
    """Extract shared fields from prediction entry."""
    item_id = item.get("id")
    error_text = item.get("error_text") or item.get("sentence") or ""
    detected_error_type = (
        item.get("detected_error_type")
        or item.get("pred_error_type_name")
    )
    detected_error_type_id = item.get("pred_error_type")
    corrected_text = item.get("corrected_text") or item.get("pred_correction")
    has_error = item.get("pred_has_error")
    return item_id, error_text, detected_error_type, corrected_text, has_error, detected_error_type_id


def get_json_response(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Helper function to call API and get JSON response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        response_content = response.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        print(f"Error calling model {model}: {e}")
        return {"error": str(e)}


def run_qualitative_evaluation(
    error_text: str,
    ground_truth_type: str,
    detected_error_type: str,
    corrected_text: str,
) -> Dict[str, Any]:
    """2단계: 평가자 모델(GPT-4.1-mini)이 정성적 평가 수행."""
    print("[GPT-4.1-mini] Awaiting qualitative evaluation...")

    user_message = (
        f"[Error_text]: {error_text}\n"
        f"[Ground Truth Error Type]: {ground_truth_type}\n"
        f"[AI's Detected Error Type]: {detected_error_type}\n"
        f"[AI's Corrected Text]: {corrected_text}"
    )

    messages = [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return get_json_response(openai_client, EVALUATOR_MODEL, messages)


def infer_dataset_variant(filename: str) -> Optional[Tuple[str, str]]:
    """파일명으로부터 normal/long + w/wo 조합 추론."""
    lower = filename.lower()
    dataset = None
    if "_normal_" in lower:
        dataset = "normal"
    elif "_long_" in lower:
        dataset = "long"

    if dataset is None:
        return None

    variant = None
    if "_wo_" in lower:
        variant = "wo"
    elif "_w_" in lower:
        variant = "w"

    if variant is None:
        return None
    return dataset, variant


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth_lookup: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """예측값 리스트를 평가하고 정리된 결과를 반환."""
    items: List[Dict[str, Any]] = []
    type_correct_flags: List[bool] = []
    detection_flags: List[bool] = []
    bert_candidates: List[str] = []
    bert_references: List[str] = []
    qualitative_accumulator: Dict[str, List[float]] = {
        "error_removal_score": [],
        "consistency_score": [],
        "minimality_score": [],
    }

    for item in predictions:
        (
            item_id,
            error_text,
            detected_error_type,
            corrected_text,
            has_error,
            detected_error_type_id,
        ) = normalize_prediction_item(item)

        if item_id is None:
            print("Skipping item with missing ID.")
            continue

        gt_info = ground_truth_lookup.get(item_id)
        if not gt_info:
            print(f"Skipping item {item_id}: ground truth not found.")
            continue

        gt_error_type_name = gt_info.get("error_type_name")
        gt_error_type_id = gt_info.get("error_type")
        gt_text = gt_info.get("ground_truth_text") or gt_info.get("sentence")

        type_correct = None
        if detected_error_type:
            type_correct = detected_error_type == gt_error_type_name
            type_correct_flags.append(type_correct)

        detection_correct = None
        if has_error is not None and gt_error_type_id is not None:
            detection_correct = has_error == (gt_error_type_id != 0)
            detection_flags.append(detection_correct)

        # 정성 평가 수행
        qualitative_output: Dict[str, Any]
        if not detected_error_type:
            qualitative_output = {"error": "Skipped, missing detected error type."}
        else:
            qualitative_output = run_qualitative_evaluation(
                error_text=error_text,
                ground_truth_type=gt_error_type_name,
                detected_error_type=detected_error_type,
                corrected_text=corrected_text or error_text,
            )

        qualitative_scores = qualitative_output.get("qualitative_scores") or {}
        for key in qualitative_accumulator.keys():
            if isinstance(qualitative_scores.get(key), (int, float)):
                qualitative_accumulator[key].append(qualitative_scores[key])

        if corrected_text and gt_text:
            bert_candidates.append(corrected_text)
            bert_references.append(gt_text)

        items.append(
            {
                "id": item_id,
                "sentence": error_text,
                "ground_truth_error_type": gt_error_type_name,
                "ground_truth_error_type_id": gt_error_type_id,
                "pred_error_type": detected_error_type,
                "pred_error_type_id": detected_error_type_id,
                "pred_has_error": has_error,
                "pred_correction": corrected_text,
                "type_correct": type_correct,
                "detection_correct": detection_correct,
                "qualitative_scores": qualitative_scores or None,
                "judge_reasoning": qualitative_output.get("judge_reasoning"),
            }
        )

    metrics = {
        "total_items": len(items),
        "type_accuracy": mean(type_correct_flags) if type_correct_flags else None,
        "detection_accuracy": mean(detection_flags) if detection_flags else None,
        "average_qualitative_scores": {
            key: mean(values) for key, values in qualitative_accumulator.items() if values
        },
    }

    bert_scores: Optional[Dict[str, float]] = None
    if bert_candidates and bert_references:
        try:
            P, R, F = score(bert_candidates, bert_references, lang="en")
            bert_scores = {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F.mean().item(),
            }
        except Exception as e:
            print(f"Could not calculate BERTScore: {e}")

    metrics["bertscore"] = bert_scores

    return {"metrics": metrics, "items": items}


def save_results(output_path: Path, payload: Dict[str, Any]) -> None:
    """결과를 JSON 파일로 저장."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation to {output_path}")


def main() -> None:
    print(f"Evaluator Model (Qualitative Judge): {EVALUATOR_MODEL}\n")

    ground_truth_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for model_dir in RESPONSE_DATA_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        for pred_path in model_dir.glob("*.json"):
            dataset_variant = infer_dataset_variant(pred_path.name)
            if not dataset_variant:
                continue

            dataset, variant = dataset_variant
            gt_path = GROUND_TRUTH_MAP.get(dataset)
            if not gt_path or not gt_path.exists():
                print(f"Ground truth file for {dataset} not found: {gt_path}")
                continue

            if dataset not in ground_truth_cache:
                ground_truth_cache[dataset] = build_ground_truth_lookup(
                    load_json_file(gt_path)
                )

            predictions = load_json_file(pred_path)
            evaluation_result = evaluate_predictions(
                predictions, ground_truth_cache[dataset]
            )

            dataset_key = f"{dataset}_{variant}"
            payload = {
                "model": model_dir.name,
                "dataset": dataset_key,
                "source_file": pred_path.name,
                **evaluation_result,
            }

            output_path = EVAL_RESULTS_DIR / model_dir.name / f"{dataset_key}.json"
            save_results(output_path, payload)


if __name__ == "__main__":
    main()
