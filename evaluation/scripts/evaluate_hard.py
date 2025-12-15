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

# hard 전용 정답 데이터
GROUND_TRUTH_PATH = DATA_DIR / "org_data" / "eval_hard_data.json"

# Evaluator prompt
EVALUATION_SYSTEM_PROMPT = """
You are an expert LLM evaluator (LLM Judge). Your task is to provide a *quantitative* evaluation of an AI's text correction that contains TWO errors.
You will receive the Error_text, the two ground truth (GT) error types, and the AI's corrected text.

Evaluate the AI's "Corrected Text" based ONLY on the following three criteria, providing a numeric score from 0 (worst) to 4 (best) for each.
**Please evaluate strictly according to the criteria.** A perfect score (4) should be reserved for flawless corrections.

1.  **Error_Removal_Score (Score 0-4):**
    * Measures whether *both* errors specified by `[Ground Truth Error Type 1]` AND `[Ground Truth Error Type 2]` were successfully and completely removed.
    * **4 (Excellent):** *Both* specified errors are 100% completely gone, and the correction itself is perfect.
    * **3 (Good):** *Both* specified errors are 100% gone, but the new phrasing introduces a *very minor* stylistic awkwardness (e.g., "unnatural" but not "wrong") without adding a new error.
    * **2 (Partial):** *Only one* of the two specified errors was fixed. The other error remains.
    * **1 (Poor):** An attempt was made, but *both* errors are still clearly present (or were only partially fixed).
    * **0 (Failed):** *Neither* of the two errors was fixed, or the text was made worse.

2.  **Consistency_Score (Score 0-4):**
    * Measures whether the `[AI's Corrected Text]` is free of *new* logical contradictions or significant grammatical errors.
    * **4 (Excellent):** The text is perfectly logical and introduces *no* new errors or contradictions whatsoever.
    * **3 (Good):** The text remains logical, but introduces a single, very minor new grammatical slip or typo (e.g., "a" vs "an", a missing comma) that does *not* affect the meaning.
    * **2 (Partial):** Introduces a noticeable new grammatical error (more than a simple typo) OR a minor logical inconsistency that wasn't there before.
    * **1 (Poor):** The correction introduces a significant new logical contradiction or a major grammatical error.
    * **0 (Failed):** The corrected text is now nonsensical, factually incorrect, or contains major new errors.

3.  **Minimality_Score (Score 0-4):**
    * Measures whether *only* the necessary changes were made to fix the errors.
    * **4 (Excellent):** Modified *only* the specific words/parts necessary to fix the two errors. All other valid sentences/parts are 100% untouched.
    * **3 (Good):** Modified *only* the sentences containing the errors. The changes are minimal, but perhaps slightly more than absolutely necessary (e.g., rewriting a small clause for fluency). No other sentences were touched.
    * **2 (Partial):** Rewrote the error-containing sentences much more than necessary, OR made a small, unnecessary change to a *different*, correct sentence.
    * **1 (Poor):** Made significant, unnecessary changes to multiple valid parts of the text (e.g., rewriting correct sentences).
    * **0 (Failed):** Completely rewrote large portions of the text, changing the meaning of valid sentences unnecessarily.

4.  **Judge_Reasoning (String):**
    * Provide a brief reasoning for your scores. Explain *why* you gave these scores, especially for scores below 4.

Provide your evaluation in a strict JSON format.

INPUT:
[Error_text]: {error_text}
[Ground Truth Error Type 1]: {ground_truth_type1}
[Ground Truth Error Type 2]: {ground_truth_type2}
[AI's Detected Error Type 1]: {detected_error_type1}
[AI's Detected Error Type 2]: {detected_error_type2}
[AI's Corrected Text]: {corrected_text}

OUTPUT FORMAT:
{
  "qualitative_scores": {
    "error_removal_score": (Integer 0-4),
    "consistency_score": (Integer 0-4),
    "minimality_score": (Integer 0-4)
  },
  "judge_reasoning": "(Your brief reasoning for the scores given.)"
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
            "error_types": [
                entry.get("error_type_1"),
                entry.get("error_type_2"),
            ],
            "error_type_names": [
                entry.get("error_type_1_name"),
                entry.get("error_type_2_name"),
            ],
            "ground_truth_text": gt_text,
        }
    print(f"Ground-truth lookup built for {len(lookup)} items.")
    return lookup


def normalize_prediction_item(
    item: Dict[str, Any]
) -> Tuple[
    Optional[int],
    str,
    List[Optional[str]],
    Optional[str],
    Optional[bool],
    List[Optional[int]],
]:
    """Extract shared fields from prediction entry."""
    item_id = item.get("id")
    error_text = item.get("error_text") or item.get("sentence") or ""
    detected_error_types = [
        item.get("detected_error_type_1")
        or item.get("pred_error_type_1_name"),
        item.get("detected_error_type_2")
        or item.get("pred_error_type_2_name"),
    ]
    detected_error_type_ids = [
        item.get("pred_error_type_1"),
        item.get("pred_error_type_2"),
    ]
    corrected_text = item.get("corrected_text") or item.get("pred_correction")
    has_error = item.get("pred_has_error")
    return (
        item_id,
        error_text,
        detected_error_types,
        corrected_text,
        has_error,
        detected_error_type_ids,
    )


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
    ground_truth_type1: str,
    ground_truth_type2: str,
    detected_error_type1: str,
    detected_error_type2: str,
    corrected_text: str,
) -> Dict[str, Any]:
    """2단계: 평가자 모델(GPT-4.1-mini)이 정성적 평가 수행."""
    print("[GPT-4.1-mini] Awaiting qualitative evaluation...")

    user_message = (
        f"[Error_text]: {error_text}\n"
        f"[Ground Truth Error Type 1]: {ground_truth_type1}\n"
        f"[Ground Truth Error Type 2]: {ground_truth_type2}\n"
        f"[AI's Detected Error Type 1]: {detected_error_type1}\n"
        f"[AI's Detected Error Type 2]: {detected_error_type2}\n"
        f"[AI's Corrected Text]: {corrected_text}"
    )

    messages = [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return get_json_response(openai_client, EVALUATOR_MODEL, messages)


def infer_dataset_variant(filename: str) -> Optional[Tuple[str, str]]:
    """파일명으로부터 hard + w/wo 조합만 추론."""
    lower = filename.lower()
    if "_hard_" not in lower:
        return None

    variant = None
    if "_wo_" in lower:
        variant = "wo"
    elif "_w_" in lower:
        variant = "w"

    if variant is None:
        return None
    return "hard", variant


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

        gt_error_type_names = [t for t in gt_info.get("error_type_names", []) if t]
        gt_error_type_ids = [t for t in gt_info.get("error_types", []) if t is not None]
        gt_text = gt_info.get("ground_truth_text") or gt_info.get("sentence")

        # 순서 무시 비교
        type_correct = None
        detected_names_set = {t for t in detected_error_type if t}
        gt_names_set = set(gt_error_type_names)
        if detected_names_set and gt_names_set:
            type_correct = detected_names_set == gt_names_set
            type_correct_flags.append(type_correct)

        detection_correct = None
        # hard: any non-zero error type means error 존재
        gt_has_error = any(t for t in gt_error_type_ids)
        if has_error is not None:
            detection_correct = has_error == gt_has_error
            detection_flags.append(detection_correct)

        # 정성 평가 수행
        qualitative_output: Dict[str, Any]
        if not detected_names_set:
            qualitative_output = {"error": "Skipped, missing detected error type."}
        else:
            gt_type1 = gt_error_type_names[0] if len(gt_error_type_names) > 0 else ""
            gt_type2 = gt_error_type_names[1] if len(gt_error_type_names) > 1 else ""
            det_type1 = detected_error_type[0] if len(detected_error_type) > 0 else ""
            det_type2 = detected_error_type[1] if len(detected_error_type) > 1 else ""
            qualitative_output = run_qualitative_evaluation(
                error_text=error_text,
                ground_truth_type1=gt_type1,
                ground_truth_type2=gt_type2,
                detected_error_type1=det_type1,
                detected_error_type2=det_type2,
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
                "ground_truth_error_types": gt_error_type_names,
                "ground_truth_error_type_ids": gt_error_type_ids,
                "pred_error_types": detected_error_type,
                "pred_error_type_ids": detected_error_type_id,
                "pred_has_error": has_error,
                "pred_correction": corrected_text,
                "type_correct": type_correct,
                "detection_correct": detection_correct,
                "qualitative_scores": qualitative_scores or None,
                "judge_reasoning": qualitative_output.get("judge_reasoning"),
            }
        )

    avg_qual_scores: Dict[str, float] = {}
    for key, values in qualitative_accumulator.items():
        avg_qual_scores[key] = mean(values) if values else 0.0

    metrics = {
        "total_items": len(items),
        "type_accuracy": mean(type_correct_flags) if type_correct_flags else None,
        "detection_accuracy": mean(detection_flags) if detection_flags else None,
        "average_qualitative_scores": avg_qual_scores,
    }

    bert_scores: Dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
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

    ground_truth_cache: Optional[Dict[int, Dict[str, Any]]] = None

    for model_dir in RESPONSE_DATA_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        for pred_path in model_dir.glob("*.json"):
            dataset_variant = infer_dataset_variant(pred_path.name)
            if not dataset_variant:
                continue

            _, variant = dataset_variant
            gt_path = GROUND_TRUTH_PATH
            if not gt_path.exists():
                print(f"Ground truth file for hard not found: {gt_path}")
                continue

            if ground_truth_cache is None:
                ground_truth_cache = build_ground_truth_lookup(
                    load_json_file(gt_path)
                )

            predictions = load_json_file(pred_path)
            evaluation_result = evaluate_predictions(
                predictions, ground_truth_cache
            )

            dataset_key = f"hard_{variant}"
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
