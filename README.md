## 파일 설명

### 평가 데이터 (org_data)
| 파일 | 설명 |
|:-----|:-----|
| `eval_normal_data.json` | Normal 평가 데이터 - 짧은 문장, 단일 에러 (30개) |
| `eval_long_data.json` | Long 평가 데이터 - 긴 문장, 단일 에러 (12개) |
| `eval_hard_data.json` | Hard 평가 데이터 - 긴 문장, 2개 에러 (12개) |

### 학습 데이터 (train_data)
| 파일 | 설명 |
|:-----|:-----|
| `train_dataset.json` | LoRA 학습용 50개 샘플 |

### LoRA 스크립트 (lora_Qwen2.5/scripts)
| 파일 | 용도 |
|:-----|:-----|
| `LoRA_train.py` | Qwen2.5-0.5B LoRA 파인튜닝 |
| `LoRA_org_w_noerror.py` | Normal/Long 데이터 평가 (No Error 옵션 포함) |
| `LoRA_org_w.o_noerror.py` | Normal/Long 데이터 평가 (No Error 옵션 없음) |
| `LoRA_hard_w_noerror.py` | Hard 데이터 평가 (No Error 옵션 포함) |
| `LoRA_hard_w.o_noerror.py` | Hard 데이터 평가 (No Error 옵션 없음) |

### Vanilla 스크립트 (vanilla/scripts)
| 파일 | 용도 |
|:-----|:-----|
| `Qwen2.5_org_w_noerror.py` | Normal/Long 데이터 평가 (No Error 옵션 포함) |
| `Qwen2.5_org_w.o_noerror.py` | Normal/Long 데이터 평가 (No Error 옵션 없음) |
| `Qwen2.5_hard_w_noerror.py` | Hard 데이터 평가 (No Error 옵션 포함) |
| `Qwen2.5_hard_wo_noerror.py` | Hard 데이터 평가 (No Error 옵션 없음) |

### 평가 스크립트 (evaluation/scripts)
| 파일 | 용도 |
|:-----|:-----|
| `evaluate_normal.py` | response_data의 normal/long (w/wo) 결과 평가|
| `evaluate_hard.py` | response_data의 hard (w/wo) 결과 평가 |

## 오류 유형 (6가지)
1. Arithmetic/Total Inconsistency - 산술/총합 불일치 (T1)
2. Unit/Measurement Mismatch - 단위/측정 불일치 (T2)
3. Temporal Order/Duration Inconsistency - 시간 순서/기간 불일치 (T3)
4. Spatial Relation Inconsistency - 공간 관계 불일치 (T4)
5. Pronoun/Referent Ambiguity - 대명사/지시 대상 모호성 (T5)
6. Action-Agent/Object Mismatch - 행위-행위자/대상 불일치 (T6)

## 결과 비교
### 전체 결과 요약

| 모델 | Normal | Long | Hard | 전체 평균 |
|:-----|:-----:|:-----:|:-----:|:-----:|
| Vanilla 0.5B (w) | 0.000 | 0.000 | 0.000 | **0.000** |
| Vanilla 0.5B (wo) | 0.167 | 0.083 | 0.208 | **0.153** |
| Vanilla 7B (w) | 0.333 | 0.167 | 0.708 | **0.403** |
| Vanilla 7B (wo) | 0.767 | 0.667 | 0.917 | **0.783** |
| LoRA 0.5B (w) | 0.400 | 0.333 | 0.333 | **0.356** |
| LoRA 0.5B (wo) | 0.700 | 0.333 | 0.417 | **0.483** |

참고:
- w: No Error 옵션 포함 프롬프트
- wo: No Error 옵션 미포함 프롬프트
- Hard 데이터는 문장당 2개의 에러 타입이 있음
- 값이 '-'인 경우 해당 데이터셋에 그 에러 타입 샘플이 없음


### 주요 분석 결과
1. 전체 성능 순위
Vanilla 7B (wo): 가장 우수한 성능 (Normal: 0.767, Long: 0.667, Hard: 0.917)
LoRA 0.5B (wo): 두 번째 (Normal: 0.700)
Vanilla 7B (w): 세 번째
2. LoRA 파인튜닝 효과
Vanilla 0.5B → LoRA 0.5B 개선:
Normal (wo): 0.167 → 0.700 (+0.533, 4.2배 개선)
Normal (w): 0.000 → 0.400 (+0.400)
Hard (wo): 0.208 → 0.417 (+0.209, 2배 개선)
3. 프롬프트 옵션 비교 (w vs wo)
wo (No Error 옵션 없음) 가 전반적으로 더 높은 성능
모델이 항상 에러를 찾아야 하므로 집중도가 높아짐
4. 문제점
LoRA 모델의 Hard 데이터에서 일부 수정 내용이 입력과 무관한 학습 데이터 패턴 출력 (과적합 징후)
T5(대명사), T6(무생물 행동) 에러 탐지가 전반적으로 약함

## 모델별 에러 타입 탐지 정확도 비교표
### Normal 데이터 정확도

| 모델 | T1 | T2 | T3 | T4 | T5 | T6 | Avg |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Vanilla 0.5B (w) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| Vanilla 0.5B (wo) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | **0.167** |
| Vanilla 7B (w) | 0.200 | 0.200 | 0.400 | 0.400 | 0.200 | 0.600 | **0.333** |
| Vanilla 7B (wo) | 1.000 | 0.400 | 1.000 | 0.200 | 1.000 | 1.000 | **0.767** |
| LoRA 0.5B (w) | 1.000 | 0.000 | 0.800 | 0.400 | 0.200 | 0.000 | **0.400** |
| LoRA 0.5B (wo) | 1.000 | 1.000 | 0.600 | 0.600 | 0.000 | 1.000 | **0.700** |

### Long 데이터 정확도

| 모델 | T1 | T2 | T3 | T4 | T5 | T6 | Avg |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Vanilla 0.5B (w) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| Vanilla 0.5B (wo) | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.083** |
| Vanilla 7B (w) | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.500 | **0.167** |
| Vanilla 7B (wo) | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | **0.667** |
| LoRA 0.5B (w) | 1.000 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | **0.333** |
| LoRA 0.5B (wo) | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | **0.333** |

### Hard 데이터 정확도 (문장당 2개 에러)

| 모델 | T1 | T2 | T3 | T4 | T5 | T6 | Avg |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Vanilla 0.5B (w) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| Vanilla 0.5B (wo) | 1.000 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 | **0.208** |
| Vanilla 7B (w) | 0.250 | 1.000 | 0.750 | 1.000 | 0.750 | 0.500 | **0.708** |
| Vanilla 7B (wo) | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | **0.917** |
| LoRA 0.5B (w) | 0.500 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | **0.333** |
| LoRA 0.5B (wo) | 1.000 | 0.250 | 1.000 | 0.250 | 0.000 | 0.000 | **0.417** |

## Evaluation

### Accuracy (type / detection)

| Model | normal_w | normal_wo | long_w | long_wo | hard_w | hard_wo |
|:------|:--------:|:---------:|:------:|:-------:|:------:|:-------:|
| Vanilla 0.5B | 0.00 / 0.17 | 0.00 / - | 0.00 / 0.17 | 0.08 / - | 0.00 / 0.00 | 0.00 / - |
| Vanilla 7B | 0.23 / 0.37 | 0.60 / - | 0.08 / 0.25 | 0.50 / - | 0.36 / 0.92 | 0.58 / - |
| LoRA 0.5B | 0.40 / 0.83 | 0.53 / - | 0.33 / 0.83 | 0.17 / - | 0.33 / 1.00 | 0.17 / - |

### 평균 정성 점수 - Error Removal

| Model | normal_w | normal_wo | long_w | long_wo | hard_w | hard_wo |
|:------|:--------:|:---------:|:------:|:-------:|:------:|:-------:|
| Vanilla 0.5B | 0.97 | 0.70 | 1.00 | 1.00 | - | 0.42 |
| Vanilla 7B | 1.60 | 2.93 | 1.17 | 2.83 | 3.36 | 3.67 |
| LoRA 0.5B | 2.00 | 2.00 | 1.83 | 2.00 | 0.25 | 0.67 |

### 평균 정성 점수 - Consistency

| Model | normal_w | normal_wo | long_w | long_wo | hard_w | hard_wo |
|:------|:--------:|:---------:|:------:|:-------:|:------:|:-------:|
| Vanilla 0.5B | 3.73 | 3.47 | 3.33 | 3.08 | - | 2.67 |
| Vanilla 7B | 3.73 | 4.00 | 2.92 | 3.17 | 4.00 | 4.00 |
| LoRA 0.5B | 3.53 | 3.80 | 2.67 | 3.50 | 4.00 | 3.00 |

### 평균 정성 점수 - Minimality

| Model | normal_w | normal_wo | long_w | long_wo | hard_w | hard_wo |
|:------|:--------:|:---------:|:------:|:-------:|:------:|:-------:|
| Vanilla 0.5B | 4.00 | 3.67 | 4.00 | 2.92 | - | 2.08 |
| Vanilla 7B | 3.93 | 3.97 | 3.75 | 3.17 | 3.91 | 4.00 |
| LoRA 0.5B | 3.80 | 3.90 | 1.42 | 2.50 | 4.00 | 2.92 |

### BERTScore - Precision

| Model | normal_w | normal_wo | long_w | long_wo | hard_w | hard_wo |
|:------|:--------:|:---------:|:------:|:-------:|:------:|:-------:|
| Vanilla 0.5B | 0.994 | 0.980 | 0.992 | 0.959 | - | 0.933 |
| Vanilla 7B | 0.968 | 0.975 | 0.950 | 0.976 | 0.986 | 0.988 |
| LoRA 0.5B | 0.984 | 0.984 | 0.949 | 0.972 | 0.999 | 0.959 |

### BERTScore - Recall

| Model | normal_w | normal_wo | long_w | long_wo | hard_w | hard_wo |
|:------|:--------:|:---------:|:------:|:-------:|:------:|:-------:|
| Vanilla 0.5B | 0.995 | 0.977 | 0.988 | 0.941 | - | 0.926 |
| Vanilla 7B | 0.964 | 0.976 | 0.901 | 0.953 | 0.976 | 0.987 |
| LoRA 0.5B | 0.979 | 0.976 | 0.905 | 0.940 | 0.996 | 0.950 |

### BERTScore - F1

| Model | normal_w | normal_wo | long_w | long_wo | hard_w | hard_wo |
|:------|:--------:|:---------:|:------:|:-------:|:------:|:-------:|
| Vanilla 0.5B | 0.995 | 0.978 | 0.990 | 0.949 | - | 0.929 |
| Vanilla 7B | 0.966 | 0.976 | 0.924 | 0.964 | 0.981 | 0.988 |
| LoRA 0.5B | 0.981 | 0.980 | 0.926 | 0.955 | 0.998 | 0.954 |

# Qwen2.5_Correction Project Structure

```
Qwen2.5_Correction/
├── data/
│   ├── org_data/                          # 평가 데이터 (정답)
│   │   ├── eval_normal_data.json          # Normal 평가 데이터 (30개, 단일 에러)
│   │   ├── eval_long_data.json            # Long 평가 데이터 (12개, 단일 에러)
│   │   └── eval_hard_data.json            # Hard 평가 데이터 (12개, 2개 에러)
│   │
│   ├── response_data/                     # 모델 예측 결과
│   │   ├── vanilla0.5B/                   # Vanilla 0.5B 결과
│   │   │   ├── vanilla_0.5B_normal_w_noerror.json
│   │   │   ├── vanilla_0.5B_normal_wo_noerror.json
│   │   │   ├── vanilla_0.5B_long_w_noerror.json
│   │   │   ├── vanilla_0.5B_long_wo_noerror.json
│   │   │   ├── vanilla_0.5B_hard_w_noerror.json
│   │   │   └── vanilla_0.5B_hard_wo_noerror.json
│   │   │
│   │   ├── vanilla7B/                     # Vanilla 7B 결과
│   │   │   ├── vanilla_7B_normal_w_noerror.json
│   │   │   ├── vanilla_7B_normal_wo_noerror.json
│   │   │   ├── vanilla_7B_long_w_noerror.json
│   │   │   ├── vanilla_7B_long_wo_noerror.json
│   │   │   ├── vanilla_7B_hard_w_noerror.json
│   │   │   └── vanilla_7B_hard_wo_noerror.json
│   │   │
│   │   └── lora/                          # LoRA 0.5B 결과
│   │       ├── lora_normal_w_noerror.json
│   │       ├── lora_normal_wo_noerror.json
│   │       ├── lora_long_w_noerror.json
│   │       ├── lora_long_wo_noerror.json
│   │       ├── lora_hard_w_noerror.json
│   │       └── lora_hard_wo_noerror.json
│   │
│   ├── train_data/
│   │   └── train_dataset.json             # LoRA 학습 데이터 (50개)
│   │
│   └── eval_results/                      # 평가 결과 (type/detection acc, 정성 점수, BERTScore)
│       ├── vanilla0.5B/
│       │   ├── vanilla_0.5B_normal_w.json
│       │   ├── vanilla_0.5B_normal_wo.json
│       │   ├── vanilla_0.5B_long_w.json
│       │   ├── vanilla_0.5B_long_wo.json
│       │   ├── vanilla_0.5B_hard_w.json
│       │   └── vanilla_0.5B_hard_wo.json
│       ├── vanilla7B/
│       │   ├── vanilla_7B_normal_w.json
│       │   ├── vanilla_7B_normal_wo.json
│       │   ├── vanilla_7B_long_w.json
│       │   ├── vanilla_7B_long_wo.json
│       │   ├── vanilla_7B_hard_w.json
│       │   └── vanilla_7B_hard_wo.json
│       └── lora/
│           ├── lora_normal_w.json
│           ├── lora_normal_wo.json
│           ├── lora_long_w.json
│           ├── lora_long_wo.json
│           ├── lora_hard_w.json
│           └── lora_hard_wo.json
│
├── evaluation/
│   └── scripts/                           # 평가 스크립트
│       ├── evaluate_normal.py             # normal/long (w/wo) evaluation (type/detection acc, LLMasJudge, BERTScore)
│       └── evaluate_hard.py               # hard (w/wo) evaluation (type/detection acc, LLMasJudge, BERTScore)
│
├── lora_Qwen2.5/
│   ├── merged_model/                      # LoRA 학습 후 병합된 모델
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │   ├── special_tokens_map.json
│   │   └── added_tokens.json
│   │
│   ├── checkpoints/                       # 학습 체크포인트
│   │   ├── checkpoint-50/
│   │   ├── checkpoint-100/
│   │   ├── checkpoint-150/
│   │   └── checkpoint-175/
│   │
│   └── scripts/
│       ├── LoRA_train.py                  # LoRA 학습 스크립트
│       ├── LoRA_org_w_noerror.py          # Normal/Long 평가 (w/ No Error)
│       ├── LoRA_org_w.o_noerror.py        # Normal/Long 평가 (w/o No Error)
│       ├── LoRA_hard_w_noerror.py         # Hard 평가 (w/ No Error)
│       └── LoRA_hard_w.o_noerror.py       # Hard 평가 (w/o No Error)
│
├── vanilla/
│   └── scripts/
│       ├── Qwen2.5_org_w_noerror.py       # Normal/Long 평가 (w/ No Error)
│       ├── Qwen2.5_org_w.o_noerror.py     # Normal/Long 평가 (w/o No Error)
│       ├── Qwen2.5_hard_w_noerror.py      # Hard 평가 (w/ No Error)
│       └── Qwen2.5_hard_wo_noerror.py     # Hard 평가 (w/o No Error)
│
└── project_structure.md                   # 프로젝트 구조 문서
```
