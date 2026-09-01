# 벡터화 방법 선택 가이드

## 개요

전처리 파이프라인에서 Word2Vec과 BERT 중 선택하여 리뷰 텍스트를 벡터화할 수 있습니다.

## 설정 방법

`src/preprocessing/main.py` 파일의 상단에서 `VECTORIZER_TYPE` 변수를 변경하세요:

```python
# ========== 벡터화 방법 설정 ==========
VECTORIZER_TYPE = "word2vec"  # 여기를 변경하여 선택
BERT_MODEL_NAME = "klue/bert-base"  # BERT 모델 이름
```

## 선택 옵션

### 1. Word2Vec (기본값)

```python
VECTORIZER_TYPE = "word2vec"
```

**특징:**

- ✅ **빠른 속도**: CPU에서도 빠르게 처리
- ✅ **낮은 메모리 사용량**: 경량 모델
- ✅ **100차원 벡터**: 컴팩트한 벡터 크기
- ⚠️ 문맥 이해가 제한적 (단어 단위)

**추천 상황:**

- 빠른 프로토타이핑이 필요할 때
- 컴퓨팅 리소스가 제한적일 때
- 대용량 데이터를 빠르게 처리해야 할 때

### 2. BERT

```python
VECTORIZER_TYPE = "bert"
```

**특징:**

- ✅ **높은 성능**: 문맥을 고려한 정교한 벡터화
- ✅ **768차원 벡터**: 풍부한 의미 표현
- ✅ **사전학습 모델**: 한국어에 특화된 모델 사용
- ⚠️ 느린 속도: GPU 권장 (CPU는 매우 느림)
- ⚠️ 높은 메모리 사용량: 최소 4GB 이상 권장

**추천 상황:**

- GPU를 사용할 수 있을 때
- 최고의 성능이 필요할 때
- 문맥 이해가 중요한 분석을 할 때

### 3. Both (둘 다 생성)

```python
VECTORIZER_TYPE = "both"
```

**특징:**

- word2vec과 bert 벡터를 모두 생성
- Parquet 파일에 두 컬럼이 모두 포함됨
- 나중에 선택적으로 사용 가능

**추천 상황:**

- 두 방법의 성능을 비교하고 싶을 때
- 나중에 어떤 방법을 사용할지 결정하고 싶을 때

## BERT 모델 선택

KLUE BERT를 기본으로 사용하지만, 다른 모델로 변경할 수 있습니다:

```python
BERT_MODEL_NAME = "klue/bert-base"        # KLUE BERT (추천)
# BERT_MODEL_NAME = "beomi/kcbert-base"   # KcBERT
# BERT_MODEL_NAME = "monologg/kobert"     # KoBERT
```

## 설치 방법

BERT를 사용하려면 추가 패키지 설치가 필요합니다:

### Conda 환경 업데이트

```bash
conda env update -f environment.yml
```

### 또는 pip로 직접 설치

```bash
pip install transformers torch
```

## 출력 데이터 구조

### Word2Vec 사용 시

```json
{
  "product_id": "선크림_with_1",
  "review_id": "12345",
  "word2vec": [0.123, -0.456, ...],  // 100차원
  "bert": null
}
```

### BERT 사용 시

```json
{
  "product_id": "선크림_with_1",
  "review_id": "12345",
  "word2vec": null,
  "bert": [0.123, -0.456, ...]  // 768차원
}
```

### Both 사용 시

```json
{
  "product_id": "선크림_with_1",
  "review_id": "12345",
  "word2vec": [0.123, -0.456, ...],  // 100차원
  "bert": [0.123, -0.456, ...]       // 768차원
}
```

## 성능 비교

| 항목        | Word2Vec   | BERT       |
| ----------- | ---------- | ---------- |
| 속도 (CPU)  | ⭐⭐⭐⭐⭐ | ⭐         |
| 속도 (GPU)  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     |
| 메모리 사용 | ⭐⭐⭐⭐⭐ | ⭐⭐       |
| 벡터 품질   | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ |
| 문맥 이해   | ⭐⭐       | ⭐⭐⭐⭐⭐ |

## 예상 처리 시간

1000개 리뷰 기준:

- **Word2Vec**: 약 1-2초
- **BERT (CPU)**: 약 10-20분
- **BERT (GPU)**: 약 30초-1분
- **Both**: Word2Vec + BERT 시간 합계

## 문제 해결

### BERT가 너무 느려요

- GPU를 사용하는 것을 권장합니다
- 또는 `VECTORIZER_TYPE = "word2vec"`으로 변경하세요

### Out of Memory 에러

- `batch_size`를 줄이세요 (기본값: 32)
- 또는 `VECTORIZER_TYPE = "word2vec"`으로 변경하세요

### GPU를 사용하고 싶어요

- PyTorch GPU 버전을 설치하세요:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- CUDA가 설치되어 있어야 합니다
