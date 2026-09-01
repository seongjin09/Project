"""
BERT 기반 벡터화 모듈
"""

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List


class BERTVectorizer:
    """
    한국어 BERT 모델을 사용한 벡터화 클래스
    """

    def __init__(self, model_name: str = "klue/bert-base"):
        """
        Args:
            model_name: 사용할 BERT 모델 이름
                - "klue/bert-base": KLUE BERT (추천)
                - "beomi/kcbert-base": KcBERT
                - "monologg/kobert": KoBERT
        """
        print(f"BERT 모델 로딩 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # 평가 모드

        # GPU 사용 가능하면 GPU 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # GPU에서 FP16 사용 (메모리 절약 + 속도 향상)
        if self.device.type == "cuda":
            self.model.half()

            # GPU 종류별 최적 배치 사이즈 설정
            gpu_name = torch.cuda.get_device_name(0)
            if "A100" in gpu_name:
                self.default_batch_size = 64
            elif "T4" in gpu_name:
                self.default_batch_size = 32
            else:
                self.default_batch_size = 16

            print(f"✓ BERT 모델 로딩 완료")
            print(f"  - Device: {gpu_name}")
            print(f"  - Precision: FP16")
            print(f"  - Default Batch Size: {self.default_batch_size}")
        else:
            self.default_batch_size = 32
            print(f"✓ BERT 모델 로딩 완료")
            print(f"  - Device: CPU")
            print(f"  - Precision: FP32")
            print(f"  - Default Batch Size: {self.default_batch_size}")

    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        단일 텍스트를 BERT 벡터로 변환 (Mean Pooling 사용)

        Args:
            text: 입력 텍스트
            max_length: 최대 토큰 길이

        Returns:
            768차원 벡터 (BERT base 모델의 경우)
        """
        if not text or not text.strip():
            # 빈 텍스트는 zero 벡터 반환
            return np.zeros(768)

        # 토큰화 및 인코딩
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        # GPU로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 모델 추론
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        # Mean Pooling: attention_mask를 고려한 평균 (SimCSE 학습 방식과 일치)
        attention_mask = inputs["attention_mask"]
        token_embeddings = (
            outputs.last_hidden_state
        )  # (batch_size, seq_len, hidden_size)

        # attention_mask를 확장하여 hidden_size 차원에 맞춤
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # 마스크된 토큰들의 합계
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        # 마스크된 토큰 개수 (0으로 나누기 방지)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Mean Pooling
        mean_embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]

        return mean_embedding

    def encode_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = None,
        use_dynamic_batching: bool = True,
    ) -> List[np.ndarray]:
        """
        여러 텍스트를 배치로 벡터화 (효율적)

        Args:
            texts: 텍스트 리스트
            max_length: 최대 토큰 길이
            batch_size: 배치 크기 (None이면 자동 설정)
            use_dynamic_batching: True면 길이별로 정렬 후 배치 (패딩 최소화)

        Returns:
            벡터 리스트
        """
        # batch_size 자동 설정
        if batch_size is None:
            batch_size = self.default_batch_size

        # Dynamic batching: 길이가 비슷한 문장끼리 묶음
        if use_dynamic_batching and len(texts) > batch_size:
            # 원본 순서 보존을 위한 인덱스
            text_with_idx = [(i, t) for i, t in enumerate(texts)]
            # 길이순 정렬 (짧은 것부터)
            text_with_idx.sort(key=lambda x: len(x[1]))
            sorted_indices = [idx for idx, _ in text_with_idx]
            sorted_texts = [text for _, text in text_with_idx]
        else:
            sorted_indices = list(range(len(texts)))
            sorted_texts = texts

        vectors = []

        for i in range(0, len(sorted_texts), batch_size):
            batch_texts = sorted_texts[i : i + batch_size]

            # 빈 텍스트 처리
            processed_texts = [t if t and t.strip() else " " for t in batch_texts]

            # 토큰화 (배치 내 최장 문장에만 맞춤)
            inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,  # 배치 내 최장 길이로 패딩 (이제 비슷한 길이끼리 묶임)
            )

            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 모델 추론
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)

            # Mean Pooling: attention_mask를 고려한 평균 (SimCSE 학습 방식과 일치)
            attention_mask = inputs["attention_mask"]
            token_embeddings = (
                outputs.last_hidden_state
            )  # (batch_size, seq_len, hidden_size)

            # attention_mask를 확장하여 hidden_size 차원에 맞춤
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )

            # 마스크된 토큰들의 합계
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            # 마스크된 토큰 개수 (0으로 나누기 방지)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

            # Mean Pooling
            batch_vectors = (sum_embeddings / sum_mask).cpu().numpy()
            vectors.extend(batch_vectors)

        # 원본 순서로 복원
        if use_dynamic_batching and len(texts) > batch_size:
            # 정렬된 순서 → 원본 순서로 재배열
            original_order_vectors = [None] * len(vectors)
            for sorted_idx, original_idx in enumerate(sorted_indices):
                original_order_vectors[original_idx] = vectors[sorted_idx]
            return original_order_vectors

        return vectors

    def get_vector_size(self) -> int:
        """벡터 차원 반환"""
        return self.model.config.hidden_size


# 모델별 인스턴스 캐시 (메모리 효율성)
_bert_vectorizer_instances = {}


def get_bert_vectorizer(model_name: str = "klue/bert-base") -> BERTVectorizer:
    """
    BERT Vectorizer 인스턴스 반환 (모델별로 캐시)
    """
    global _bert_vectorizer_instances

    if model_name not in _bert_vectorizer_instances:
        _bert_vectorizer_instances[model_name] = BERTVectorizer(model_name)

    return _bert_vectorizer_instances[model_name]
