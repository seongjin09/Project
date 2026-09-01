"""
Hugging Face Hub 모델 다운로드 기반 벡터화 모듈
- HF Hub에서 모델을 자동 다운로드하여 로컬 추론
- API 엔드포인트 의존 없음 (410/404 에러 원천 차단)
- 로컬/Streamlit Cloud 모두 동일하게 동작
"""

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Optional
import os


class HuggingFaceAPIVectorizer:
    """
    Hugging Face Hub에서 모델을 다운로드하여 로컬 추론하는 벡터화 클래스

    - API 호출 대신 모델 파일을 직접 다운로드 후 추론
    - torch + transformers 사용 (BERTVectorizer와 동일한 추론 방식)
    - 최초 1회만 다운로드, 이후 HF 캐시에서 로드
    """

    def __init__(
        self,
        model_id: str = "fullfish/multicampus_semantic",
        api_token: Optional[str] = None,
    ):
        """
        Args:
            model_id: Hugging Face 모델 ID (예: "fullfish/multicampus_semantic")
            api_token: Hugging Face API 토큰 (private 모델일 경우 필요)
        """
        self.model_id = model_id
        self.api_token = api_token or os.getenv("HF_TOKEN")

        print(f"🔄 Hugging Face Hub에서 모델 다운로드 중: {model_id}")

        # HF Hub에서 모델 + 토크나이저 다운로드 (캐시됨)
        token = self.api_token if self.api_token else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModel.from_pretrained(model_id, token=token)
        self.model.eval()

        # CPU 사용 (Streamlit Cloud는 GPU 없음)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.device.type == "cuda":
            self.model.half()

        print(f"✓ 모델 로딩 완료")
        print(f"  - Model: {model_id}")
        print(f"  - Device: {self.device}")
        print(f"  - Hidden Size: {self.model.config.hidden_size}")

    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        단일 텍스트를 벡터로 변환 (Mean Pooling)

        Args:
            text: 입력 텍스트
            max_length: 최대 토큰 길이

        Returns:
            768차원 벡터 (모델에 따라 다를 수 있음)
        """
        if not text or not text.strip():
            return np.zeros(self.model.config.hidden_size)

        # 토큰화
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        # 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        # Mean Pooling (attention_mask 고려)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        mean_embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
        return mean_embedding

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        max_length: int = 512,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        여러 텍스트를 배치로 벡터화

        Args:
            texts: 입력 텍스트 리스트
            batch_size: 배치 크기
            max_length: 최대 토큰 길이
            show_progress: 진행상황 표시 여부

        Returns:
            (len(texts), hidden_size) 크기의 numpy 배열
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            if show_progress:
                print(f"처리 중: {i}/{len(texts)}")

            # 빈 텍스트 처리
            processed = [t if t and t.strip() else " " for t in batch]

            inputs = self.tokenizer(
                processed,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)

            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state

            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def get_vector_size(self) -> int:
        """벡터 차원 반환"""
        return self.model.config.hidden_size


# 싱글톤 인스턴스 캐싱
_hf_vectorizer_instance = None


def get_hf_api_vectorizer(
    model_id: str = "fullfish/multicampus_semantic",
    api_token: Optional[str] = None,
) -> HuggingFaceAPIVectorizer:
    """
    HuggingFaceAPIVectorizer 싱글톤 인스턴스 반환

    Args:
        model_id: Hugging Face 모델 ID
        api_token: Hugging Face API 토큰 (private 모델일 경우 필요)

    Returns:
        HuggingFaceAPIVectorizer 인스턴스
    """
    global _hf_vectorizer_instance

    if _hf_vectorizer_instance is None:
        _hf_vectorizer_instance = HuggingFaceAPIVectorizer(
            model_id=model_id, api_token=api_token
        )

    return _hf_vectorizer_instance
