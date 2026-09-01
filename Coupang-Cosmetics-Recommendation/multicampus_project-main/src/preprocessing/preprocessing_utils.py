"""
전처리 파이프라인에서 사용되는 유틸리티 함수들
"""

import os
import re
import glob
import pickle
import unicodedata
import numpy as np
import pandas as pd
import sys
from io import StringIO

# 전역 토크나이저 변수
_tokenizer = None
_tokenizer_type = None


def init_tokenizer(tokenizer_type="kiwi"):
    """토크나이저 초기화 (Kiwi, Okt, Mecab 중 선택)"""
    global _tokenizer, _tokenizer_type

    if _tokenizer is not None and _tokenizer_type == tokenizer_type:
        return _tokenizer

    _tokenizer_type = tokenizer_type

    if tokenizer_type == "kiwi":
        from kiwipiepy import Kiwi

        # quantization 경고 억제
        _original_stderr = sys.stderr
        sys.stderr = StringIO()
        _tokenizer = Kiwi()
        sys.stderr = _original_stderr
        print(f"✓ Kiwi 토크나이저 초기화 완료")

    elif tokenizer_type == "okt":
        from konlpy.tag import Okt

        _tokenizer = Okt()
        print(f"✓ Okt 토크나이저 초기화 완료")

    elif tokenizer_type == "mecab":
        from konlpy.tag import Mecab

        _tokenizer = Mecab()
        print(f"✓ Mecab 토크나이저 초기화 완료")

    else:
        raise ValueError(
            f"지원하지 않는 토크나이저: {tokenizer_type}. 'kiwi', 'okt', 'mecab' 중 선택하세요."
        )

    return _tokenizer


def load_stopwords(filename="stopwords-ko.txt"):
    """불용어 로드"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        stopword_path = os.path.join(base_dir, filename)
        with open(stopword_path, "r", encoding="utf-8") as f:
            return set([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        print(f"\n[오류] 불용어 파일이 없습니다: {stopword_path}")
        print("작업을 중단합니다. 파일을 확인해주세요.")
        raise
    except Exception as e:
        print(f"\n[오류] 파일 읽기 중 알 수 없는 문제가 발생했습니다: {e}")
        raise


def get_tokens(text, stopwords):
    """텍스트를 토큰화 (선택된 토크나이저 사용)"""
    global _tokenizer, _tokenizer_type

    if _tokenizer is None:
        raise RuntimeError(
            "토크나이저가 초기화되지 않았습니다. init_tokenizer()를 먼저 호출하세요."
        )

    if not isinstance(text, str):
        return []
    clean_text = re.sub(r"[^가-힣0-9\s]", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    tokens = []

    if _tokenizer_type == "kiwi":
        # Kiwi: analyze() 결과에서 형태소 추출
        result = _tokenizer.analyze(clean_text)
        if result and len(result) > 0:
            for token in result[0][0]:
                word = token.form
                pos = token.tag
                # NNG(일반명사), NNP(고유명사), VV(동사), VA(형용사)
                if pos in ("NNG", "NNP", "VV", "VA") and word not in stopwords:
                    tokens.append(word)

    elif _tokenizer_type in ["okt", "mecab"]:
        # Okt, Mecab: pos() 메서드 사용
        for word, pos in _tokenizer.pos(clean_text):
            # NNG(일반명사), NNP(고유명사), VV(동사), VA(형용사)
            if pos in ("NNG", "NNP", "VV", "VA") and word not in stopwords:
                tokens.append(word)

    return tokens


def cosine_similarity(vec1, vec2):
    """두 벡터 간 코사인 유사도 계산"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


class TokenIterator:
    """Word2Vec 학습을 위한 토큰 Iterator (메모리 효율적)"""

    def __init__(self, token_dir):
        self.token_dir = token_dir
        self.token_files = glob.glob(os.path.join(token_dir, "*.pkl"))

    def __iter__(self):
        for token_file in self.token_files:
            try:
                with open(token_file, "rb") as f:
                    file_tokens = pickle.load(f)
                    for tokens in file_tokens:
                        if tokens:  # 빈 토큰 리스트는 제외
                            yield tokens
            except Exception as e:
                print(f"  [경고] 토큰 파일 읽기 실패: {token_file} - {e}")
                continue


# =========================
# Parquet 파일 로딩 함수
# =========================


def load_products_parquet(
    parquet_path="./data/processed_data/integrated_products_vector.parquet",
):
    """
    상품 벡터 Parquet 파일 로드

    Args:
        parquet_path: Parquet 파일 경로

    Returns:
        DataFrame: 상품 정보 (product_vector, representative_review_id 포함)
    """
    try:
        df = pd.read_parquet(parquet_path)
        return df
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {parquet_path}")
        return None
    except Exception as e:
        print(f"[오류] Parquet 파일 읽기 실패: {e}")
        return None


def load_reviews_parquet(
    parquet_path="./data/processed_data/integrated_reviews_detail.parquet",
    product_id=None,
):
    """
    리뷰 상세 Parquet 파일 로드 (필터링 옵션)

    Args:
        parquet_path: Parquet 파일 경로
        product_id: 특정 상품 ID로 필터링 (None이면 전체 로드)

    Returns:
        DataFrame: 리뷰 상세 정보 (tokens, word2vec 포함)

    Note:
        product_id는 이미 카테고리와 조합된 고유 ID입니다 (예: "로션_1")
    """
    try:
        df = pd.read_parquet(parquet_path)

        if product_id:
            # product_id를 NFC로 정규화하여 비교
            normalized_id = unicodedata.normalize("NFC", str(product_id))
            df = df[df["product_id"] == normalized_id]

        return df
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {parquet_path}")
        return None
    except Exception as e:
        print(f"[오류] Parquet 파일 읽기 실패: {e}")
        return None


def load_reviews_by_products(
    product_ids, parquet_path="./data/processed_data/integrated_reviews_detail.parquet"
):
    """
    여러 상품의 리뷰를 한 번에 로드

    Args:
        product_ids: 상품 ID 리스트
        parquet_path: Parquet 파일 경로

    Returns:
        DataFrame: 필터링된 리뷰 데이터
    """
    try:
        df = pd.read_parquet(parquet_path, filters=[("product_id", "in", product_ids)])
        return df
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {parquet_path}")
        return None
    except Exception as e:
        print(f"[오류] Parquet 파일 읽기 실패: {e}")
        return None


# 테스트
if __name__ == "__main__":
    products_df = load_products_parquet()
    if products_df is not None:
        print("상품 데이터 샘플:")
        print(products_df.head())
        print("\n")

    reviews_df = load_reviews_parquet(product_id="선스틱_1")
    if reviews_df is not None:
        print("리뷰 데이터 샘플:")
        print(reviews_df.head())

    reviews_dfs = load_reviews_by_products(product_ids=["선스틱_1", "선쿠션_선팩트_1"])
    if reviews_dfs is not None:
        print("여러 상품 리뷰 데이터 샘플:")
        print(reviews_dfs.head())
        # 처음꺼랑 마지막꺼
        # print(reviews_dfs.iloc[0])
        # print(reviews_dfs.iloc[-1])
