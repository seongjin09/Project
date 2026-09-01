"""
감성 분석 모델 유틸리티 함수

단일 텍스트 또는 배치 텍스트에 대한 감성 점수 예측
"""

import sys
import os
import re
import joblib
import numpy as np
from konlpy.tag import Okt
from gensim.models import Word2Vec

# 형태소 분석기 초기화
okt = Okt()

# 전역 변수 (캐싱용)
_model_word2vec = None
_model_bert = None
_stopwords = None
_w2v_model = None
_bert_vectorizer = None


def load_stopwords():
    """불용어 로드"""
    stopwords_path = os.path.join(
        os.path.dirname(__file__), "../preprocessing/stopwords-ko.txt"
    )
    if not os.path.exists(stopwords_path):
        raise FileNotFoundError(f"불용어 파일을 찾을 수 없습니다: {stopwords_path}")

    with open(stopwords_path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f if line.strip()])


def load_model(vectorizer_type="word2vec"):
    """감성 분석 모델 로드

    Args:
        vectorizer_type (str): "word2vec" 또는 "bert"
    """
    model_filename = f"logistic_regression_sentiment_{vectorizer_type}.joblib"
    model_path = os.path.join(os.path.dirname(__file__), "../../models", model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    return joblib.load(model_path)


def load_word2vec_model():
    """Word2Vec 모델 로드"""
    model_path = os.path.join(
        os.path.dirname(__file__), "../../models/word2vec_model.model"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Word2Vec 모델 파일을 찾을 수 없습니다: {model_path}")

    return Word2Vec.load(model_path)


def load_bert_vectorizer():
    """BERT Vectorizer 로드"""
    import sys

    bert_path = os.path.join(os.path.dirname(__file__), "../preprocessing")
    if bert_path not in sys.path:
        sys.path.insert(0, bert_path)

    from bert_vectorizer import get_bert_vectorizer

    return get_bert_vectorizer("klue/bert-base")


def initialize(vectorizer_type="word2vec"):
    """모델, 불용어, 벡터화 모델을 메모리에 로드 (한 번만)

    Args:
        vectorizer_type (str): "word2vec" 또는 "bert"
    """
    global _model_word2vec, _model_bert, _stopwords, _w2v_model, _bert_vectorizer

    if vectorizer_type == "word2vec":
        if _model_word2vec is None:
            _model_word2vec = load_model("word2vec")
            print("✓ Word2Vec 감성 분석 모델 로드 완료")

        if _w2v_model is None:
            _w2v_model = load_word2vec_model()
            print(f"✓ Word2Vec 벡터화 모델 로드 완료")
            print(f"  - 어휘 크기: {len(_w2v_model.wv):,}개")
            print(f"  - 벡터 차원: {_w2v_model.vector_size}차원")

    elif vectorizer_type == "bert":
        if _model_bert is None:
            _model_bert = load_model("bert")
            print("✓ BERT 감성 분석 모델 로드 완료")

        if _bert_vectorizer is None:
            _bert_vectorizer = load_bert_vectorizer()
            print(f"✓ BERT 벡터화 모델 로드 완료")

    if _stopwords is None:
        _stopwords = load_stopwords()
        print(f"✓ 불용어 로드 완료: {len(_stopwords)}개")


def tokenize_text(text):
    """텍스트를 토큰화 (전처리 파이프라인과 동일한 방식)"""
    if _stopwords is None:
        initialize("word2vec")  # 토큰화만 필요한 경우 word2vec 초기화

    if not isinstance(text, str) or not text.strip():
        return []

    # 특수문자 제거, 한글/숫자만 유지
    clean_text = re.sub(r"[^가-힣0-9\s]", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # 형태소 분석 (명사, 동사, 형용사만)
    tokens = []
    for word, pos in okt.pos(clean_text, stem=True):
        if pos in ("Noun", "Verb", "Adjective") and word not in _stopwords:
            tokens.append(word)

    return tokens


def tokens_to_vector(tokens):
    """토큰을 Word2Vec 벡터로 변환 (평균)"""
    if _w2v_model is None:
        initialize()

    if not tokens:
        return np.zeros(_w2v_model.vector_size)

    # Word2Vec 모델에서 각 토큰의 벡터를 가져옴
    valid_vectors = []
    for token in tokens:
        if token in _w2v_model.wv:
            valid_vectors.append(_w2v_model.wv[token])

    if not valid_vectors:
        # 벡터가 없는 경우 0 벡터 반환
        return np.zeros(_w2v_model.vector_size)

    # 평균 벡터 반환
    return np.mean(valid_vectors, axis=0)


def predict_sentiment(text, verbose=False, vectorizer_type="word2vec"):
    """
    텍스트의 감성 점수 예측

    Args:
        text (str): 분석할 텍스트
        verbose (bool): 상세 정보 출력 여부
        vectorizer_type (str): "word2vec" 또는 "bert"

    Returns:
        float: 긍정 확률 (0.0 ~ 1.0)
               1.0에 가까울수록 긍정적

    Examples:
        >>> score = predict_sentiment("이 제품 정말 좋아요!")
        >>> print(f"감성 점수: {score:.3f}")
    """
    global _model_word2vec, _model_bert, _w2v_model, _bert_vectorizer

    # 초기화
    if vectorizer_type == "word2vec":
        if _model_word2vec is None or _w2v_model is None:
            initialize("word2vec")
        model = _model_word2vec
    else:
        if _model_bert is None or _bert_vectorizer is None:
            initialize("bert")
        model = _model_bert

    if verbose:
        print(f"\n[사용 모델: {vectorizer_type.upper()}]")
        print(f"원문: {text}")

    # 벡터화
    if vectorizer_type == "word2vec":
        # 1. 토큰화
        tokens = tokenize_text(text)

        if verbose:
            print(f"토큰: {tokens}")

        if not tokens:
            if verbose:
                print("⚠️ 유효한 토큰이 없습니다. 중립 점수 반환")
            return 0.5  # 중립

        # 2. Word2Vec 벡터화
        vector = tokens_to_vector(tokens)
    else:
        # BERT 벡터화 (토큰화 불필요, 원문 그대로 사용)
        vector = _bert_vectorizer.encode(text)

        if verbose:
            print(f"BERT 벡터 차원: {len(vector)}")

    # 3. 감성 점수 예측
    proba = model.predict_proba([vector])[0][1]

    if verbose:
        print(f"감성 점수: {proba:.4f}")
        sentiment_label = "긍정" if proba >= 0.6 else "부정" if proba <= 0.4 else "중립"
        print(f"판정: {sentiment_label}")

    return proba


def batch_predict(texts, show_progress=False, vectorizer_type="word2vec"):
    """
    여러 텍스트를 배치로 예측

    Args:
        texts (list): 텍스트 리스트
        show_progress (bool): 진행률 표시 여부
        vectorizer_type (str): "word2vec" 또는 "bert"

    Returns:
        list: 감성 점수 리스트

    Examples:
        >>> texts = ["좋아요", "별로예요", "그냥 그래요"]
        >>> scores = batch_predict(texts)
    """
    # 초기화
    initialize(vectorizer_type)

    results = []
    iterator = enumerate(texts)

    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(texts), desc="감성 분석")
        except ImportError:
            pass

    for i, text in iterator:
        score = predict_sentiment(text, verbose=False, vectorizer_type=vectorizer_type)
        results.append(score)

    return results


def get_sentiment_label(score, positive_threshold=0.6, negative_threshold=0.4):
    """
    감성 점수를 레이블로 변환

    Args:
        score (float): 감성 점수 (0.0 ~ 1.0)
        positive_threshold (float): 긍정 판단 임계값
        negative_threshold (float): 부정 판단 임계값

    Returns:
        str: "긍정", "부정", "중립"
    """
    if score >= positive_threshold:
        return "긍정"
    elif score <= negative_threshold:
        return "부정"
    else:
        return "중립"


# 예시 문장
EXAMPLE_SENTENCES = [
    # 긍정 리뷰
    "이 제품 정말 만족스럽습니다. 강력 추천합니다!",
    "가성비 최고예요. 품질도 너무 좋고 배송도 빨라요.",
    "피부가 촉촉해지고 발림성도 훌륭해요. 재구매 의사 100%",
    "향도 좋고 사용감이 부드러워서 매일 사용하고 있어요.",
    "민감한 피부인데 자극 없이 잘 맞아요. 대박",
    # 부정 리뷰
    "전혀 효과가 없네요. 돈 아깝습니다.",
    "피부에 트러블이 생겼어요. 실망스럽습니다.",
    "냄새가 너무 강하고 발림성도 별로예요.",
    "가격 대비 품질이 실망스럽습니다. 재구매 의사 없음",
    "배송도 늦고 제품도 별로입니다. 최악이에요.",
    # 중립 리뷰
    "그냥 평범한 제품이에요. 나쁘지도 좋지도 않아요.",
    "가격은 저렴한데 효과는 모르겠어요.",
    "특별한 느낌은 없지만 쓸만해요.",
]


def main():
    """메인 함수 - 테스트 및 데모용"""
    print("=" * 70)
    print("감성 분석 모델 유틸리티 - 예시 실행")
    print("=" * 70)

    # ========== Word2Vec 모델 테스트 ==========
    print("\n" + "=" * 70)
    print("Word2Vec 모델 테스트")
    print("=" * 70)

    print("\n모델 초기화 중...")
    initialize("word2vec")

    # 예시 문장들 테스트
    print("\n예시 문장 감성 분석 결과:\n")

    for i, text in enumerate(EXAMPLE_SENTENCES, 1):
        score = predict_sentiment(text, verbose=False, vectorizer_type="word2vec")
        label = get_sentiment_label(score)

        emoji = "😊" if label == "긍정" else "😞" if label == "부정" else "😐"

        print(f"{i:2d}. [{emoji} {label}] {score:.3f} | {text[:50]}")

    # ========== BERT 모델 테스트 ==========
    print("\n" + "=" * 70)
    print("BERT 모델 테스트")
    print("=" * 70)

    print("\n모델 초기화 중...")
    initialize("bert")

    # 예시 문장들 테스트
    print("\n예시 문장 감성 분석 결과:\n")

    for i, text in enumerate(EXAMPLE_SENTENCES, 1):
        score = predict_sentiment(text, verbose=False, vectorizer_type="bert")
        label = get_sentiment_label(score)

        emoji = "😊" if label == "긍정" else "😞" if label == "부정" else "😐"

        print(f"{i:2d}. [{emoji} {label}] {score:.3f} | {text[:50]}")

    # ========== 배치 예측 비교 테스트 ==========
    print("\n" + "=" * 70)
    print("배치 예측 비교 테스트 (Word2Vec vs BERT)")
    print("=" * 70)

    test_texts = [
        "정말 좋은 제품이에요!",
        "완전 실망했어요...",
        "그냥 보통이에요",
    ]

    scores_w2v = batch_predict(test_texts, vectorizer_type="word2vec")
    scores_bert = batch_predict(test_texts, vectorizer_type="bert")

    print(f"\n{'텍스트':<30} {'Word2Vec':<12} {'BERT':<12} {'차이':<8}")
    print("-" * 70)
    for text, score_w2v, score_bert in zip(test_texts, scores_w2v, scores_bert):
        label_w2v = get_sentiment_label(score_w2v)
        label_bert = get_sentiment_label(score_bert)
        diff = abs(score_w2v - score_bert)
        print(
            f"{text:<30} {score_w2v:.3f} ({label_w2v:<3}) {score_bert:.3f} ({label_bert:<3}) {diff:.3f}"
        )

    print("\n" + "=" * 70)
    print("사용 예시:")
    print("  from model_utils import predict_sentiment")
    print("  # Word2Vec 사용")
    print(
        '  score = predict_sentiment("이 제품 정말 좋아요!", vectorizer_type="word2vec")'
    )
    print("  # BERT 사용")
    print('  score = predict_sentiment("이 제품 정말 좋아요!", vectorizer_type="bert")')
    print("=" * 70)


if __name__ == "__main__":
    main()
    predict_sentiment(
        "나는 이것 저것 많은 제품을 썼는데 그 중에서 제일 좋다",
        verbose=True,
    )
    predict_sentiment(
        "나는 이것 저것 많은 제품을 써봤는데 대부분 별로 였어 그런데 이 제품은 매우 좋아",
        verbose=True,
    )
