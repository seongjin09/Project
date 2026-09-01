"""
전처리 파이프라인 Phase 1, 2, 3 함수들
"""

import json
import os
import glob
import pickle
import warnings
import sys
import unicodedata
from contextlib import contextmanager
from collections import Counter
import numpy as np
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from preprocess_format import preprocess_format
from brand_standardizer import brand_standardizer
from drop_missing_val_splitter import drop_missing_val_splitter
from skintype import classify_product
from sentiment_analysis import (
    analyze_skin_type_frequency,
    analyze_category_sentiment,
    analyze_product_sentiment,
)
from preprocessing_utils import (
    load_stopwords,
    get_tokens,
    cosine_similarity,
    TokenIterator,
)

# gensim 내부 경고 억제
warnings.filterwarnings("ignore", category=RuntimeWarning, module="gensim")


@contextmanager
def suppress_stderr():
    """stderr 출력을 임시로 억제"""
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr


MAX_WORKERS = max(1, cpu_count() - 1)


def preprocess_and_tokenize_file(args):
    """
    Phase 1: 파일 전처리 + 토큰화 (병렬 실행)
    - 포맷 전처리, 브랜드 표준화, 결측치 제거, 토큰화를 한 번에 수행
    - 토큰 결과를 임시 파일로 저장
    - 리뷰 개수가 최소 개수 미만인 상품 제외
    """
    input_path, pre_data_dir, processed_data_dir, temp_tokens_dir, min_reviews = args

    file_name = os.path.basename(input_path)
    stopwords = load_stopwords()

    try:
        # 상대 경로 계산
        rel_path = os.path.relpath(input_path, pre_data_dir)
        rel_dir = os.path.dirname(rel_path)
        output_dir = os.path.join(processed_data_dir, rel_dir)

        # 출력 파일명 계산
        base_name = os.path.splitext(file_name)[0]
        if base_name.startswith("result_"):
            base_name = base_name[7:]

        output_with_text = os.path.join(
            output_dir, f"processed_{base_name}_with_text.json"
        )
        output_without_text = os.path.join(
            output_dir, f"processed_{base_name}_without_text.json"
        )

        # 이미 처리된 파일이면 스킵
        if os.path.exists(output_with_text) and os.path.exists(output_without_text):
            return {"status": "skipped", "file": file_name}

        # 1. JSON 파일 로드
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2. 포맷 전처리
        temp_file = f"temp_{os.getpid()}_{base_name}.json"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        data = preprocess_format(temp_file)

        if os.path.exists(temp_file):
            os.remove(temp_file)

        # 3. 브랜드 표준화
        data = brand_standardizer(data)

        # 4. 결측치 제거 및 분할
        with_text, without_text = drop_missing_val_splitter(data)

        # 4-1. 리뷰 개수 필터링: 최소 개수 미만인 상품 제외
        filtered_with_text = []
        for product in with_text.get("data", []):
            review_count = len(product.get("reviews", {}).get("data", []))
            if review_count >= min_reviews:
                filtered_with_text.append(product)

        filtered_without_text = []
        for product in without_text.get("data", []):
            review_count = product.get("product_info", {}).get("total_reviews", 0)
            if review_count >= min_reviews:
                filtered_without_text.append(product)

        # 필터링된 결과로 교체
        with_text["data"] = filtered_with_text
        without_text["data"] = filtered_without_text

        # 필터링 후 데이터가 없으면 스킵
        if not with_text.get("data") and not without_text.get("data"):
            return {
                "status": "skipped",
                "file": file_name,
                "reason": f"모든 상품의 리뷰가 {min_reviews}개 미만",
            }

        # 4-2. without_text의 product_id 수정 (카테고리별 순차 번호)
        # with_text에서 사용한 마지막 번호 다음부터 시작
        next_product_num = len(filtered_with_text) + 1

        for product_idx, product in enumerate(without_text.get("data", [])):
            p_info = product.get("product_info", {})
            original_id = p_info.get("product_id", p_info.get("id", ""))
            category = unicodedata.normalize("NFC", str(base_name))
            unique_product_id = unicodedata.normalize(
                "NFC", f"{category}_{next_product_num + product_idx}"
            )
            p_info["product_id"] = unique_product_id
            p_info["original_product_id"] = original_id
            p_info["category"] = category

        # 5. 토큰화 (한 번만 수행하고 저장)
        all_tokens = []  # Word2Vec 학습용
        tokenized_data = []  # 나중에 벡터화에 사용할 토큰 저장

        for product_idx, product in enumerate(with_text.get("data", [])):
            p_info = product.get("product_info", {})

            # product_id를 카테고리별 순차 번호로 고유하게 생성 (카테고리_1, 카테고리_2, ...)
            original_id = p_info.get("product_id", p_info.get("id", ""))
            category = unicodedata.normalize("NFC", str(base_name))  # NFC 정규화
            unique_product_id = unicodedata.normalize(
                "NFC", f"{category}_{product_idx + 1}"
            )  # 순차 번호 (1부터 시작)

            # product_info에 고유 ID 업데이트
            p_info["product_id"] = unique_product_id
            p_info["original_product_id"] = original_id  # 원본 ID 보존
            p_info["category"] = category

            # skin_type 추가
            skin_result = classify_product(product)
            p_info["skin_type"] = skin_result.get("skin_type", "미분류")

            product_tokens = {
                "product_id": unique_product_id,
                "reviews": [],
            }

            for review in product.get("reviews", {}).get("data", []):
                full_text = review.get("full_text", "")
                tokens = get_tokens(full_text, stopwords)

                # label 생성 (score 기반: 4-5점=긍정(1), 1-2점=부정(0), 3점=중립(제외))
                score = review.get("score", 3)
                if score >= 4:
                    label = 1  # 긍정
                elif score <= 2:
                    label = 0  # 부정
                else:
                    label = None  # 중립 (분석 제외)

                # 원본 리뷰 객체에 tokens와 label 주입 (감성 분석을 위해)
                review["tokens"] = tokens
                review["label"] = label

                # 토큰 저장 (글자 수, 토큰 수 포함)
                product_tokens["reviews"].append(
                    {
                        "review_id": review.get("id"),
                        "tokens": tokens,
                        "score": score,
                        "label": label,
                        "char_length": len(full_text),
                        "token_count": len(tokens),
                    }
                )

                if tokens:
                    all_tokens.append(tokens)

            tokenized_data.append(product_tokens)

        # 6. 토큰을 임시 파일로 저장 (Word2Vec 학습용)
        os.makedirs(temp_tokens_dir, exist_ok=True)
        token_file = os.path.join(temp_tokens_dir, f"{base_name}_tokens.pkl")
        with open(token_file, "wb") as f:
            pickle.dump(all_tokens, f)

        # 7. 토큰화된 데이터 저장 (벡터화에 재사용)
        tokenized_file = os.path.join(temp_tokens_dir, f"{base_name}_tokenized.pkl")
        with open(tokenized_file, "wb") as f:
            pickle.dump(
                {
                    "with_text": with_text,
                    "without_text": without_text,
                    "tokenized_data": tokenized_data,
                },
                f,
            )

        return {
            "status": "success",
            "file": file_name,
            "token_count": len(all_tokens),
            "output_dir": output_dir,
            "base_name": base_name,
        }

    except Exception as e:
        return {"status": "error", "file": file_name, "error": str(e)}


def train_global_word2vec(temp_tokens_dir):
    """
    Phase 2: Iterator 방식으로 Word2Vec 모델 학습 (메모리 효율적)
    """
    print("\n" + "=" * 60)
    print("전역 Word2Vec 모델 학습 시작 (Iterator 방식)")
    print("=" * 60)

    # TokenIterator를 사용하여 메모리에 모든 토큰을 올리지 않음
    token_iterator = TokenIterator(temp_tokens_dir)

    # 토큰 파일 개수 확인
    token_files = glob.glob(os.path.join(temp_tokens_dir, "*_tokens.pkl"))
    print(f"토큰 파일 수: {len(token_files)}개")

    if not token_files:
        print("[경고] 토큰 파일이 없습니다. Word2Vec 학습을 건너뜁니다.")
        return None

    # Word2Vec 모델 학습 (Skip-gram, Iterator 방식) - stderr 억제
    with suppress_stderr():
        model = Word2Vec(
            sentences=token_iterator,
            vector_size=100,
            window=5,
            min_count=3,
            workers=MAX_WORKERS,
            sg=1,  # Skip-gram
        )

    print(f"Word2Vec 모델 학습 완료 (어휘 크기: {len(model.wv):,})")
    return model


def vectorize_file(args):
    """
    Phase 3: 저장된 토큰을 재사용하여 벡터화 + 대표 리뷰 선정 (병렬 실행)
    - JSON: 상품 요약 정보만 저장 (대표 벡터 포함)
    - 리뷰 상세 정보는 반환하여 Parquet로 통합 저장
    - vectorizer_type 리스트에 따라 여러 모델의 벡터 생성
    """
    (
        base_name,
        temp_tokens_dir,
        output_dir,
        w2v_model,
        vectorizers,  # dict: {model_name: vectorizer}
        vectorizer_type,  # list: ["word2vec", "bert", "roberta", ...]
        batch_size,  # int or None: 배치 사이즈 (테스트용)
    ) = args

    try:
        import time

        # 저장된 토큰화 데이터 로드
        tokenized_file = os.path.join(temp_tokens_dir, f"{base_name}_tokenized.pkl")
        with open(tokenized_file, "rb") as f:
            saved_data = pickle.load(f)

        with_text = saved_data["with_text"]
        without_text = saved_data["without_text"]
        tokenized_data = saved_data["tokenized_data"]

        # 상품 요약 정보 & 리뷰 상세 정보 수집
        product_summaries = []
        review_details = []

        # 모델별 처리 시간 측정
        model_times = {model_name: 0.0 for model_name in vectorizer_type}

        for product_idx, product in enumerate(with_text.get("data", [])):
            # 모델별 벡터 리스트 저장
            review_vectors_by_model = {model_name: [] for model_name in vectorizer_type}

            product_tokens = tokenized_data[product_idx]
            product_info = product.get("product_info", {})
            reviews_data = product.get("reviews", {}).get("data", [])

            # 리뷰 정보 미리 수집
            review_infos = []
            full_texts = []
            for review_idx, review in enumerate(reviews_data):
                saved_review = product_tokens["reviews"][review_idx]
                tokens = saved_review["tokens"]
                score = saved_review["score"]
                full_text = review.get("full_text", "")

                # 감성 라벨링
                if score >= 4:
                    label = 1  # 긍정
                elif score <= 2:
                    label = 0  # 부정
                else:
                    label = None  # 중립

                review_detail = {
                    "product_id": product_info.get("product_id"),
                    "id": review.get("id"),
                    "full_text": full_text,
                    "title": review.get("title", ""),
                    "content": review.get("content", ""),
                    "score": score,
                    "label": label,
                    "tokens": tokens,
                    "char_length": saved_review["char_length"],
                    "token_count": saved_review["token_count"],
                    "date": review.get("date"),
                    "collected_at": review.get("collected_at"),
                    "nickname": review.get("nickname"),
                    "has_image": review.get("has_image"),
                    "helpful_count": review.get("helpful_count"),
                }

                review_infos.append(
                    {
                        "review_detail": review_detail,
                        "tokens": tokens,
                        "review_id": review.get("id"),
                        "review_idx": review_idx,
                    }
                )
                full_texts.append(full_text)

            # Word2Vec 벡터 생성 (개별 처리)
            if "word2vec" in vectorizer_type and w2v_model:
                model_start = time.time()
                for info in review_infos:
                    tokens = info["tokens"]
                    word_vectors = [
                        w2v_model.wv[w] for w in tokens if w in w2v_model.wv
                    ]
                    if word_vectors:
                        vec = np.mean(word_vectors, axis=0)
                    else:
                        vec = np.zeros(100)

                    info["review_detail"]["word2vec"] = vec.tolist()
                    review_vectors_by_model["word2vec"].append(
                        {
                            "vector": vec,
                            "review_id": info["review_id"],
                            "review_idx": info["review_idx"],
                        }
                    )
                model_times["word2vec"] += time.time() - model_start

            # Transformer 모델 벡터 생성 (배치 처리)
            transformer_models = [m for m in vectorizer_type if m in vectorizers]

            for model_name in transformer_models:
                model_start = time.time()

                # 배치로 한 번에 벡터화 (batch_size 파라미터 사용)
                vectors = vectorizers[model_name].encode_batch(
                    full_texts, batch_size=batch_size
                )

                for idx, (info, vec) in enumerate(zip(review_infos, vectors)):
                    info["review_detail"][model_name] = vec.tolist()
                    review_vectors_by_model[model_name].append(
                        {
                            "vector": vec,
                            "review_id": info["review_id"],
                            "review_idx": info["review_idx"],
                        }
                    )

                model_times[model_name] += time.time() - model_start

            # 사용하지 않는 모델은 None 설정
            for model_name in vectorizer_type:
                if model_name != "word2vec" and model_name not in vectorizers:
                    for info in review_infos:
                        info["review_detail"][model_name] = None

            # review_details에 추가
            for info in review_infos:
                review_details.append(info["review_detail"])

            # 각 모델별 상품 대표 벡터 생성
            for model_name in vectorizer_type:
                review_vectors = review_vectors_by_model[model_name]

                if review_vectors:
                    # 상품 벡터 = 리뷰 벡터들의 평균
                    product_vec = np.mean(
                        [rv["vector"] for rv in review_vectors], axis=0
                    )
                    product_info[f"product_vector_{model_name}"] = product_vec.tolist()

                    # 대표 리뷰 선정 (상품 벡터와 가장 유사한 리뷰)
                    max_sim = -1
                    rep_id = None
                    for rv in review_vectors:
                        sim = cosine_similarity(product_vec, rv["vector"])
                        if sim > max_sim:
                            max_sim = sim
                            rep_id = rv["review_id"]

                    product_info[f"representative_review_id_{model_name}"] = rep_id
                    product_info[f"representative_similarity_{model_name}"] = float(
                        max_sim
                    )
                else:
                    product_info[f"product_vector_{model_name}"] = []
                    product_info[f"representative_review_id_{model_name}"] = None
                    product_info[f"representative_similarity_{model_name}"] = 0.0

            # 상품별 감성 키워드 분석
            sentiment_result = analyze_product_sentiment(
                product, top_n=30, min_doc_freq=5
            )
            product_info["sentiment_analysis"] = sentiment_result

            product_summaries.append(product_info)

        # without_text의 리뷰도 review_details에 추가
        for product in without_text.get("data", []):
            product_info_without = product.get("product_info", {})
            product_id = product_info_without.get("product_id")

            for review in product.get("reviews", {}).get("data", []):
                review_detail = {
                    "product_id": product_id,
                    "id": review.get("id"),
                    "full_text": "",  # 텍스트 없음
                    "title": review.get("title", ""),
                    "content": review.get("content", ""),
                    "score": review.get("score"),
                    "label": None,  # 텍스트 없으므로 라벨 없음
                    "tokens": [],
                    "char_length": 0,
                    "token_count": 0,
                    "date": review.get("date"),
                    "collected_at": review.get("collected_at"),
                    "nickname": review.get("nickname"),
                    "has_image": review.get("has_image"),
                    "helpful_count": review.get("helpful_count"),
                }

                # 모든 모델에 대해 None 설정
                for model_name in vectorizer_type:
                    review_detail[model_name] = None

                review_details.append(review_detail)

        # 카테고리별 감성 키워드 분석
        category_sentiment = analyze_category_sentiment(
            with_text.get("data", []), top_n=30, min_doc_freq=20
        )

        return {
            "status": "success",
            "file": base_name,
            "product_summaries": product_summaries,
            "review_details": review_details,
            "category_sentiment": category_sentiment,
            "model_times": model_times,  # 모델별 처리 시간 추가
        }

    except Exception as e:
        return {"status": "error", "file": base_name, "error": str(e)}
