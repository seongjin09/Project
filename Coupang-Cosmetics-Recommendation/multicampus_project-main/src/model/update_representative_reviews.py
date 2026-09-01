"""
대표 리뷰 업데이트 (긍정/부정 Top 5)

기존 대표 리뷰 필드(representative_review_id, representative_similarity)를 삭제하고,
긍정/부정 각각 5개의 대표 리뷰를 선정하여 배열로 저장합니다.

선정 기준:
  - 긍정 대표 리뷰: 긍정 리뷰(label=1)들의 roberta_semantic 벡터 평균과의 유사도 * 0.7
                    + sentiment_score * 0.3
  - 부정 대표 리뷰: 부정 리뷰(label=0)들의 roberta_semantic 벡터 평균과의 유사도 * 0.7
                    + (1 - sentiment_score) * 0.3

실행 순서:
  preprocessing/main.py → sentiment_vectorize.py → semantic_vectorize.py
  → predict_sentiment_scores.py → update_representative_reviews.py (이 파일)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

# utils 모듈 import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.environment import get_execution_mode

# ========== 설정 ==========
TOP_N = 5  # 긍정/부정 각각 뽑을 대표 리뷰 개수
SIMILARITY_WEIGHT = 0.7  # 벡터 유사도 가중치
SENTIMENT_WEIGHT = 0.3  # 감성 점수 가중치
VECTOR_COL = "roberta_semantic"  # 사용할 벡터 컬럼

# ========== 환경별 경로 설정 ==========
exec_mode = get_execution_mode("auto")

if exec_mode == "colab":
    DATA_DIR = "/content/data/new_processed_data"
    print("[알림] Colab 환경: /content 로컬 스토리지 사용")
else:
    DATA_DIR = "./data/new_processed_data"


def cosine_similarity(vec1, vec2):
    """두 벡터 간 코사인 유사도 계산"""
    vec1 = np.array(vec1, dtype=np.float32)
    vec2 = np.array(vec2, dtype=np.float32)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def select_representative_reviews(reviews_df, label_value, top_n=TOP_N):
    """
    특정 라벨(긍정/부정)의 대표 리뷰 Top N 선정

    Args:
        reviews_df: 리뷰 DataFrame (roberta_semantic, sentiment_score, label 포함)
        label_value: 1(긍정) 또는 0(부정)
        top_n: 선정할 대표 리뷰 개수

    Returns:
        list[dict]: [{"id": review_id, "score": combined_score}, ...]
    """
    # 해당 라벨의 리뷰만 필터링
    label_reviews = reviews_df[reviews_df["label"] == label_value].copy()

    if label_reviews.empty:
        return []

    # roberta_semantic 벡터가 유효한 리뷰만 필터링
    valid_mask = label_reviews[VECTOR_COL].apply(
        lambda x: x is not None and isinstance(x, (list, np.ndarray)) and len(x) > 0
    )
    valid_reviews = label_reviews[valid_mask]

    if valid_reviews.empty:
        return []

    # 벡터 평균 계산 (해당 라벨 리뷰들의 centroid)
    vectors = np.array(valid_reviews[VECTOR_COL].tolist(), dtype=np.float32)
    centroid = np.mean(vectors, axis=0)

    # 각 리뷰의 유사도 계산
    similarities = []
    for idx, row in valid_reviews.iterrows():
        vec = row[VECTOR_COL]
        sim = cosine_similarity(centroid, vec)
        similarities.append(sim)

    valid_reviews = valid_reviews.copy()
    valid_reviews["_sim"] = similarities

    # 감성 점수 처리
    valid_reviews["_sentiment"] = valid_reviews["sentiment_score"].fillna(0.5)

    # 가중 점수 계산
    if label_value == 1:
        # 긍정: sentiment_score가 높을수록 좋음
        valid_reviews["_combined"] = (
            valid_reviews["_sim"] * SIMILARITY_WEIGHT
            + valid_reviews["_sentiment"] * SENTIMENT_WEIGHT
        )
    else:
        # 부정: sentiment_score가 0에 가까울수록 부정적 → (1 - sentiment_score)
        valid_reviews["_combined"] = (
            valid_reviews["_sim"] * SIMILARITY_WEIGHT
            + (1 - valid_reviews["_sentiment"]) * SENTIMENT_WEIGHT
        )

    # 점수 높은 순으로 정렬하여 Top N 선정
    top_reviews = valid_reviews.nlargest(top_n, "_combined")

    result = []
    for _, row in top_reviews.iterrows():
        result.append(
            {
                "id": row["id"],
                "score": round(float(row["_combined"]), 4),
            }
        )

    return result


def process_product_reviews(product_id, product_reviews_df):
    """
    단일 상품의 긍정/부정 대표 리뷰 선정

    Args:
        product_id: 상품 ID
        product_reviews_df: 해당 상품의 리뷰 DataFrame

    Returns:
        dict: {
            "positive_representative_ids": [...],
            "positive_representative_scores": [...],
            "negative_representative_ids": [...],
            "negative_representative_scores": [...],
        }
    """
    # 긍정 대표 리뷰 Top 5
    pos_reps = select_representative_reviews(product_reviews_df, label_value=1)
    # 부정 대표 리뷰 Top 5
    neg_reps = select_representative_reviews(product_reviews_df, label_value=0)

    return {
        "positive_representative_ids": [r["id"] for r in pos_reps],
        "positive_representative_scores": [r["score"] for r in pos_reps],
        "negative_representative_ids": [r["id"] for r in neg_reps],
        "negative_representative_scores": [r["score"] for r in neg_reps],
    }


def main():
    print("=" * 70)
    print("대표 리뷰 업데이트 (긍정/부정 Top 5)")
    print("=" * 70)
    print(f"\n실행 환경: {exec_mode.upper()}")
    print(f"데이터 경로: {DATA_DIR}")
    print(f"벡터 컬럼: {VECTOR_COL}")
    print(f"가중치: 유사도 {SIMILARITY_WEIGHT} + 감성점수 {SENTIMENT_WEIGHT}")
    print(f"선정 개수: 긍정 {TOP_N}개, 부정 {TOP_N}개\n")

    PARTITIONED_REVIEWS_DIR = os.path.join(DATA_DIR, "partitioned_reviews")
    PRODUCTS_FINAL_DIR = os.path.join(DATA_DIR, "integrated_products_final")

    # ========== Step 1: partitioned_reviews에서 대표 리뷰 선정 ==========
    print("=" * 70)
    print("Step 1: 카테고리별 대표 리뷰 선정")
    print("=" * 70)

    review_files = glob.glob(
        os.path.join(PARTITIONED_REVIEWS_DIR, "category=*", "data.parquet")
    )

    if not review_files:
        print(f"\n[오류] {PARTITIONED_REVIEWS_DIR}에 파일이 없습니다.")
        return

    print(f"처리할 파일: {len(review_files)}개\n")

    # 상품별 대표 리뷰 결과 저장
    product_representatives = {}
    total_products = 0

    for file_path in review_files:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        category = parent_dir.replace("category=", "")
        print(f"[{category}] 처리 중...")

        df = pd.read_parquet(file_path)

        # 필수 컬럼 확인
        required_cols = [VECTOR_COL, "sentiment_score", "label", "product_id", "id"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  [경고] 필요한 컬럼 없음: {missing}. 건너뜀.")
            continue

        # 상품별 처리
        product_ids = df["product_id"].unique()
        category_count = 0

        for product_id in product_ids:
            product_df = df[df["product_id"] == product_id]
            result = process_product_reviews(product_id, product_df)
            product_representatives[product_id] = result
            category_count += 1

        total_products += category_count
        pos_with_reps = sum(
            1
            for pid in product_ids
            if len(product_representatives[pid]["positive_representative_ids"]) > 0
        )
        neg_with_reps = sum(
            1
            for pid in product_ids
            if len(product_representatives[pid]["negative_representative_ids"]) > 0
        )
        print(
            f"  ✓ {category_count}개 상품 처리 완료 "
            f"(긍정대표 있음: {pos_with_reps}, 부정대표 있음: {neg_with_reps})"
        )

    print(f"\n✓ 총 {total_products:,}개 상품의 대표 리뷰 선정 완료\n")

    # ========== Step 2: integrated_products_final 업데이트 ==========
    print("=" * 70)
    print("Step 2: integrated_products_final 업데이트")
    print("=" * 70)

    product_files = glob.glob(
        os.path.join(PRODUCTS_FINAL_DIR, "category=*", "data.parquet")
    )

    if not product_files:
        print(f"\n[오류] {PRODUCTS_FINAL_DIR}에 파일이 없습니다.")
        return

    print(f"처리할 파일: {len(product_files)}개\n")

    total_updated = 0

    for file_path in product_files:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        category = parent_dir.replace("category=", "")
        print(f"[{category}] 업데이트 중...")

        df = pd.read_parquet(file_path)

        # 기존 대표 리뷰 필드 삭제 (모든 모델의 representative 필드)
        cols_to_drop = [
            col
            for col in df.columns
            if col.startswith("representative_review_id_")
            or col.startswith("representative_similarity_")
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"  - 기존 필드 삭제: {cols_to_drop}")

        # 새 필드 추가
        df["positive_representative_ids"] = df["product_id"].apply(
            lambda pid: product_representatives.get(pid, {}).get(
                "positive_representative_ids", []
            )
        )
        df["positive_representative_scores"] = df["product_id"].apply(
            lambda pid: product_representatives.get(pid, {}).get(
                "positive_representative_scores", []
            )
        )
        df["negative_representative_ids"] = df["product_id"].apply(
            lambda pid: product_representatives.get(pid, {}).get(
                "negative_representative_ids", []
            )
        )
        df["negative_representative_scores"] = df["product_id"].apply(
            lambda pid: product_representatives.get(pid, {}).get(
                "negative_representative_scores", []
            )
        )

        # 업데이트된 행 수 카운트
        updated = df["positive_representative_ids"].apply(len).gt(0).sum()
        total_updated += updated

        # 파일 저장
        df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=False)

        file_size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"  ✓ {len(df):,}개 상품 저장 완료 ({file_size_mb:.2f} MB)")

    # ========== 최종 요약 ==========
    print("\n" + "=" * 70)
    print("대표 리뷰 업데이트 완료!")
    print("=" * 70)
    print(f"✓ 총 {total_products:,}개 상품 처리")
    print(f"✓ 총 {total_updated:,}개 상품에 대표 리뷰 저장됨")
    print(f"✓ 벡터: {VECTOR_COL}")
    print(f"✓ 가중치: 유사도 {SIMILARITY_WEIGHT} + 감성점수 {SENTIMENT_WEIGHT}")
    print(f"\n추가된 필드:")
    print(f"  - positive_representative_ids: 긍정 대표 리뷰 ID 배열 (최대 {TOP_N}개)")
    print(
        f"  - positive_representative_scores: 긍정 대표 리뷰 점수 배열 (최대 {TOP_N}개)"
    )
    print(f"  - negative_representative_ids: 부정 대표 리뷰 ID 배열 (최대 {TOP_N}개)")
    print(
        f"  - negative_representative_scores: 부정 대표 리뷰 점수 배열 (최대 {TOP_N}개)"
    )
    print(f"\n삭제된 필드:")
    print(f"  - representative_review_id_*")
    print(f"  - representative_similarity_*")
    print("=" * 70)


if __name__ == "__main__":
    main()
