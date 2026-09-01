"""
[전처리 후처리 단계]
전처리 완료된 Parquet 파일에 sentiment_score 추가

실행 순서:
  1. preprocessing/main.py (전처리) → sentiment_score=None
  2. model/train_sentiment_model.py (모델 학습)
  3. model/predict_sentiment_scores.py (이 파일) → sentiment_score 채우기

목적:
  - 전처리된 파티션 파일(partitioned_reviews_category_*.parquet)에 sentiment_score 컬럼 업데이트
  - 추천 시스템에서 미리 계산된 점수를 빠르게 조회 가능
"""

import os
import pandas as pd
import numpy as np
import joblib
import glob


def load_model(model_path):
    """
    학습된 모델 로드

    Args:
        model_path: 모델 파일 경로

    Returns:
        model: 로드된 모델
    """
    print(f"모델 로드 중: {model_path}")
    model = joblib.load(model_path)
    print("✓ 모델 로드 완료")
    return model


def predict_sentiment_scores(model, df, vector_type="roberta"):
    """
    DataFrame에 sentiment_score 예측 (벡터화 효율적 처리)

    Args:
        model: 학습된 모델
        df: 리뷰 DataFrame
        vector_type: 사용할 벡터 타입 ("word2vec", "bert", "roberta", "koelectra")

    Returns:
        int: 업데이트된 리뷰 개수
    """
    print(f"\nsentiment_score 예측 중 ({vector_type} 벡터 사용)...")

    # 지정된 벡터가 None이 아닌 행만 필터
    if vector_type not in df.columns:
        print(f"[경고] '{vector_type}' 컬럼이 존재하지 않습니다.")
        df["sentiment_score"] = None
        return 0

    valid_mask = df[vector_type].notna()
    valid_indices = df[valid_mask].index

    if len(valid_indices) == 0:
        print("✓ 벡터가 있는 리뷰: 0개")
        df["sentiment_score"] = None
        return 0

    # 벡터를 numpy array로 변환
    X = np.array(df.loc[valid_indices, vector_type].tolist())

    # 배치 예측 (긍정 확률 = 클래스 1)
    probas = model.predict_proba(X)[:, 1]

    # DataFrame에 업데이트
    df.loc[valid_indices, "sentiment_score"] = probas
    df.loc[~valid_mask, "sentiment_score"] = None

    updated_count = len(valid_indices)
    print(f"✓ {updated_count:,}개 리뷰에 sentiment_score 추가 완료")
    print(f"  (벡터 없음: {(~valid_mask).sum():,}개)")

    return updated_count


def process_category_files(
    model, partitioned_reviews_dir, vector_type="roberta_sentiment"
):
    """
    카테고리별 파티션 파일 처리 (Hive 파티셔닝 형식)

    Args:
        model: 학습된 모델
        partitioned_reviews_dir: partitioned_reviews 디렉토리 경로
        vector_type: 사용할 벡터 타입

    Returns:
        dict: 상품별 sentiment_score 평균값 {product_id: avg_score}
    """
    # Hive 파티셔닝: category=*/data.parquet 패턴
    parquet_files = glob.glob(
        os.path.join(partitioned_reviews_dir, "category=*", "data.parquet")
    )

    print(f"\n처리할 파일: {len(parquet_files)}개")

    total_updated = 0
    product_sentiment_scores = {}  # 상품별 평균 점수 저장

    for file_path in parquet_files:
        # 카테고리 추출: .../category=선스틱/data.parquet → 선스틱
        parent_dir = os.path.basename(os.path.dirname(file_path))
        category = parent_dir.replace("category=", "")
        print(f"\n[{category}] 처리 중...")

        # 파일 로드 (DataFrame으로 직접)
        df = pd.read_parquet(file_path)

        # sentiment_score 예측
        updated = predict_sentiment_scores(model, df, vector_type)
        total_updated += updated

        # 파일 덮어쓰기
        df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=False)

        file_size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"✓ 저장 완료: {file_path} ({file_size_mb:.2f} MB)")

        # 상품별 sentiment_score 평균 계산
        if "sentiment_score" in df.columns and "product_id" in df.columns:
            product_avg = df.groupby("product_id")["sentiment_score"].mean()
            product_sentiment_scores.update(product_avg.to_dict())

    return total_updated, product_sentiment_scores


def update_products_sentiment_scores(product_sentiment_scores, products_final_dir):
    """
    integrated_products_final 파일에 상품별 sentiment_score 업데이트

    Args:
        product_sentiment_scores: 상품별 평균 sentiment_score {product_id: avg_score}
        products_final_dir: integrated_products_final 디렉토리 경로
    """
    print("\n" + "=" * 70)
    print("상품별 sentiment_score 업데이트 중...")
    print("=" * 70)

    # Hive 파티셔닝: category=*/data.parquet 패턴
    parquet_files = glob.glob(
        os.path.join(products_final_dir, "category=*", "data.parquet")
    )

    total_products_updated = 0

    for file_path in parquet_files:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        category = parent_dir.replace("category=", "")
        print(f"\n[{category}] 상품 데이터 업데이트 중...")

        # 파일 로드
        df = pd.read_parquet(file_path)

        # sentiment_score 컬럼 추가 (없으면 생성)
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = None

        # product_id에 해당하는 평균 점수 매핑
        updated_count = 0
        for idx, row in df.iterrows():
            product_id = row["product_id"]
            if product_id in product_sentiment_scores:
                df.at[idx, "sentiment_score"] = product_sentiment_scores[product_id]
                updated_count += 1

        # 파일 덮어쓰기
        df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=False)

        total_products_updated += updated_count
        print(f"✓ {updated_count}개 상품에 sentiment_score 업데이트 완료")

    return total_products_updated


def main():
    """
    메인 함수
    """
    print("=" * 70)
    print("Sentiment Score 예측 및 업데이트 (RoBERTa 벡터 사용)")
    print("=" * 70)

    # 경로 설정
    MODEL_PATH = "./models/roberta_sentiment_XGBoost.joblib"  # roberta 모델 사용
    PROCESSED_DATA_DIR = "./data/new_processed_data"
    PARTITIONED_REVIEWS_DIR = os.path.join(PROCESSED_DATA_DIR, "partitioned_reviews")
    PRODUCTS_FINAL_DIR = os.path.join(PROCESSED_DATA_DIR, "integrated_products_final")
    VECTOR_TYPE = "roberta_sentiment"  # 사용할 벡터 타입

    # 1. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"\n[오류] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("먼저 train_sentiment_model.py를 실행하여 모델을 학습하세요.")
        print(f"또는 models/ 디렉토리에서 roberta_sentiment 모델을 확인하세요.")
        return

    model = load_model(MODEL_PATH)

    # 2. 리뷰별 sentiment_score 예측 및 저장 (partitioned_reviews 업데이트)
    print("\n" + "=" * 70)
    print("Step 1: 리뷰별 sentiment_score 예측")
    print("=" * 70)
    total_updated, product_sentiment_scores = process_category_files(
        model, PARTITIONED_REVIEWS_DIR, VECTOR_TYPE
    )

    # 3. 상품별 평균 sentiment_score 계산 및 저장 (integrated_products_final 업데이트)
    print("\n" + "=" * 70)
    print("Step 2: 상품별 평균 sentiment_score 계산 및 업데이트")
    print("=" * 70)
    total_products_updated = update_products_sentiment_scores(
        product_sentiment_scores, PRODUCTS_FINAL_DIR
    )

    # 최종 요약
    print("\n" + "=" * 70)
    print("업데이트 완료!")
    print("=" * 70)
    print(f"✓ 총 {total_updated:,}개 리뷰에 sentiment_score 추가됨")
    print(f"✓ 총 {total_products_updated:,}개 상품에 평균 sentiment_score 추가됨")
    print(f"✓ 사용된 벡터: {VECTOR_TYPE.upper()}")
    print("\n이제 추천 시스템에서 sentiment_score를 활용할 수 있습니다.")
    print("  - partitioned_reviews: 리뷰별 긍정 확률 (0.0 ~ 1.0)")
    print("  - integrated_products_final: 상품별 평균 긍정 확률")
    print("=" * 70)


if __name__ == "__main__":
    main()
