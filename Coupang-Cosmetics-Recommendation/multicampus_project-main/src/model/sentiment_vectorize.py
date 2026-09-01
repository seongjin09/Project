"""
미세조정된 커스텀 모델로 재벡터화
기존 pickle 파일을 읽어서 커스텀 모델로 벡터화하여 새로운 폴더에 저장
"""

import os
import sys
import glob
import time
import pickle
from tqdm import tqdm

try:
    # 1. 로컬(Local) 및 일반 스크립트 실행 환경
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_path = os.path.abspath(
        os.path.join(current_dir, "..", "preprocessing")
    )

    if preprocessing_path not in sys.path:
        sys.path.insert(0, preprocessing_path)

except NameError:
    # 2. 코랩(Colab) 및 주피터 노트북 환경
    current_path = os.getcwd()
    preprocessing_path = os.path.abspath(
        os.path.join(current_path, "src", "preprocessing")
    )

    if preprocessing_path not in sys.path:
        sys.path.append(preprocessing_path)

# preprocessing 모듈 import
from bert_vectorizer import get_bert_vectorizer
from preprocessing_phases import vectorize_file

# utils 모듈 import
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
from utils.environment import get_execution_mode

# ========== 환경별 경로 설정 ==========
exec_mode = get_execution_mode("auto")

if exec_mode == "colab":
    TEMP_TOKENS_DIR = "/content/data/temp_tokens"
    INPUT_DATA_DIR = "/content/data/processed_data"
    OUTPUT_DATA_DIR = "/content/data/new_processed_data"
    MODELS_BASE_DIR = "/content/models/fine_tuned"
    print("[알림] Colab 환경: /content 로컬 스토리지 사용 (빠른 I/O)")
else:
    TEMP_TOKENS_DIR = "./data/temp_tokens"
    INPUT_DATA_DIR = "./data/processed_data"
    OUTPUT_DATA_DIR = "./data/new_processed_data"
    MODELS_BASE_DIR = "./models/fine_tuned"

# ========== 사용할 모델 선택 ==========
# 원하는 모델만 리스트에 포함시키세요
# 예: ["word2vec", "bert_sentiment", "roberta_sentiment", "koelectra_sentiment"]
MODELS_TO_USE = ["roberta_sentiment"]

# 모델 경로 설정
MODEL_PATHS = {
    "word2vec": os.path.join(MODELS_BASE_DIR, "word2vec_model.model"),
    "bert_sentiment": os.path.join(MODELS_BASE_DIR, "bert_sentiment_final"),
    "roberta_sentiment": os.path.join(MODELS_BASE_DIR, "roberta_sentiment_final"),
    "koelectra_sentiment": os.path.join(MODELS_BASE_DIR, "koelectra_sentiment_final"),
}


def main():
    print("\n" + "=" * 80)
    print(f"{'커스텀 모델 재벡터화 시작 (새 폴더 저장 모드)':^80}")
    print("=" * 80 + "\n")

    print(f"실행 환경: {exec_mode.upper()}")
    print(f"데이터 경로: {INPUT_DATA_DIR}")
    print(f"출력 경로: {OUTPUT_DATA_DIR}")
    print(f"모델 경로: {MODELS_BASE_DIR}\n")

    # ========== 배치 사이즈 설정 (GPU 감지) ==========
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "A100" in gpu_name:
            batch_size = 64
        elif "T4" in gpu_name:
            batch_size = 32
        else:
            batch_size = 16
        print(f"GPU 감지: {gpu_name}")
        print(f"배치 사이즈: {batch_size} (GPU 최적화)\n")
    else:
        batch_size = 32  # CPU 기본값
        print(f"GPU 미감지: CPU 모드")
        print(f"배치 사이즈: {batch_size}\n")

    # 토큰화된 파일 찾기 (기존 경로에서 읽음)
    tokenized_files = glob.glob(os.path.join(TEMP_TOKENS_DIR, "*_tokenized.pkl"))

    if not tokenized_files:
        print(f"[오류] {TEMP_TOKENS_DIR}에 토큰 파일이 없습니다.")
        return

    print(f"발견된 토큰 파일: {len(tokenized_files)}개\n")

    # 모델 로딩
    print("=" * 60)
    print("모델 로딩 중...")
    print("=" * 60 + "\n")

    w2v_model = None
    vectorizers = {}
    models_to_use = []

    for model_name in MODELS_TO_USE:
        if model_name not in MODEL_PATHS:
            print(f"[경고] {model_name}은(는) 지원되지 않는 모델입니다. 건너뜀.")
            continue

        model_path = MODEL_PATHS[model_name]

        if model_name == "word2vec":
            # Word2Vec 모델 로드
            if os.path.exists(model_path):
                print(f"WORD2VEC 모델 로딩... ({model_path})")
                from gensim.models import Word2Vec

                w2v_model = Word2Vec.load(model_path)
                models_to_use.append(model_name)
                print(f"✓ WORD2VEC 로드 완료 (어휘: {len(w2v_model.wv):,}개)")
            else:
                print(f"[경고] Word2Vec 모델 없음. 건너뜀.")
                print(f"      경로: {model_path}")
        else:
            # Transformer 모델 로드
            if os.path.exists(model_path):
                print(f"{model_name.upper()} 모델 로딩... ({model_path})")
                vectorizers[model_name] = get_bert_vectorizer(model_path)
                models_to_use.append(model_name)
                print(f"✓ {model_name.upper()} 로드 완료")
            else:
                print(f"[경고] {model_name} 모델 없음. 건너뜀.")
                print(f"      경로: {model_path}")

    if not models_to_use:
        print("\n[오류] 사용 가능한 모델이 없습니다.")
        print("모델을 먼저 학습하거나 MODELS_TO_USE 설정을 확인하세요.")
        return

    print(f"\n사용할 모델: {', '.join(models_to_use)}")
    print("배치 사이즈: GPU에 따라 자동 설정\n")

    # 재벡터화 실행
    print("=" * 60)
    print("재벡터화 시작")
    print("=" * 60 + "\n")

    start_time = time.time()
    all_products = []
    all_reviews = []

    for tokenized_file in tqdm(tokenized_files, desc="재벡터화", unit="파일"):
        base_name = os.path.basename(tokenized_file).replace("_tokenized.pkl", "")

        # 출력 디렉토리는 새 폴더 구조를 따름
        # 기존 카테고리 구조를 파악하기 위해 INPUT_DATA_DIR 검색
        existing_pickles = glob.glob(
            os.path.join(INPUT_DATA_DIR, "**", f"{base_name}.pkl"), recursive=True
        )

        # 결과물은 무조건 OUTPUT_DATA_DIR 하위로 경로 재설정
        if existing_pickles:
            rel_dir = os.path.relpath(
                os.path.dirname(existing_pickles[0]), INPUT_DATA_DIR
            )
            output_dir = os.path.join(OUTPUT_DATA_DIR, rel_dir)
        else:
            output_dir = OUTPUT_DATA_DIR

        args = (
            base_name,
            TEMP_TOKENS_DIR,
            output_dir,
            w2v_model,  # word2vec 모델 전달 (None일 수도 있음)
            vectorizers,  # Transformer 모델들
            models_to_use,  # 사용할 모델 목록
            batch_size,  # Colab이면 64, 로컬이면 None (자동)
        )

        result = vectorize_file(args)

        if result["status"] == "success":
            all_products.extend(result["product_summaries"])
            all_reviews.extend(result["review_details"])
            tqdm.write(f"  [완료] {base_name}")
        else:
            tqdm.write(f"  [에러] {base_name} - {result.get('error', 'Unknown')}")

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"재벡터화 완료 - 소요 시간: {elapsed_time:.2f}초")
    print(f"총 상품: {len(all_products):,}개, 총 리뷰: {len(all_reviews):,}개")
    print("=" * 80 + "\n")

    # ========== Parquet 파일 저장 (새로운 폴더에 생성) ==========
    print("=" * 80)
    print(f"Parquet 파일 생성 중: {OUTPUT_DATA_DIR}")
    print("=" * 80 + "\n")

    import pandas as pd

    # 새 폴더 내 세부 디렉토리 설정
    PRODUCTS_FINAL_DIR = os.path.join(OUTPUT_DATA_DIR, "integrated_products_final")
    DETAILED_STATS_DIR = os.path.join(OUTPUT_DATA_DIR, "detailed_stats")
    PARTITIONED_REVIEWS_DIR = os.path.join(OUTPUT_DATA_DIR, "partitioned_reviews")
    CATEGORY_SUMMARY_DIR = os.path.join(OUTPUT_DATA_DIR, "category_summary")

    # 모든 출력 디렉토리 생성
    for d in [
        PRODUCTS_FINAL_DIR,
        DETAILED_STATS_DIR,
        PARTITIONED_REVIEWS_DIR,
        CATEGORY_SUMMARY_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # 1. integrated_products_final 저장 (통계 재계산 + 카테고리별 파티셔닝)
    print("[1/4] integrated_products_final 생성 중...")

    products_final = []
    for product in all_products:
        product_id = product.get("product_id")
        product_reviews = [r for r in all_reviews if r.get("product_id") == product_id]
        text_reviews = [r for r in product_reviews if r.get("full_text")]
        no_text_reviews = [r for r in product_reviews if not r.get("full_text")]

        rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in product_reviews:
            score = r.get("score")
            if score in rating_dist:
                rating_dist[score] += 1

        category = "_".join(product_id.split("_")[:-1]) if "_" in product_id else "기타"
        sentiment = product.get("sentiment_analysis", {})
        category_path_str = product.get("category_path", "")

        products_final_dict = {
            "product_id": product_id,
            "product_name": product.get("product_name"),
            "brand": product.get("brand"),
            "category": category,
            "category_path": category_path_str,
            "path": category_path_str.split(" > ")[-1] if category_path_str else "",
            "price": product.get("price"),
            "delivery_type": product.get("delivery_type"),
            "product_url": product.get("product_url"),
            "img_url": product.get("img_url"),
            "skin_type": product.get("skin_type", "미분류"),
            "top_keywords": [
                kw.get("word", "") for kw in sentiment.get("positive_special", [])[:5]
            ],
            "sentiment_analysis": sentiment,
            "avg_rating_with_text": (
                sum(r.get("score", 0) for r in text_reviews) / len(text_reviews)
                if text_reviews
                else 0.0
            ),
            "avg_rating_without_text": (
                sum(r.get("score", 0) for r in no_text_reviews) / len(no_text_reviews)
                if no_text_reviews
                else 0.0
            ),
            "text_review_ratio": (
                len(text_reviews) / len(product_reviews) if product_reviews else 0.0
            ),
            "total_reviews": len(product_reviews),
            "rating_1": rating_dist[1],
            "rating_2": rating_dist[2],
            "rating_3": rating_dist[3],
            "rating_4": rating_dist[4],
            "rating_5": rating_dist[5],
        }

        # 벡터 필드 동적 추가 (선택된 모든 모델)
        for model_name in models_to_use:
            products_final_dict.update(
                {
                    f"product_vector_{model_name}": product.get(
                        f"product_vector_{model_name}"
                    ),
                    f"representative_review_id_{model_name}": product.get(
                        f"representative_review_id_{model_name}"
                    ),
                    f"representative_similarity_{model_name}": product.get(
                        f"representative_similarity_{model_name}"
                    ),
                }
            )

        products_final.append(products_final_dict)

    df_products_final = pd.DataFrame(products_final)

    # 카테고리별 Hive 파티셔닝 저장
    products_count = 0
    for category in df_products_final["category"].unique():
        df_category = df_products_final[df_products_final["category"] == category]
        partition_dir = os.path.join(PRODUCTS_FINAL_DIR, f"category={category}")
        os.makedirs(partition_dir, exist_ok=True)

        category_path = os.path.join(partition_dir, "data.parquet")
        df_category.to_parquet(
            category_path, engine="pyarrow", compression="snappy", index=False
        )

        file_size_mb = os.path.getsize(category_path) / 1024 / 1024
        products_count += 1
        print(f"  ✓ {category}: {len(df_category):,}개 상품 ({file_size_mb:.2f} MB)")

    print(f"✓ 총 {products_count}개 카테고리 파티션 생성 완료")

    # 2. category_summary 저장
    print("\n[2/4] category_summary 생성 중...")

    category_stats = {}
    for review in all_reviews:
        product_id = review.get("product_id")
        category = "_".join(product_id.split("_")[:-1]) if "_" in product_id else "기타"

        if category not in category_stats:
            category_stats[category] = {
                "category": category,
                "total_reviews": 0,
                "text_reviews": 0,
                "no_text_reviews": 0,
                "rating_1": 0,
                "rating_2": 0,
                "rating_3": 0,
                "rating_4": 0,
                "rating_5": 0,
            }

        stats = category_stats[category]
        stats["total_reviews"] += 1

        if review.get("full_text", "").strip():
            stats["text_reviews"] += 1
        else:
            stats["no_text_reviews"] += 1

        score = review.get("score")
        if score in [1, 2, 3, 4, 5]:
            stats[f"rating_{score}"] += 1

    df_category_summary = pd.DataFrame(list(category_stats.values()))
    category_summary_path = os.path.join(CATEGORY_SUMMARY_DIR, "data.parquet")
    df_category_summary.to_parquet(
        category_summary_path, engine="pyarrow", compression="snappy", index=False
    )
    print(f"✓ category_summary 저장 완료 ({len(df_category_summary):,}개 카테고리)")

    # 3. detailed_stats 저장 (positive_special/negative_special 키 사용)
    print("\n[3/4] detailed_stats 카테고리별 파티션 생성 중...")

    stats_by_category = {}
    for product in all_products:
        product_id = product.get("product_id")
        category = "_".join(product_id.split("_")[:-1]) if "_" in product_id else "기타"
        sentiment = product.get("sentiment_analysis", {})

        for stype in ["positive_special", "negative_special"]:
            for kw in sentiment.get(stype, []):
                stats_by_category.setdefault(category, []).append(
                    {
                        "product_id": product_id,
                        "word": kw.get("word"),
                        "diff": kw.get("diff"),
                        "pos": kw.get("pos"),
                        "neg": kw.get("neg"),
                        "pos_n": kw.get("pos_n"),
                        "neg_n": kw.get("neg_n"),
                        "support": kw.get("support"),
                        "balanced_ratio": kw.get("balanced_ratio"),
                        "score": kw.get("score"),
                        "sentiment_type": (
                            "positive" if "positive" in stype else "negative"
                        ),
                    }
                )

    stats_count = 0
    for category, stats_data in stats_by_category.items():
        df_stats = pd.DataFrame(stats_data)
        partition_dir = os.path.join(DETAILED_STATS_DIR, f"category={category}")
        os.makedirs(partition_dir, exist_ok=True)

        stats_path = os.path.join(partition_dir, "data.parquet")
        df_stats.to_parquet(
            stats_path, engine="pyarrow", compression="snappy", index=False
        )
        stats_count += 1
        print(f"  ✓ {category}: {len(df_stats):,}개 키워드")

    print(f"✓ 총 {stats_count}개 카테고리 파티션 생성 완료")

    # 4. partitioned_reviews 저장 (기존 벡터 보존 + 커스텀 모델 벡터 추가)
    print("\n[4/4] partitioned_reviews 카테고리별 파티션 생성 중...")

    reviews_by_category = {}
    for review in all_reviews:
        product_id = review.get("product_id")
        category = "_".join(product_id.split("_")[:-1]) if "_" in product_id else "기타"

        review_dict = {
            "product_id": product_id,
            "id": review.get("id"),
            "full_text": review.get("full_text", ""),
            "title": review.get("title", ""),
            "content": review.get("content", ""),
            "has_text": bool(review.get("full_text", "").strip()),
            "score": review.get("score"),
            "label": review.get("label"),
            "tokens": review.get("tokens"),
            "char_length": review.get("char_length"),
            "token_count": review.get("token_count"),
            "date": review.get("date"),
            "collected_at": review.get("collected_at"),
            "nickname": review.get("nickname"),
            "has_image": review.get("has_image"),
            "helpful_count": review.get("helpful_count"),
            "sentiment_score": review.get("sentiment_score"),
        }

        # 벡터 필드 동적 추가 (선택된 모든 모델)
        for model_name in models_to_use:
            review_dict[model_name] = review.get(model_name)

        reviews_by_category.setdefault(category, []).append(review_dict)

    reviews_count = 0
    for category, review_data in reviews_by_category.items():
        df_reviews = pd.DataFrame(review_data)
        partition_dir = os.path.join(PARTITIONED_REVIEWS_DIR, f"category={category}")
        os.makedirs(partition_dir, exist_ok=True)

        reviews_path = os.path.join(partition_dir, "data.parquet")
        df_reviews.to_parquet(
            reviews_path, engine="pyarrow", compression="snappy", index=False
        )

        review_size_mb = os.path.getsize(reviews_path) / 1024 / 1024
        reviews_count += 1
        print(f"  ✓ {category}: {len(df_reviews):,}개 리뷰 ({review_size_mb:.2f} MB)")

    print(f"✓ 총 {reviews_count}개 카테고리 파티션 생성 완료")

    print("\n" + "=" * 80)
    print(f"✓ 모든 파일이 새 폴더에 저장되었습니다!")
    print(f"  - 출력 경로: {OUTPUT_DATA_DIR}")
    print(f"  - 사용된 모델: {', '.join(models_to_use)}")
    print("  - 통계 필드 재계산 완료")
    print("  - 카테고리별 파티셔닝 완료")
    print("=" * 80 + "\n")

    # ========== Colab: Google Drive 백업 ==========
    if exec_mode == "colab":
        print("\n" + "=" * 60)
        print("Google Drive 백업 시작")
        print("=" * 60)

        try:
            from google.colab import drive
            import shutil

            # Drive 마운트 (이미 마운트되어 있으면 스킵)
            if not os.path.exists("/content/drive"):
                print("\nDrive 마운트 중...")
                drive.mount("/content/drive")

            # 백업 경로 설정
            drive_backup_base = "/content/drive/MyDrive/multicampus_project_backup"
            drive_data = os.path.join(drive_backup_base, "data/new_processed_data")

            # 기존 백업 삭제 (덮어쓰기 위해)
            if os.path.exists(drive_data):
                print(f"\n기존 데이터 백업 삭제 중: {drive_data}")
                shutil.rmtree(drive_data)
                time.sleep(5)  # 구글 드라이브가 동기화할 시간을 2초 정도 줍니다.

            # 재벡터화 데이터 백업
            if os.path.exists(OUTPUT_DATA_DIR):
                print(f"\n재벡터화 데이터를 Drive로 백업 중...")
                shutil.copytree(OUTPUT_DATA_DIR, drive_data)
                backup_size = (
                    sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(drive_data)
                        for filename in filenames
                    )
                    / 1024
                    / 1024
                )
                print(f"✓ 데이터 백업 완료: {drive_data}")
                print(f"  - 크기: {backup_size:.1f} MB")

            print("\n" + "=" * 60)
            print("Drive 백업 완료!")
            print(f"백업 위치: {drive_backup_base}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"\n[경고] Drive 백업 실패: {e}")
            print("세션 종료 시 /content 데이터가 삭제될 수 있습니다.")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
