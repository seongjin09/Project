"""
문맥 파악용(Semantic) 벡터 추가
기존 new_processed_data에 Semantic 벡터 필드만 추가

목적:
- 긍부정 라벨링 기반 벡터 → 문맥 파악 기반 벡터로 전환
- "촉촉하고 하얗지 않은 선크림" 같은 의미 기반 검색 지원
- 키워드 유사도가 아닌 문맥 유사도 기반 추천

추가되는 필드:
- 리뷰: roberta_semantic (개별 리뷰 벡터)
- 상품: product_vector_roberta_semantic (상품 대표 벡터)
- 상품: representative_review_id_roberta_semantic
- 상품: representative_similarity_roberta_semantic
"""

import os
import sys
import glob
import time
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
    DATA_DIR = "/content/data/new_processed_data"
    MODELS_BASE_DIR = "/content/models/fine_tuned"
    print("[알림] Colab 환경: /content 로컬 스토리지 사용 (빠른 I/O)")
else:
    TEMP_TOKENS_DIR = "./data/temp_tokens"
    DATA_DIR = "./data/new_processed_data"
    MODELS_BASE_DIR = "./models/fine_tuned"

# ========== Semantic 모델 전용 설정 ==========
# roberta_semantic만 사용 (문맥 파악용)
MODELS_TO_USE = ["roberta_semantic"]

# 모델 경로 설정 (SimCSE로 학습된 의미 유사도 모델)
MODEL_PATHS = {
    "roberta_semantic": os.path.join(MODELS_BASE_DIR, "roberta_semantic_final"),
}


def main():
    print("\n" + "=" * 80)
    print(f"{'Semantic 벡터 필드 추가 (new_processed_data 업데이트)':^80}")
    print("=" * 80 + "\n")

    print(f"실행 환경: {exec_mode.upper()}")
    print(f"데이터 경로: {DATA_DIR}")
    print(f"모델 경로: {MODELS_BASE_DIR}")
    print(f"사용 모델: roberta_semantic_final (SimCSE 학습)")
    print(f"작업: 기존 파일에 semantic 벡터 필드 추가\n")

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
        print("preprocessing/main.py를 먼저 실행하세요.")
        return

    print(f"발견된 토큰 파일: {len(tokenized_files)}개\n")

    # 모델 로딩
    print("=" * 60)
    print("Semantic 모델 로딩 중...")
    print("=" * 60 + "\n")

    vectorizers = {}
    models_to_use = []

    for model_name in MODELS_TO_USE:
        if model_name not in MODEL_PATHS:
            print(f"[오류] {model_name}은(는) 지원되지 않는 모델입니다.")
            return

        model_path = MODEL_PATHS[model_name]

        # Semantic Transformer 모델 로드
        if os.path.exists(model_path):
            print(f"{model_name.upper()} 모델 로딩... ({model_path})")
            vectorizers[model_name] = get_bert_vectorizer(model_path)
            models_to_use.append(model_name)
            print(f"✓ {model_name.upper()} 로드 완료")
            print("  - 학습 방법: SimCSE (Contrastive Learning)")
            print("  - 특화 기능: 의미 기반 문맥 파악")
        else:
            print(f"[오류] {model_name} 모델 없음.")
            print(f"      경로: {model_path}")
            print("\n먼저 fine_tune_semantic_model.py를 실행하여 모델을 학습하세요:")
            print("  python src/model/fine_tune_semantic_model.py")
            return

    print(f"\n사용할 모델: {', '.join(models_to_use)}")
    print("배치 사이즈: GPU에 따라 자동 설정\n")

    # Semantic 벡터화 실행 (토큰 파일에서)
    print("=" * 60)
    print("Semantic 벡터 생성 중...")
    print("=" * 60 + "\n")

    start_time = time.time()
    semantic_products = {}  # product_id를 키로 사용
    semantic_reviews = {}  # (product_id, review_id)를 키로 사용

    for tokenized_file in tqdm(tokenized_files, desc="벡터화", unit="파일"):
        base_name = os.path.basename(tokenized_file).replace("_tokenized.pkl", "")

        # 임시 디렉토리에 결과 저장
        temp_output_dir = "./temp_semantic_output"
        os.makedirs(temp_output_dir, exist_ok=True)

        args = (
            base_name,
            TEMP_TOKENS_DIR,
            temp_output_dir,
            None,  # word2vec 사용 안 함
            vectorizers,  # Semantic Transformer 모델
            models_to_use,  # roberta_semantic만 사용
            batch_size,
        )

        result = vectorize_file(args)

        if result["status"] == "success":
            # 상품 벡터 저장
            for product in result["product_summaries"]:
                product_id = product.get("product_id")
                semantic_products[product_id] = {
                    "product_vector_roberta_semantic": product.get(
                        "product_vector_roberta_semantic"
                    ),
                    "representative_review_id_roberta_semantic": product.get(
                        "representative_review_id_roberta_semantic"
                    ),
                    "representative_similarity_roberta_semantic": product.get(
                        "representative_similarity_roberta_semantic"
                    ),
                }

            # 리뷰 벡터 저장
            for review in result["review_details"]:
                product_id = review.get("product_id")
                review_id = review.get("id")
                semantic_reviews[(product_id, review_id)] = {
                    "roberta_semantic": review.get("roberta_semantic")
                }

            tqdm.write(f"  [완료] {base_name}")
        else:
            tqdm.write(f"  [에러] {base_name} - {result.get('error', 'Unknown')}")

    # 임시 폴더 삭제
    import shutil

    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"Semantic 벡터 생성 완료 - 소요 시간: {elapsed_time:.2f}초")
    print(
        f"총 상품: {len(semantic_products):,}개, 총 리뷰: {len(semantic_reviews):,}개"
    )
    print("=" * 80 + "\n")

    # ========== 기존 Parquet 파일에 필드 추가 ==========
    print("=" * 80)
    print(f"기존 파일에 Semantic 벡터 필드 추가 중: {DATA_DIR}")
    print("=" * 80 + "\n")

    import pandas as pd

    # 기존 폴더 경로
    PRODUCTS_FINAL_DIR = os.path.join(DATA_DIR, "integrated_products_final")
    PARTITIONED_REVIEWS_DIR = os.path.join(DATA_DIR, "partitioned_reviews")

    # 1. integrated_products_final 업데이트
    print("[1/2] integrated_products_final에 semantic 벡터 추가 중...")

    # 기존 카테고리별 파티션 파일들 읽기
    product_partitions = glob.glob(
        os.path.join(PRODUCTS_FINAL_DIR, "category=*", "data.parquet")
    )

    if not product_partitions:
        print(f"[오류] {PRODUCTS_FINAL_DIR}에 파티션 파일이 없습니다.")
        print("re_vectorize.py를 먼저 실행하세요.")
        return

    updated_products = 0
    for partition_path in tqdm(product_partitions, desc="상품 파티션 업데이트"):
        # 기존 파일 읽기
        df_products = pd.read_parquet(partition_path)

        # Semantic 벡터 필드 추가
        df_products["product_vector_roberta_semantic"] = df_products["product_id"].map(
            lambda pid: semantic_products.get(pid, {}).get(
                "product_vector_roberta_semantic"
            )
        )
        df_products["representative_review_id_roberta_semantic"] = df_products[
            "product_id"
        ].map(
            lambda pid: semantic_products.get(pid, {}).get(
                "representative_review_id_roberta_semantic"
            )
        )
        df_products["representative_similarity_roberta_semantic"] = df_products[
            "product_id"
        ].map(
            lambda pid: semantic_products.get(pid, {}).get(
                "representative_similarity_roberta_semantic"
            )
        )

        # 덮어쓰기
        df_products.to_parquet(
            partition_path, engine="pyarrow", compression="snappy", index=False
        )
        updated_products += len(df_products)

    print(
        f"✓ {len(product_partitions)}개 파티션, {updated_products:,}개 상품 업데이트 완료"
    )

    # 2. partitioned_reviews 업데이트
    print("\n[2/2] partitioned_reviews에 semantic 벡터 추가 중...")

    # 기존 카테고리별 파티션 파일들 읽기
    review_partitions = glob.glob(
        os.path.join(PARTITIONED_REVIEWS_DIR, "category=*", "data.parquet")
    )

    if not review_partitions:
        print(f"[오류] {PARTITIONED_REVIEWS_DIR}에 파티션 파일이 없습니다.")
        return

    updated_reviews = 0
    for partition_path in tqdm(review_partitions, desc="리뷰 파티션 업데이트"):
        # 기존 파일 읽기
        df_reviews = pd.read_parquet(partition_path)

        # Semantic 벡터 필드 추가
        df_reviews["roberta_semantic"] = df_reviews.apply(
            lambda row: semantic_reviews.get((row["product_id"], row["id"]), {}).get(
                "roberta_semantic"
            ),
            axis=1,
        )

        # 덮어쓰기
        df_reviews.to_parquet(
            partition_path, engine="pyarrow", compression="snappy", index=False
        )
        updated_reviews += len(df_reviews)

    print(
        f"✓ {len(review_partitions)}개 파티션, {updated_reviews:,}개 리뷰 업데이트 완료"
    )

    print("\n" + "=" * 80)
    print(f"✓ Semantic 벡터 필드 추가 완료!")
    print(f"  - 업데이트 경로: {DATA_DIR}")
    print(f"  - 사용된 모델: roberta_semantic (SimCSE 학습)")
    print(f"  - 추가된 리뷰 필드: roberta_semantic")
    print(f"  - 추가된 상품 필드: product_vector_roberta_semantic")
    print(f"  - 기존 roberta 벡터 유지 + semantic 벡터 추가")
    print("=" * 80 + "\n")

    print("다음 단계:")
    print("  1. recommend_similar_products.py 수정:")
    print("     - model_name 파라미터 추가 (roberta or roberta_semantic)")
    print("  2. query_text로 '촉촉하고 하얗지 않은 선크림' 검색 테스트")
    print("  3. 긍부정 모델 vs 의미 모델 성능 비교")

    # ========== Colab: Google Drive 백업 ==========
    if exec_mode == "colab":
        print("\n" + "=" * 60)
        print("Google Drive 백업 시작")
        print("=" * 60)

        try:
            from google.colab import drive

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

            # 업데이트된 데이터 백업
            if os.path.exists(DATA_DIR):
                print(f"\n업데이트된 데이터를 Drive로 백업 중...")
                shutil.copytree(DATA_DIR, drive_data)
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
