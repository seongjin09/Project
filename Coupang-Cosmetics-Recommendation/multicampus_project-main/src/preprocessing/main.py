"""
전처리 파이프라인 메인 orchestrator
"""

import json
import os
import sys
import glob
import time
from datetime import datetime
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from preprocessing_phases import (
    preprocess_and_tokenize_file,
    train_global_word2vec,
    vectorize_file,
    MAX_WORKERS,
)
from sentiment_analysis import analyze_skin_type_frequency

# utils 모듈 import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.environment import get_execution_mode

# ========== 실행 환경 설정 ==========
# "auto": 자동 감지 (Colab이면 순차, 아니면 병렬)
# "colab": Colab 강제 (순차 처리)
# "local": 로컬 강제 (병렬 처리)
EXECUTION_MODE = "auto"  # 여기를 변경하여 선택

# ========== 토크나이저 설정 ==========
# "kiwi": Kiwi (기본, 빠르고 정확)
# "okt": Okt (Twitter 한국어 처리기)
# "mecab": Mecab (속도 최고, 사전 설치 필요)
TOKENIZER = "mecab"  # 여기를 변경하여 선택

# ========== 벡터화 방법 설정 ==========
# 리스트로 여러 모델 동시 사용 가능
# 예: ["word2vec", "bert"] 또는 ["word2vec", "bert", "roberta", "koelectra"]
VECTORIZER_TYPE = ["roberta"]  # 여기를 변경하여 선택

# 모델별 설정
MODEL_CONFIGS = {
    "bert": "klue/bert-base",
    "roberta": "klue/roberta-base",
    "koelectra": "monologg/koelectra-base-v3-discriminator",
}

# ========== 리뷰 필터링 설정 ==========
MIN_REVIEWS_PER_PRODUCT = 30  # 이 개수 이하의 리뷰를 가진 상품 제외

# ========== 배치 사이즈 테스트 모드 ==========
BATCH_SIZE_TEST_MODE = False  # True면 Phase 3를 여러 배치로 테스트
BATCH_SIZES_TO_TEST = [16, 32, 64, 128, 256, 512, 1024]  # 테스트할 배치 사이즈 목록


# ========== 워커 프로세스 초기화 함수 ==========
def init_worker(tokenizer_name):
    """
    Pool 워커 프로세스 초기화 시 호출
    각 워커에서 토크나이저를 초기화
    """
    from preprocessing_utils import init_tokenizer

    init_tokenizer(tokenizer_name)


def main():
    """
    최적화된 전처리 파이프라인:
    Phase 1: 전처리 + 토큰화 (환경에 따라 병렬/순차)
    Phase 2: Iterator 방식 Word2Vec 학습 (메모리 효율적)
    Phase 3: 벡터화 + 대표 리뷰 선정 (환경에 따라 병렬/순차)
    """
    # 실행 환경 결정
    exec_mode = get_execution_mode(EXECUTION_MODE)
    use_parallel = exec_mode == "local"

    # Transformer 모델 사용 시 순차 처리 강제 (pickle 직렬화 문제)
    transformer_models_requested = [m for m in VECTORIZER_TYPE if m in MODEL_CONFIGS]
    if transformer_models_requested and use_parallel:
        print("[알림] Transformer 모델 사용 시 Phase 3는 순차 처리됩니다 (직렬화 제한)")
        use_parallel_phase3 = False
    else:
        use_parallel_phase3 = use_parallel

    # 시작 시간 기록
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 환경별 경로 설정
    if exec_mode == "colab":
        BASE_DIR = "/content/data"
        print("[알림] Colab 환경: /content 로컬 스토리지 사용 (빠른 I/O)")
    else:
        BASE_DIR = "./data"

    PRE_DATA_DIR = os.path.join(BASE_DIR, "pre_data")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
    TEMP_TOKENS_DIR = os.path.join(BASE_DIR, "temp_tokens")

    print("\n" + "=" * 60)
    print(f"{'최적화된 전처리 파이프라인 시작':^60}")
    print(f"{'시작 시간: ' + start_datetime:^60}")
    print(
        f"{'실행 환경: ' + exec_mode.upper() + ' (' + ('병렬' if use_parallel else '순차') + ')':^60}"
    )
    print(f"{'벡터화 모델: ' + ', '.join(VECTORIZER_TYPE):^60}")
    print(f"{'데이터 경로: ' + PRE_DATA_DIR:^60}")
    print("=" * 60 + "\n")

    # pre_data 디렉토리의 모든 JSON 파일 찾기
    json_files = glob.glob(os.path.join(PRE_DATA_DIR, "**", "*.json"), recursive=True)

    if not json_files:
        print(f"\n[오류] {PRE_DATA_DIR} 디렉토리에서 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(json_files)}개 파일 발견")
    print(f"병렬 처리 워커 수: {MAX_WORKERS}개\n")

    # ========== Phase 1: 전처리 + 토큰화 ==========
    print("=" * 60)
    print(f"Phase 1: 전처리 및 토큰화 ({'병렬' if use_parallel else '순차'} 처리)")
    print(f"토크나이저: {TOKENIZER.upper()}")
    print("=" * 60)

    # 토크나이저 초기화
    from preprocessing_utils import init_tokenizer

    init_tokenizer(TOKENIZER)

    phase1_start = time.time()
    os.makedirs(TEMP_TOKENS_DIR, exist_ok=True)

    args_list = [
        (
            input_path,
            PRE_DATA_DIR,
            PROCESSED_DATA_DIR,
            TEMP_TOKENS_DIR,
            MIN_REVIEWS_PER_PRODUCT,
        )
        for input_path in json_files
    ]

    skipped_count = 0
    phase1_results = []

    if use_parallel:
        # 병렬 처리 (로컬 환경) - 각 워커에서 토크나이저 초기화
        with Pool(MAX_WORKERS, initializer=init_worker, initargs=(TOKENIZER,)) as pool:
            for result in tqdm(
                pool.imap_unordered(preprocess_and_tokenize_file, args_list),
                total=len(json_files),
                desc="전처리 및 토큰화",
                unit="파일",
            ):
                if result["status"] == "skipped":
                    skipped_count += 1
                    tqdm.write(f"  [건너뜀] {result['file']}")
                elif result["status"] == "success":
                    phase1_results.append(result)
                    tqdm.write(
                        f"  [완료] {result['file']} - 토큰: {result['token_count']:,}개"
                    )
                else:
                    tqdm.write(
                        f"  [에러] {result['file']} - {result.get('error', 'Unknown')}"
                    )
    else:
        # 순차 처리 (Colab 환경 - JPype/Konlpy 충돌 방지)
        for args in tqdm(args_list, desc="전처리 및 토큰화", unit="파일"):
            result = preprocess_and_tokenize_file(args)

            if result["status"] == "skipped":
                skipped_count += 1
                print(f"  [건너뜀] {result['file']}")
            elif result["status"] == "success":
                phase1_results.append(result)
                print(f"  [완료] {result['file']} - 토큰: {result['token_count']:,}개")
            else:
                print(f"  [에러] {result['file']} - {result.get('error', 'Unknown')}")

    phase1_time = time.time() - phase1_start
    print(f"\nPhase 1 완료 - 소요 시간: {phase1_time:.2f}초")
    print(f"  처리 완료: {len(phase1_results)}개")
    print(f"  건너뜀: {skipped_count}개\n")

    # ========== Phase 2: 벡터화 모델 준비 ==========
    phase2_start = time.time()
    w2v_model = None
    vectorizers = {}  # 모델별 vectorizer 저장

    if "word2vec" in VECTORIZER_TYPE:
        print("\n" + "=" * 60)
        print("Phase 2-1: Word2Vec 모델 학습")
        print("=" * 60)
        w2v_model = train_global_word2vec(TEMP_TOKENS_DIR)
        if not w2v_model:
            print("[오류] Word2Vec 모델 학습 실패")
        else:
            # Word2Vec 모델 저장
            if exec_mode == "colab":
                w2v_model_path = "/content/models/word2vec_model.model"
            else:
                w2v_model_path = "./models/word2vec_model.model"
            os.makedirs(os.path.dirname(w2v_model_path), exist_ok=True)
            w2v_model.save(w2v_model_path)
            print(f"✓ Word2Vec 모델 저장 완료: {w2v_model_path}")
            print(f"  - 어휘 크기: {len(w2v_model.wv):,}개")
            print(f"  - 벡터 차원: {w2v_model.vector_size}차원")

    # Transformer 기반 모델들 로딩
    transformer_models = [m for m in VECTORIZER_TYPE if m in MODEL_CONFIGS]
    if transformer_models:
        print("\n" + "=" * 60)
        print(f"Phase 2-2: Transformer 모델 로딩 ({len(transformer_models)}개)")
        print("=" * 60)
        from bert_vectorizer import get_bert_vectorizer

        for model_name in transformer_models:
            model_path = MODEL_CONFIGS[model_name]
            print(f"\n{model_name.upper()} 로딩 중... ({model_path})")
            vectorizers[model_name] = get_bert_vectorizer(model_path)
            print(f"✓ {model_name.upper()} 로드 완료")

    phase2_time = time.time() - phase2_start
    print(f"\nPhase 2 완료 - 소요 시간: {phase2_time:.2f}초\n")

    # ========== Phase 3: 벡터화 + 대표 리뷰 선정 ==========
    # 배치 사이즈 테스트 모드
    if BATCH_SIZE_TEST_MODE:
        print("\n" + "=" * 60)
        print(f"{'배치 사이즈 테스트 모드':^60}")
        print(f"테스트할 배치 사이즈: {BATCH_SIZES_TO_TEST}")
        print("=" * 60 + "\n")

        batch_test_results = []

        for test_batch_size in BATCH_SIZES_TO_TEST:
            print("\n" + "=" * 60)
            print(f"배치 사이즈 {test_batch_size} 테스트 중...")
            print("=" * 60)

            phase3_start = time.time()

            vectorize_args = [
                (
                    result["base_name"],
                    TEMP_TOKENS_DIR,
                    result["output_dir"],
                    w2v_model,
                    vectorizers,
                    VECTORIZER_TYPE,
                    test_batch_size,  # 배치 사이즈 전달
                )
                for result in phase1_results
            ]

            all_products = []
            all_reviews = []
            total_model_times = {model_name: 0.0 for model_name in VECTORIZER_TYPE}

            # 순차 처리 (테스트의 일관성을 위해)
            for args in tqdm(
                vectorize_args, desc=f"배치 {test_batch_size}", unit="파일"
            ):
                result = vectorize_file(args)

                if result["status"] == "success":
                    all_products.extend(result["product_summaries"])
                    all_reviews.extend(result["review_details"])
                    for model_name, model_time in result.get("model_times", {}).items():
                        total_model_times[model_name] += model_time

            phase3_time = time.time() - phase3_start

            # 결과 저장
            test_result = {
                "batch_size": test_batch_size,
                "total_time": phase3_time,
                "products_count": len(all_products),
                "reviews_count": len(all_reviews),
                "model_times": total_model_times.copy(),
            }
            batch_test_results.append(test_result)

            print(f"\n배치 {test_batch_size} 완료:")
            print(f"  - 총 소요 시간: {phase3_time:.2f}초")
            print(f"  - 처리 상품: {len(all_products):,}개")
            print(f"  - 처리 리뷰: {len(all_reviews):,}개")
            for model_name in sorted(VECTORIZER_TYPE):
                print(
                    f"  - {model_name.upper()}: {total_model_times[model_name]:.2f}초"
                )

        # 테스트 결과 종합
        print("\n" + "=" * 80)
        print(f"{'배치 사이즈 테스트 결과 요약':^80}")
        print("=" * 80)
        print(f"{'배치':>10} {'총 시간':>12} {'상품수':>10} {'리뷰수':>10}", end="")
        for model_name in sorted(VECTORIZER_TYPE):
            print(f" {model_name.upper():>10}", end="")
        print()
        print("-" * 80)

        for result in batch_test_results:
            print(
                f"{result['batch_size']:>10} {result['total_time']:>11.2f}s {result['products_count']:>10,} {result['reviews_count']:>10,}",
                end="",
            )
            for model_name in sorted(VECTORIZER_TYPE):
                model_time = result["model_times"].get(model_name, 0.0)
                print(f" {model_time:>9.2f}s", end="")
            print()

        # 가장 빠른 배치 사이즈 찾기
        fastest = min(batch_test_results, key=lambda x: x["total_time"])
        print("\n" + "=" * 80)
        print(
            f"✓ 가장 빠른 배치 사이즈: {fastest['batch_size']} ({fastest['total_time']:.2f}초)"
        )
        print("=" * 80 + "\n")

        # 테스트 모드에서는 여기서 종료
        return

    # 일반 모드: Phase 3 한 번만 실행
    print("=" * 60)
    print(
        f"Phase 3: 벡터화 및 대표 리뷰 선정 ({'병렬' if use_parallel_phase3 else '순차'} 처리)"
    )
    print("=" * 60)

    phase3_start = time.time()

    vectorize_args = [
        (
            result["base_name"],
            TEMP_TOKENS_DIR,
            result["output_dir"],
            w2v_model,
            vectorizers,  # dict로 전달
            VECTORIZER_TYPE,  # list로 전달
            None,  # 배치 사이즈 (None이면 bert_vectorizer에서 자동 설정)
        )
        for result in phase1_results
    ]

    all_products = []
    all_reviews = []

    # 모델별 시간 집계
    total_model_times = {model_name: 0.0 for model_name in VECTORIZER_TYPE}

    if use_parallel_phase3:
        # 병렬 처리 (로컬 환경, word2vec만 사용 시) - 각 워커에서 토크나이저 초기화
        with Pool(MAX_WORKERS, initializer=init_worker, initargs=(TOKENIZER,)) as pool:
            for result in tqdm(
                pool.imap_unordered(vectorize_file, vectorize_args),
                total=len(vectorize_args),
                desc="벡터화 및 대표 리뷰 선정",
                unit="파일",
            ):
                if result["status"] == "success":
                    all_products.extend(result["product_summaries"])
                    all_reviews.extend(result["review_details"])
                    # 모델별 시간 집계
                    for model_name, model_time in result.get("model_times", {}).items():
                        total_model_times[model_name] += model_time
                    tqdm.write(f"  [완료] {result['file']}")
                else:
                    tqdm.write(
                        f"  [에러] {result['file']} - {result.get('error', 'Unknown')}"
                    )
    else:
        # 순차 처리 (Colab 환경 또는 Transformer 사용 시)
        for args in tqdm(vectorize_args, desc="벡터화 및 대표 리뷰 선정", unit="파일"):
            result = vectorize_file(args)

            if result["status"] == "success":
                all_products.extend(result["product_summaries"])
                all_reviews.extend(result["review_details"])
                # 모델별 시간 집계
                for model_name, model_time in result.get("model_times", {}).items():
                    total_model_times[model_name] += model_time
                print(f"  [완료] {result['file']}")
            else:
                print(f"  [에러] {result['file']} - {result.get('error', 'Unknown')}")

    phase3_time = time.time() - phase3_start
    print(f"\nPhase 3 완료 - 소요 시간: {phase3_time:.2f}초")
    print(f"처리된 상품: {len(all_products):,}개")
    print(f"처리된 리뷰: {len(all_reviews):,}개")

    # 모델별 처리 시간 출력
    print(f"\n[모델별 벡터화 소요 시간]")
    for model_name in sorted(VECTORIZER_TYPE):
        model_time = total_model_times.get(model_name, 0.0)
        print(f"  - {model_name.upper()}: {model_time:.2f}초")

    # ========== 새로운 Parquet 구조 생성 ==========
    print("=" * 60)
    print("새로운 Parquet 구조 생성 중...")
    print("=" * 60)

    # 출력 디렉토리 (Hive 파티셔닝 형식)
    # 이미 위에서 PROCESSED_DATA_DIR를 환경에 맞게 설정했으므로 그대로 사용
    DATA_DIR = PROCESSED_DATA_DIR
    CATEGORY_SUMMARY_DIR = os.path.join(DATA_DIR, "category_summary")
    PRODUCTS_FINAL_DIR = os.path.join(DATA_DIR, "integrated_products_final")
    DETAILED_STATS_DIR = os.path.join(DATA_DIR, "detailed_stats")
    PARTITIONED_REVIEWS_DIR = os.path.join(DATA_DIR, "partitioned_reviews")

    os.makedirs(CATEGORY_SUMMARY_DIR, exist_ok=True)
    os.makedirs(PRODUCTS_FINAL_DIR, exist_ok=True)
    os.makedirs(DETAILED_STATS_DIR, exist_ok=True)
    os.makedirs(PARTITIONED_REVIEWS_DIR, exist_ok=True)

    # 1. integrated_products_final.parquet 생성
    print("\n[1/3] integrated_products_final.parquet 생성 중...")

    if not all_products:
        print("[경고] all_products가 비어있습니다. Phase 3 결과를 확인하세요.")
        print("전처리를 처음부터 다시 실행해야 할 수 있습니다.")
        return

    products_final = []
    for product in all_products:
        product_id = product.get("product_id")

        # 텍스트 유무별 리뷰 통계 계산
        product_reviews = [r for r in all_reviews if r.get("product_id") == product_id]

        text_reviews = [r for r in product_reviews if r.get("full_text")]
        no_text_reviews = [r for r in product_reviews if not r.get("full_text")]

        avg_rating_with_text = (
            sum(r.get("score", 0) for r in text_reviews) / len(text_reviews)
            if text_reviews
            else 0.0
        )
        avg_rating_without_text = (
            sum(r.get("score", 0) for r in no_text_reviews) / len(no_text_reviews)
            if no_text_reviews
            else 0.0
        )
        total_reviews = len(product_reviews)
        text_review_ratio = (
            len(text_reviews) / total_reviews if total_reviews > 0 else 0.0
        )

        # rating_distribution 계산 (1~5점 각각 몇 개인지)
        rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in product_reviews:
            score = r.get("score")
            if score in rating_dist:
                rating_dist[score] += 1

        # top_keywords 추출 (긍정 키워드 상위 5개)
        sentiment = product.get("sentiment_analysis", {})
        pos_keywords = sentiment.get("positive_special", [])[:5]
        top_keywords = [kw.get("word", "") for kw in pos_keywords]

        # product_id에서 카테고리 추출
        if "_" in product_id:
            category = "_".join(product_id.split("_")[:-1])
        else:
            category = "기타"

        # category_path에서 마지막 경로 추출
        category_path_str = product.get("category_path", "")
        path = category_path_str.split(" > ")[-1] if category_path_str else ""

        products_final_dict = {
            # 원본 product_info 필드
            "product_id": product_id,
            "product_name": product.get("product_name"),
            "brand": product.get("brand"),
            "category": category,
            "category_path": category_path_str,
            "path": path,
            "price": product.get("price"),
            "delivery_type": product.get("delivery_type"),
            "product_url": product.get("product_url"),
            "img_url": product.get("img_url"),
            # 전처리 추가 필드
            "skin_type": product.get("skin_type", "미분류"),
            "top_keywords": top_keywords,
            "sentiment_analysis": product.get("sentiment_analysis"),
            # 통계 필드
            "avg_rating_with_text": avg_rating_with_text,
            "avg_rating_without_text": avg_rating_without_text,
            "text_review_ratio": text_review_ratio,
            "total_reviews": total_reviews,
            "rating_1": rating_dist[1],
            "rating_2": rating_dist[2],
            "rating_3": rating_dist[3],
            "rating_4": rating_dist[4],
            "rating_5": rating_dist[5],
        }

        # 벡터 필드 동적 추가 (word2vec, bert, roberta, koelectra)
        for model_name in VECTORIZER_TYPE:
            products_final_dict[f"product_vector_{model_name}"] = product.get(
                f"product_vector_{model_name}"
            )
            products_final_dict[f"representative_review_id_{model_name}"] = product.get(
                f"representative_review_id_{model_name}"
            )
            products_final_dict[f"representative_similarity_{model_name}"] = (
                product.get(f"representative_similarity_{model_name}")
            )

        products_final.append(products_final_dict)

    df_products_final = pd.DataFrame(products_final)

    # product_id로 정렬 (카테고리_숫자 형식에서 숫자 기준으로 정렬)
    def extract_numeric_sort_key(product_id):
        """product_id에서 카테고리와 숫자를 분리하여 정렬 키 생성"""
        parts = product_id.rsplit("_", 1)  # 마지막 언더스코어 기준으로 분리
        if len(parts) == 2:
            category, num_str = parts
            try:
                return (category, int(num_str))  # 카테고리명, 숫자(정수)로 반환
            except ValueError:
                return (category, 0)  # 숫자 변환 실패 시 0으로 처리
        return (product_id, 0)  # 언더스코어가 없으면 원본 그대로

    df_products_final["_sort_key"] = df_products_final["product_id"].apply(
        extract_numeric_sort_key
    )
    df_products_final = df_products_final.sort_values("_sort_key").reset_index(
        drop=True
    )
    df_products_final = df_products_final.drop(columns=["_sort_key"])  # 임시 컬럼 제거

    # integrated_products_final 저장 (Hive 파티셔닝 - 카테고리별)
    print("\nintegrated_products_final 카테고리별 저장 중...")
    products_count = 0
    for category in (
        df_products_final["product_id"]
        .apply(lambda x: "_".join(x.split("_")[:-1]) if "_" in x else "기타")
        .unique()
    ):
        df_category = df_products_final[
            df_products_final["product_id"].str.startswith(category + "_")
        ]

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

    # 1-1. category_summary.parquet 생성 (카테고리별 전체 통계)
    print("\n[1-1] category_summary.parquet 생성 중...")

    category_stats = {}

    # all_reviews를 순회하여 카테고리별 통계 집계
    for review in all_reviews:
        product_id = review.get("product_id")
        if "_" in product_id:
            category = "_".join(product_id.split("_")[:-1])
        else:
            category = "기타"

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

        full_text = review.get("full_text", "")
        if full_text and full_text.strip():
            stats["text_reviews"] += 1
        else:
            stats["no_text_reviews"] += 1

        score = review.get("score")
        if score in [1, 2, 3, 4, 5]:
            stats[f"rating_{score}"] += 1

    df_category_summary = pd.DataFrame(list(category_stats.values()))
    # 1. category_summary 저장 (Hive 파티셔닝)
    category_summary_path = os.path.join(CATEGORY_SUMMARY_DIR, "data.parquet")
    df_category_summary.to_parquet(
        category_summary_path, engine="pyarrow", compression="snappy", index=False
    )

    print(f"✓ 저장 완료: {category_summary_path}")
    print(f"  - 카테고리 수: {len(df_category_summary):,}개")
    for _, row in df_category_summary.iterrows():
        print(
            f"  • {row['category']}: 전체 {row['total_reviews']:,}개 (텍스트 {row['text_reviews']:,}개, 없음 {row['no_text_reviews']:,}개)"
        )

    # 2. detailed_stats/ 카테고리별 파티션 생성
    print("\n[2/3] detailed_stats 카테고리별 파티션 생성 중...")

    stats_by_category = {}
    for product in all_products:
        product_id = product.get("product_id")

        # product_id에서 카테고리 추출 (예: "선크림_123" -> "선크림")
        # product_id는 "카테고리_원본ID" 형식 (마지막 언더스코어 기준으로 분리)
        if "_" in product_id:
            category = "_".join(product_id.split("_")[:-1])
        else:
            category = "기타"

        sentiment = product.get("sentiment_analysis", {})

        # 긍정 키워드
        for kw in sentiment.get("positive_special", []):
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
                    "sentiment_type": "positive",
                }
            )

        # 부정 키워드
        for kw in sentiment.get("negative_special", []):
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
                    "sentiment_type": "negative",
                }
            )

    stats_count = 0
    for category, stats_data in stats_by_category.items():
        df_stats = pd.DataFrame(stats_data)
        # Hive 파티션 디렉토리 생성
        partition_dir = os.path.join(DETAILED_STATS_DIR, f"category={category}")
        os.makedirs(partition_dir, exist_ok=True)

        stats_path = os.path.join(partition_dir, "data.parquet")
        df_stats.to_parquet(
            stats_path, engine="pyarrow", compression="snappy", index=False
        )
        stats_count += 1
        print(f"  ✓ {category}: {len(df_stats):,}개 키워드")

    print(f"✓ 총 {stats_count}개 카테고리 파티션 생성 완료")

    # 3. partitioned_reviews/ 카테고리별 파티션 생성
    print("\n[3/3] partitioned_reviews 카테고리별 파티션 생성 중...")

    # 리뷰를 카테고리별로 그룹화
    reviews_by_category = {}
    for review in all_reviews:
        product_id = review.get("product_id")
        # product_id에서 카테고리 추출 (예: "선크림_123" -> "선크림")
        # product_id는 "카테고리_원본ID" 형식 (마지막 언더스코어 기준으로 분리)
        if "_" in product_id:
            category = "_".join(product_id.split("_")[:-1])
        else:
            category = "기타"

        full_text = review.get("full_text", "")
        has_text = bool(full_text and full_text.strip())

        review_dict = {
            "product_id": product_id,
            "id": review.get("id"),
            "full_text": full_text,
            "title": review.get("title", ""),
            "content": review.get("content", ""),
            "has_text": has_text,
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
            "sentiment_score": None,  # 나중에 모델 예측값 추가
        }

        # 벡터 필드 동적 추가
        for model_name in VECTORIZER_TYPE:
            review_dict[model_name] = review.get(model_name)

        reviews_by_category.setdefault(category, []).append(review_dict)

    reviews_count = 0
    for category, review_data in reviews_by_category.items():
        df_reviews = pd.DataFrame(review_data)
        # Hive 파티션 디렉토리 생성
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

    # ========== 임시 파일 유지 (재벡터화를 위해) ==========
    print(f"\n✓ 토큰 파일 유지: {TEMP_TOKENS_DIR}")
    print("  (재벡터화 시 필요하므로 삭제하지 않음)")

    # 종료 시간 및 소요 시간 계산
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print("\n" + "=" * 60)
    print(f"{'전체 파이프라인 완료!':^60}")
    print(f"{'종료 시간: ' + end_datetime:^60}")
    print(f"{'총 소요 시간: ' + f'{hours}시간 {minutes}분 {seconds}초':^60}")
    print(
        f"{'Phase 1: ' + f'{phase1_time:.1f}초 | Phase 2: {phase2_time:.1f}초 | Phase 3: {phase3_time:.1f}초':^60}"
    )
    print("=" * 60 + "\n")

    # ========== Colab: Google Drive 백업 ==========
    if exec_mode == "colab":
        print("=" * 60)
        print("Google Drive 백업 시작")
        print("=" * 60)

        try:
            from google.colab import drive

            # Drive 마운트 (이미 마운트되어 있으면 스킵)
            if not os.path.exists("/content/drive"):
                print("\nDrive 마운트 중...")
                drive.mount("/content/drive")

            import shutil

            # 백업 경로 설정
            drive_backup_base = "/content/drive/MyDrive/multicampus_project_backup"
            drive_processed = os.path.join(drive_backup_base, "data", "processed_data")
            drive_models = os.path.join(drive_backup_base, "models")
            drive_temp_tokens = os.path.join(drive_backup_base, "data", "temp_tokens")

            # 기존 백업 삭제 (덮어쓰기 위해)
            if os.path.exists(drive_processed):
                print(f"\n기존 백업 삭제 중: {drive_processed}")
                shutil.rmtree(drive_processed)
            if os.path.exists(drive_models):
                print(f"기존 모델 백업 삭제 중: {drive_models}")
                shutil.rmtree(drive_models)
            if os.path.exists(drive_temp_tokens):
                print(f"기존 temp_tokens 백업 삭제 중: {drive_temp_tokens}")
                shutil.rmtree(drive_temp_tokens)

            # processed_data 백업
            print(f"\n처리된 데이터를 Drive로 백업 중...")
            if os.path.exists(PROCESSED_DATA_DIR):
                shutil.copytree(PROCESSED_DATA_DIR, drive_processed)
                backup_size = (
                    sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(drive_processed)
                        for filename in filenames
                    )
                    / 1024
                    / 1024
                )
                print(f"✓ 데이터 백업 완료: {drive_processed}")
                print(f"  - 크기: {backup_size:.1f} MB")

            # models 백업
            local_models = "/content/models"
            if os.path.exists(local_models):
                print(f"\n모델을 Drive로 백업 중...")
                shutil.copytree(local_models, drive_models)
                print(f"✓ 모델 백업 완료: {drive_models}")

            # temp_tokens 백업
            if os.path.exists(TEMP_TOKENS_DIR):
                print(f"\ntemp_tokens을 Drive로 백업 중...")
                shutil.copytree(TEMP_TOKENS_DIR, drive_temp_tokens)
                temp_tokens_size = (
                    sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(drive_temp_tokens)
                        for filename in filenames
                    )
                    / 1024
                    / 1024
                )
                print(f"✓ temp_tokens 백업 완료: {drive_temp_tokens}")
                print(f"  - 크기: {temp_tokens_size:.1f} MB")

            print("\n" + "=" * 60)
            print("Drive 백업 완료!")
            print(f"백업 위치: {drive_backup_base}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"\n[경고] Drive 백업 실패: {e}")
            print("세션 종료 시 /content 데이터가 삭제될 수 있습니다.")


if __name__ == "__main__":
    main()
