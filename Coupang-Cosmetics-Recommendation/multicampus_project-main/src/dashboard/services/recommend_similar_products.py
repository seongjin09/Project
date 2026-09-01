import os
import numpy as np
import pandas as pd
import glob
from typing import List, Optional, Dict, Any


# Athena 함수는 AWS 모드일 때만 lazy import
def _lazy_load_athena():
    from services.athena_queries import load_products_data_from_athena

    return load_products_data_from_athena


# BERTVectorizer 임포트 (로컬 모델 사용 시 필요, API 모드에서는 선택적)
import sys

sys.path.append("./services")

try:
    from bert_vectorizer import BERTVectorizer
except (ImportError, Exception) as e:
    # Streamlit Cloud나 API 전용 환경에서는 로드 실패 가능 (정상)
    BERTVectorizer = None
    print(f"[INFO] BERTVectorizer 로드 실패 (API 모드 사용 시 정상): {e}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1 is None or vec2 is None:
        return 0.0

    # 벡터가 리스트인 경우 numpy 배열로 변환
    if isinstance(vec1, list):
        vec1 = np.array(vec1)
    if isinstance(vec2, list):
        vec2 = np.array(vec2)

    # 벡터가 문자열인 경우 (예: "[-0.1676, ...]") numpy 배열로 변환
    if isinstance(vec1, str):
        try:
            vec1 = np.array(eval(vec1))
        except:
            return 0.0
    if isinstance(vec2, str):
        try:
            vec2 = np.array(eval(vec2))
        except:
            return 0.0

    # 0 벡터인 경우 처리
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def load_products_data_local(
    categories: Optional[List[str]] = None,
    vector_type: str = "roberta_semantic",
) -> pd.DataFrame:
    """
    로컬 Parquet에서 상품 데이터 로드 (벡터 포함)
    프로젝트 루트 기준: data/new_processed_data/integrated_products_final/
    """
    from pathlib import Path

    # 프로젝트 루트 계산: services/recommend_similar_products.py → 4단계 상위
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    products_dir = (
        project_root / "data" / "new_processed_data" / "integrated_products_final"
    )

    if not products_dir.exists():
        print(f"[오류] 로컬 데이터 디렉토리가 없습니다: {products_dir}")
        return pd.DataFrame()

    import pyarrow.dataset as ds
    import pyarrow as pa

    try:
        dataset = ds.dataset(products_dir, format="parquet", partitioning="hive")
        all_products = dataset.to_table().to_pandas()
    except pa.lib.ArrowTypeError:
        # 스키마 불일치(large_string vs string) 시 개별 파일 로드
        parquet_files = glob.glob(str(products_dir / "category=*" / "data.parquet"))
        dfs = []
        for f in parquet_files:
            part_df = pd.read_parquet(f)
            cat = f.split("category=")[1].split("/")[0]
            part_df["category"] = cat
            dfs.append(part_df)
        all_products = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if "category" in all_products.columns:
        import unicodedata

        all_products["category"] = (
            all_products["category"].astype(str).str.replace("_", "/", regex=False)
        )
        # macOS NFD → NFC 정규화 (파일시스템 인코딩 차이 해결)
        all_products["category"] = all_products["category"].apply(
            lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x
        )

    # 카테고리 필터링
    if categories:
        import unicodedata

        # 검색어도 NFC 정규화
        expanded = []
        for c in categories:
            c_nfc = unicodedata.normalize("NFC", c) if isinstance(c, str) else c
            expanded.append(c_nfc)
            expanded.append(c_nfc.replace("/", "_"))
            expanded.append(c_nfc.replace("_", "/"))
        expanded = list(set(expanded))
        all_products = all_products[all_products["category"].isin(expanded)]

    # 벡터 컬럼 확인
    vector_col = f"product_vector_{vector_type}"
    if vector_col not in all_products.columns:
        print(f"[경고] '{vector_col}' 컬럼이 없습니다.")

    return all_products


def load_products_data(
    processed_data_dir: str = "./data/processed_data",
    categories: Optional[List[str]] = None,
    vector_type: str = "roberta_semantic",
) -> pd.DataFrame:
    """
    상품 데이터 로드 (integrated_products_final)

    Args:
        processed_data_dir: processed_data 디렉토리 경로
        categories: 로드할 카테고리 리스트 (None이면 전체)
        vector_type: 사용할 벡터 타입 ("roberta", "bert", "koelectra", "word2vec")

    Returns:
        pd.DataFrame: 상품 데이터
    """
    products_final_dir = os.path.join(processed_data_dir, "integrated_products_final")

    # Hive 파티셔닝: category=*/data.parquet 패턴
    if categories is None:
        # 모든 카테고리 로드
        parquet_files = glob.glob(
            os.path.join(products_final_dir, "category=*", "data.parquet")
        )
    else:
        # 특정 카테고리만 로드
        parquet_files = []
        for category in categories:
            file_path = os.path.join(
                products_final_dir, f"category={category}", "data.parquet"
            )
            if os.path.exists(file_path):
                parquet_files.append(file_path)

    if not parquet_files:
        return pd.DataFrame()

    # 모든 파일 로드 및 병합
    dfs = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        dfs.append(df)

    all_products = pd.concat(dfs, ignore_index=True)

    # 필요한 컬럼 확인
    vector_col = f"product_vector_{vector_type}"
    if vector_col not in all_products.columns:
        raise ValueError(
            f"'{vector_col}' 컬럼이 존재하지 않습니다. "
            f"사용 가능한 벡터 타입을 확인하세요."
        )

    return all_products


def recommend_similar_products(
    product_id: Optional[str] = None,
    query_text: Optional[str] = None,
    categories: Optional[List[str]] = None,
    top_n: int = 10,
    processed_data_dir: str = "./data/processed_data",
    vector_type: str = "roberta_semantic",
    exclude_self: bool = True,
    vectorizer=None,
    data=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    유사 상품 추천, 문맥 검색, 전체 랭킹

    점수 계산 방식:
    - product_id가 있을 때 (유사 상품 추천 - roberta_sentiment 벡터 사용):
      점수 = 유사도 * 0.5 + 긍정확률 * 0.3 + 정규화_평점 * 0.2
    - query_text가 있을 때 (문맥 검색 - roberta_semantic 벡터 사용):
      점수 = (유사도^2) * 0.95 + 긍정확률 * 0.03 + 정규화_평점 * 0.02 (유사도 지수적 강화)
    - 둘 다 None일 때 (전체 랭킹):
      점수 = 긍정확률 * 0.6 + 정규화_평점 * 0.4
    - 정규화_평점 = 평균평점 / 5.0

    Args:
        product_id: 기준 상품 ID (예: "로션_1"), None이면 전체 랭킹
        query_text: 검색 문장 (예: "촉촉하고 하얗지 않은 선크림")
        categories: 검색할 카테고리 리스트 (None이면 모든 카테고리)
        top_n: 반환할 추천 상품 개수 (카테고리별)
        processed_data_dir: processed_data 디렉토리 경로
        vector_type: 사용할 벡터 타입
        exclude_self: 자기 자신을 결과에서 제외할지 여부 (product_id 모드에만 적용)
        vectorizer: BERTVectorizer 인스턴스 (query_text 사용 시 필요)
        data: 외부에서 필터링된 DataFrame (None이면 Athena에서 로드)

    Returns:
        Dict[str, List[Dict]]: 카테고리별 추천 상품 딕셔너리
            {
                "로션": [
                    {
                        "product_id": "로션_2",
                        "product_name": "상품명",
                        "brand": "브랜드",
                        "recommend_score": 0.85,
                        "cosine_similarity": 0.92,
                        "sentiment_score": 0.78,
                        "category": "로션",
                        ...
                    },
                    ...
                ],
                "크림": [...],
                ...
            }
    """
    # 1. product_id와 query_text 동시 사용 불가
    if product_id is not None and query_text is not None:
        raise ValueError("product_id와 query_text를 동시에 사용할 수 없습니다.")

    # 2. 상품 데이터 로드
    # data가 주어지면 그것을 사용, 아니면 Athena에서 로드
    if data is not None:
        # 외부에서 필터링된 데이터 사용 (피부 타입 등)
        print(f"외부 필터링 데이터 사용 ({len(data):,}개 상품)")
        if product_id is not None:
            # product_id 모드에서는 base_products와 all_products 모두 필요
            base_products = data
            target_product = base_products[base_products["product_id"] == product_id]

            if target_product.empty:
                print(f"[오류] 상품 ID '{product_id}'를 찾을 수 없습니다.")
                return {}

            target_product = target_product.iloc[0]
            print(f"✓ 기준 상품: {target_product.get('product_name', product_id)}")

            # categories 필터링
            if categories:
                all_products = data[data["category"].isin(categories)]
            else:
                all_products = data
        else:
            # query_text나 전체 랭킹 모드
            if categories:
                all_products = data[data["category"].isin(categories)]
            else:
                all_products = data
    else:
        # 데이터 소스 확인 (session_state에서)
        import streamlit as _st

        _data_source = _st.session_state.get("data_source", "local")

        if _data_source == "local":
            # 로컬 Parquet에서 로드
            all_products = load_products_data_local(
                categories=categories, vector_type=vector_type
            )
            if product_id is not None:
                target_product = all_products[all_products["product_id"] == product_id]
                if target_product.empty:
                    print(f"[오류] 상품 ID '{product_id}'를 찾을 수 없습니다.")
                    return {}
                target_product = target_product.iloc[0]
                print(f"✓ 기준 상품: {target_product.get('product_name', product_id)}")
                # 카테고리 필터링 (기준 상품은 전체에서 이미 찾음)
                if categories:
                    filtered = all_products[all_products["category"].isin(categories)]
                    # 기준 상품이 필터에 포함되지 않을 수 있으므로 합치기
                    all_products = pd.concat(
                        [
                            filtered,
                            all_products[all_products["product_id"] == product_id],
                        ]
                    ).drop_duplicates("product_id")
        else:
            # Athena에서 로드 (기존 방식)
            load_products_data_from_athena = _lazy_load_athena()
            if product_id is not None:
                print(f"기준 상품 로드 중... (product_id={product_id})")
                base_products = load_products_data_from_athena(
                    categories=None, vector_type=vector_type
                )

                target_product = base_products[
                    base_products["product_id"] == product_id
                ]

                if target_product.empty:
                    print(f"[오류] 상품 ID '{product_id}'를 찾을 수 없습니다.")
                    return {}

                target_product = target_product.iloc[0]
                print(f"✓ 기준 상품: {target_product.get('product_name', product_id)}")

                print(f"비교 대상 상품 로드 중... (카테고리: {categories or '전체'})")
                all_products = load_products_data_from_athena(
                    categories=categories, vector_type=vector_type
                )
            else:
                print(f"상품 데이터 로드 중... (카테고리: {categories or '전체'})")
                all_products = load_products_data_from_athena(
                    categories=categories, vector_type=vector_type
                )

    if all_products.empty:
        print("[경고] 상품 데이터를 찾을 수 없습니다.")
        return []

    print(f"✓ {len(all_products):,}개 상품 로드 완료")

    # 3. 모드에 따라 분기 처리
    target_vector = None
    target_product_name = None
    weights = None  # [유사도, 긍정확률, 정규화평점]
    is_semantic_search = False  # 문맥 검색 여부

    if product_id is not None:
        # 모드 1: 유사 상품 추천 (roberta_sentiment 벡터 사용)
        print(f"\n[모드] 유사 상품 추천 (product_id={product_id})")
        print("감성 분석 기반 roberta_sentiment 벡터 사용")

        # target_product는 이미 위에서 찾음
        target_product_name = target_product.get("product_name", product_id)
        vector_col = f"product_vector_{vector_type}"
        target_vector = target_product[vector_col]

        if target_vector is None or (
            isinstance(target_vector, (list, np.ndarray)) and len(target_vector) == 0
        ):
            print(f"[오류] 상품 '{product_id}'의 벡터가 없습니다.")
            return {}

        weights = [0.5, 0.3, 0.2]
        print(f"✓ 기준 상품: {target_product_name}")
        print("점수 = 유사도 * 0.5 + 긍정확률 * 0.3 + 정규화_평점 * 0.2")

    elif query_text is not None:
        # 모드 2: 문맥 검색 (roberta_semantic 벡터 사용)
        print(f"\n[모드] 문맥 검색 (query='{query_text}')")
        print("문맥 파악용 roberta_semantic 벡터 사용")

        if vectorizer is None:
            raise ValueError(
                "query_text를 사용하려면 vectorizer 파라미터가 필요합니다. "
                "roberta_semantic_final 모델로 초기화된 BERTVectorizer 인스턴스를 전달하세요."
            )

        # 쿼리를 벡터로 변환 (roberta_semantic 모델 사용)
        print("쿼리 벡터화 중... (roberta_semantic 모델)")
        target_vector = vectorizer.encode(query_text)

        # Semantic 벡터 컬럼으로 변경
        vector_col = f"product_vector_roberta_semantic"
        if vector_col not in all_products.columns:
            raise ValueError(
                f"'{vector_col}' 컬럼이 존재하지 않습니다. "
                f"semantic_vectorize.py를 먼저 실행하세요."
            )

        is_semantic_search = True
        weights = [
            0.95,
            0.03,
            0.02,
        ]  # 문맥 검색 시 유사도 비중 극대화 (유사도 제곱 적용)
        target_product_name = query_text
        print(f"✓ 검색 쿼리: {query_text}")
        print(
            "점수 = (유사도^2) * 0.95 + 긍정확률 * 0.03 + 정규화_평점 * 0.02 (유사도 지수적 강화)"
        )

    else:
        # 모드 3: 전체 랭킹 (유사도 없음)
        print(f"\n[모드] 전체 상품 랭킹")
        weights = [0.0, 0.6, 0.4]
        print("점수 = 긍정확률 * 0.6 + 정규화_평점 * 0.4")

    # 4. 모든 상품과 비교하여 점수 계산
    print(f"\n점수 계산 중...")
    results = []
    vector_col = (
        f"product_vector_{vector_type}"
        if product_id is not None
        else "product_vector_roberta_semantic"
    )

    for idx, product in all_products.iterrows():
        # 자기 자신 제외 (옵션, product_id 모드에만 적용)
        if (
            exclude_self
            and product_id is not None
            and product["product_id"] == product_id
        ):
            continue

        # sentiment_score 추출 (없으면 0.5로 기본값)
        sentiment = product.get("sentiment_score")
        if sentiment is None or pd.isna(sentiment):
            sentiment = 0.5

        # 정규화 평점 계산 (평균평점 / 5.0)
        avg_rating = product.get("avg_rating_with_text", 0)
        if avg_rating is None or pd.isna(avg_rating):
            avg_rating = 0
        normalized_rating = avg_rating / 5.0

        if target_vector is not None:
            # 유사 상품 추천 또는 문맥 검색 모드: 유사도 계산 필요
            product_vector = product[vector_col]

            # 벡터가 없으면 스킵
            if product_vector is None or (
                isinstance(product_vector, (list, np.ndarray))
                and len(product_vector) == 0
            ):
                continue

            # 코사인 유사도 계산
            similarity = cosine_similarity(target_vector, product_vector)

            # 최종 점수 계산 (가중치 적용)
            if is_semantic_search:
                # 문맥 검색 시 유사도를 제곱하여 지수적으로 강화
                recommend_score = (
                    (similarity**2) * weights[0]
                    + sentiment * weights[1]
                    + normalized_rating * weights[2]
                )
            else:
                # 유사 상품 추천
                recommend_score = (
                    similarity * weights[0]
                    + sentiment * weights[1]
                    + normalized_rating * weights[2]
                )
        else:
            # 전체 랭킹 모드: 유사도 불필요
            similarity = None

            # 최종 점수 = 긍정확률 * 0.6 + 정규화_평점 * 0.4
            recommend_score = sentiment * weights[1] + normalized_rating * weights[2]

        # 결과 저장
        result = {
            "product_id": product["product_id"],
            "product_name": product.get("product_name", ""),
            "brand": product.get("brand", ""),
            "category": product.get("category", ""),
            "price": product.get("price"),
            "recommend_score": float(recommend_score),
            "sentiment_score": float(sentiment),
            "normalized_rating": float(normalized_rating),
            "avg_rating": float(avg_rating),
            "total_reviews": product.get("total_reviews", 0),
            "avg_rating_with_text": product.get("avg_rating_with_text", 0),
            "top_keywords": product.get("top_keywords", []),
            "product_url": product.get("product_url", ""),
        }

        # 유사도는 product_id가 있을 때만 포함
        if similarity is not None:
            result["cosine_similarity"] = float(similarity)

        results.append(result)

    # 5. 카테고리별로 그룹화
    from collections import defaultdict

    category_results = defaultdict(list)

    for result in results:
        category = result["category"]
        category_results[category].append(result)

    # 5. 각 카테고리별로 점수 높은 순으로 정렬 후 상위 N개 선택
    final_results = {}
    total_count = 0

    for category, products in category_results.items():
        # 점수 높은 순으로 정렬
        products.sort(key=lambda x: x["recommend_score"], reverse=True)
        # 상위 N개 선택
        top_products = products[:top_n]
        final_results[category] = top_products
        total_count += len(top_products)

    print(f"✓ 추천 상품 {total_count}개 생성 완료 ({len(final_results)}개 카테고리)")
    for category, products in final_results.items():
        print(f"  - {category}: {len(products)}개")

    return final_results


def print_recommendations(recommendations: Dict[str, List[Dict[str, Any]]]):
    """
    추천 결과를 보기 좋게 출력

    Args:
        recommendations: recommend_similar_products() 결과 (카테고리별 딕셔너리)
    """
    if not recommendations:
        print("추천 결과가 없습니다.")
        return

    print("\n" + "=" * 100)
    print("추천 상품 목록 (카테고리별)")
    print("=" * 100)

    for category, products in recommendations.items():
        print(f"\n[{category}] - {len(products)}개 상품")
        print("-" * 110)
        print(
            f"{'순위':<5} {'상품명':<30} {'브랜드':<15} {'점수':<8} {'유사도':<8} {'감성':<8} {'평점':<8}"
        )
        print("-" * 110)

        for rank, rec in enumerate(products, 1):
            # None 값 처리
            product_name = rec["product_name"] or ""
            brand = rec["brand"] or ""

            # 길이 제한 적용
            product_name = (
                product_name[:28] + ".." if len(product_name) > 30 else product_name
            )
            brand = brand[:13] + ".." if len(brand) > 15 else brand

            # 유사도는 있을 때만 표시
            similarity_str = (
                f"{rec.get('cosine_similarity', 0.0):<8.3f}"
                if "cosine_similarity" in rec
                else "N/A     "
            )

            print(
                f"{rank:<5} "
                f"{product_name:<30} "
                f"{brand:<15} "
                f"{rec['recommend_score']:<8.3f} "
                f"{similarity_str} "
                f"{rec['sentiment_score']:<8.3f} "
                f"{rec.get('avg_rating', 0):<8.1f}"
            )

    print("\n" + "=" * 100)


# 사용 예시
if __name__ == "__main__":
    # 예시 1: 특정 카테고리에서 추천
    print("=" * 100)
    print("예시 1: 로션 카테고리에서 유사 상품 추천")
    print("=" * 100)

    results = recommend_similar_products(
        product_id="로션_1",
        categories=["로션"],
        top_n=2,
    )

    print_recommendations(results)
    print("\n\n")
    print("=" * 100)

    # 예시 2: 모든 카테고리에서 추천
    print("\n\n" + "=" * 100)
    print("예시 2: 모든 카테고리에서 유사 상품 추천")
    print("=" * 100)

    results = recommend_similar_products(
        product_id="로션_1",
        categories=None,
        top_n=2,
    )

    print_recommendations(results)

    # 예시 3: 상품 미입력 (필터링된것중 전체 랭킹)
    print("\n\n" + "=" * 100)
    print("예시 3: 상품 미입력시 전체 랭킹")
    print("=" * 100)

    results = recommend_similar_products(
        product_id=None,
        categories=None,
        top_n=2,
    )

    print_recommendations(results)

    # 예시 4: 문맥 검색 (BERTVectorizer 필요 - Semantic 모델)
    print("\n\n" + "=" * 100)
    print("예시 4: 문맥 검색 (Semantic 벡터 사용)")
    print("=" * 100)

    vectorizer = BERTVectorizer(model_name="./models/fine_tuned/roberta_semantic_final")

    results = recommend_similar_products(
        query_text="지성 피부에 좋은 기름지지 않은 묽은 로션",
        categories=None,
        top_n=5,
        vectorizer=vectorizer,
    )

    print_recommendations(results)

    print("\n\n" + "=" * 100)

    results = recommend_similar_products(
        query_text="건성 피부에 좋은 기름지고 꾸덕한 로션",
        categories=None,
        top_n=5,
        vectorizer=vectorizer,
    )

    print_recommendations(results)
