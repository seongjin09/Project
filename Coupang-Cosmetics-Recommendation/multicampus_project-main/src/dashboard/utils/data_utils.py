"""
데이터 로딩 및 컬럼 정규화 유틸리티
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

from utils.load_data import make_df, load_raw_df


# =========================
# 데이터 소스 헬퍼
# =========================
def _get_data_source() -> str:
    """현재 데이터 소스 설정 반환 ('local' 또는 'aws')"""
    return st.session_state.get("data_source", "local")


def _get_project_root() -> Path:
    """프로젝트 루트 디렉토리 반환"""
    # src/dashboard/utils/data_utils.py → 3단계 상위 = src/dashboard
    dashboard_dir = Path(__file__).resolve().parent.parent
    # src/dashboard → 2단계 상위 = 프로젝트 루트
    return dashboard_dir.parent.parent


def _get_local_data_dir() -> Path:
    """로컬 데이터 디렉토리 경로 반환"""
    return _get_project_root() / "data" / "new_processed_data"


def _import_athena_functions():
    """AWS 모드일 때만 Athena 함수 import (lazy)"""
    from services.athena_queries import (
        fetch_all_products,
        fetch_reviews_by_product,
        fetch_top_reviews_text,
    )

    return fetch_all_products, fetch_reviews_by_product, fetch_top_reviews_text


# =========================
# 로컬 데이터 로딩 함수들
# =========================
@st.cache_data(show_spinner=False)
def load_products_local() -> pd.DataFrame:
    """로컬 Parquet에서 전체 상품 데이터 로드"""
    import pyarrow.dataset as ds
    import pyarrow as pa

    products_dir = _get_local_data_dir() / "integrated_products_final"

    # 스키마 불일치(large_string vs string) 처리
    try:
        dataset = ds.dataset(products_dir, format="parquet", partitioning="hive")
        df = dataset.to_table().to_pandas()
    except pa.lib.ArrowTypeError:
        # 파티션별 개별 로드 후 병합
        import glob

        parquet_files = glob.glob(str(products_dir / "category=*" / "data.parquet"))
        dfs = []
        for f in parquet_files:
            part_df = pd.read_parquet(f)
            # 디렉토리명에서 카테고리 추출
            cat = f.split("category=")[1].split("/")[0]
            part_df["category"] = cat
            dfs.append(part_df)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if "category" in df.columns:
        import unicodedata

        df["category"] = df["category"].astype(str).str.replace("_", "/", regex=False)
        # macOS NFD → NFC 정규화
        df["category"] = df["category"].apply(
            lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x
        )

    return df


@st.cache_data(show_spinner=False)
def load_reviews_local(product_id: str) -> pd.DataFrame:
    """로컬 Parquet에서 특정 상품의 리뷰 데이터 로드"""
    reviews_dir = _get_local_data_dir() / "partitioned_reviews"
    if not reviews_dir.exists():
        return pd.DataFrame()

    import pyarrow.dataset as ds

    dataset = ds.dataset(reviews_dir, format="parquet", partitioning="hive")
    table = dataset.to_table(filter=ds.field("product_id") == product_id)
    df = table.to_pandas()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False)

    return df


def load_top_reviews_local(
    product_id: str, product_info: pd.Series, limit: int = 5, sentiment_type: str = None
) -> pd.DataFrame:
    """로컬 Parquet에서 대표 리뷰 텍스트 로드"""
    import json

    if sentiment_type == "negative":
        ids = product_info.get("negative_representative_ids", [])
    else:
        ids = product_info.get("positive_representative_ids", [])

    if isinstance(ids, str):
        ids = ids.strip()
        if not ids or ids in ("null", "NULL"):
            return pd.DataFrame()
        try:
            ids = json.loads(ids)
        except Exception:
            return pd.DataFrame()

    # numpy array → list 변환
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()

    if ids is None or not isinstance(ids, list) or not ids:
        return pd.DataFrame()

    review_ids = ids[:limit]

    all_reviews = load_reviews_local(product_id)
    if all_reviews.empty:
        return pd.DataFrame()

    review_ids_int = [int(rid) for rid in review_ids]
    result = all_reviews[all_reviews["id"].isin(review_ids_int)]

    cols = ["id", "full_text", "title", "content", "score", "sentiment_score", "date"]
    available_cols = [c for c in cols if c in result.columns]
    return result[available_cols].copy()


# =========================
# 통합 데이터 로딩 (로컬/AWS 자동 분기)
# =========================
def load_top_reviews_athena(
    product_id: str, product_info: pd.Series, limit: int = 5, sentiment_type: str = None
) -> pd.DataFrame:
    """
    대표 리뷰 텍스트 N개 로드 (데이터 소스에 따라 로컬/Athena 자동 분기)
    """
    # 로컬 모드면 로컬 함수 사용
    if _get_data_source() == "local":
        return load_top_reviews_local(product_id, product_info, limit, sentiment_type)

    # AWS 모드
    _, _, fetch_top_reviews_text = _import_athena_functions()
    import json

    # sentiment_type에 따라 적절한 ID 배열 선택
    if sentiment_type == "negative":
        ids = product_info.get("negative_representative_ids", [])
        print(f"[DEBUG] Negative IDs raw: {ids}, type: {type(ids)}")
    else:  # positive 또는 None
        ids = product_info.get("positive_representative_ids", [])
        print(f"[DEBUG] Positive IDs raw: {ids}, type: {type(ids)}")

    # 배열이 문자열로 저장되어 있을 경우 파싱
    if isinstance(ids, str):
        # 빈 문자열이나 null 체크
        ids = ids.strip()
        if not ids or ids == "null" or ids == "NULL":
            print(f"[DEBUG] Empty or null string")
            return pd.DataFrame()
        try:
            ids = json.loads(ids)
            print(f"[DEBUG] Parsed IDs: {ids}")
        except Exception as e:
            print(f"[DEBUG] JSON parse error: {e}")
            return pd.DataFrame()

    # None이거나 리스트가 아닌 경우
    if ids is None or not isinstance(ids, list):
        print(f"[DEBUG] IDs is None or not a list")
        return pd.DataFrame()

    # 빈 리스트 체크
    if not ids:
        print(f"[DEBUG] Empty list")
        return pd.DataFrame()

    # limit만큼만 가져오기
    review_ids = ids[:limit]
    print(f"[DEBUG] Final review_ids to fetch: {review_ids}")

    result = fetch_top_reviews_text(product_id, review_ids)
    print(f"[DEBUG] Fetched reviews count: {len(result)}")
    return result


DEFAULT_IMAGE_URL = "https://tr.rbxcdn.com/180DAY-981c49e917ba903009633ed32b3d0ef7/420/420/Hat/Webp/noFilter"


@st.cache_data(ttl=300, show_spinner=False)
def load_products_from_athena() -> pd.DataFrame:
    """Athena에서 전체 상품 데이터 로드"""
    fetch_all_products, _, _ = _import_athena_functions()
    return fetch_all_products()


@st.cache_data(ttl=300, show_spinner=False)
def load_reviews_athena(product_id: str) -> pd.DataFrame:
    """리뷰 데이터 로드 (데이터 소스에 따라 로컬/Athena 자동 분기)"""
    if _get_data_source() == "local":
        return load_reviews_local(product_id)
    _, fetch_reviews_by_product, _ = _import_athena_functions()
    return fetch_reviews_by_product(product_id)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """UI에서 사용할 컬럼들 정규화 및 매핑"""
    df = df.copy()

    SUB_ALIAS = {
        "알로에/수딩/에프터선": "알로에/수딩/애프터선",
        "립메이크업/포인트리무버": "리무버",
        "립/아이리무버": "리무버",
    }

    # 카테고리 정규화
    def normalize_sub_category(x: str) -> str:
        if not isinstance(x, str):
            return ""
        x = x.strip()
        x = x.replace("_", "/")
        x = re.sub(r"\s+", "", x)
        x = SUB_ALIAS.get(x, x)
        return x

    # main/middle/sub 매핑
    MAIN_TO_MIDDLES = {
        "메이크업": [
            "립메이크업",
            "바디메이크업",
            "베이스메이크업",
            "아이메이크업",
            "치크메이크업",
        ],
        "선케어/태닝": ["선케어/태닝"],
        "스킨케어": ["스킨케어"],
        "클렌징/필링": ["클렌징/필링"],
    }

    MIDDLE_TO_SUBS = {
        "립메이크업": ["립스틱", "립케어", "틴트/립글로스"],
        "바디메이크업": ["바디메이크업", "헤나/타투스티커"],
        "베이스메이크업": [
            "메이크업픽서",
            "베이스/프라이머",
            "컨실러",
            "쿠션/팩트",
            "파우더/파우더팩트",
            "파운데이션",
            "BB/CC크림/톤업크림",
        ],
        "아이메이크업": [
            "마스카라",
            "아이라이너",
            "아이메이크업/포인트리무버",
            "아이브로우",
            "아이섀도",
            "아이팔레트",
        ],
        "치크메이크업": ["블러셔", "하이라이터"],
        "선케어/태닝": [
            "선스틱",
            "선스프레이",
            "선케어/선크림/선로션",
            "선쿠션/선팩트",
            "알로에/수딩/애프터선",
            "자외선차단패치",
            "태닝오일",
        ],
        "스킨케어": [
            "기초세트",
            "로션",
            "미스트",
            "스킨",
            "에센스/세럼/앰플",
            "오일",
            "멀티밤/스틱",
            "아이/넥크림",
            "올인원",
            "페이셜크림",
        ],
        "클렌징/필링": [
            "리무버",
            "스크럽/필링",
            "클렌징밤/크림/로션",
            "클렌징비누",
            "클렌징세트",
            "클렌징오일",
            "클렌징워터",
            "클렌징젤/파우더",
            "클렌징티슈/시트",
            "클렌징폼",
        ],
    }

    def category_mapping():
        sub_to_middle = {}
        for middle, subs in MIDDLE_TO_SUBS.items():
            for sub in subs:
                sub_norm = normalize_sub_category(sub)
                sub_to_middle[sub_norm] = middle

        middle_to_main = {}
        for main, middles in MAIN_TO_MIDDLES.items():
            for middle in middles:
                middle_to_main[middle] = main

        sub_to_main = {}
        for sub_norm, middle in sub_to_middle.items():
            sub_to_main[sub_norm] = middle_to_main.get(middle, "")

        return sub_to_main, sub_to_middle

    SUB_TO_MAIN, SUB_TO_MIDDLE = category_mapping()

    if "category" in df.columns:
        df["sub_category"] = df["category"].apply(normalize_sub_category)
    else:
        df["sub_category"] = ""

    df["middle_category"] = df["sub_category"].map(SUB_TO_MIDDLE).fillna("")
    df["main_category"] = df["sub_category"].map(SUB_TO_MAIN).fillna("")

    # 매핑 실패 -> 기타 카테고리
    unknown_mask = df["main_category"].eq("") & df["sub_category"].ne("")

    df.loc[unknown_mask, "main_category"] = "기타"
    df.loc[unknown_mask, "middle_category"] = "기타"

    # category가 비어있는 경우
    empty_mask = df["sub_category"].eq("")
    df.loc[empty_mask, "main_category"] = "기타"
    df.loc[empty_mask, "middle_category"] = "기타"
    df.loc[empty_mask, "sub_category"] = "기타"

    # 카테고리 경로
    df["category_path_norm"] = np.where(
        df["main_category"].astype(str).str.strip()
        == df["middle_category"].astype(str).str.strip(),
        df["main_category"].astype(str).str.strip()
        + " > "
        + df["sub_category"].astype(str).str.strip(),
        df["main_category"].astype(str).str.strip()
        + " > "
        + df["middle_category"].astype(str).str.strip()
        + " > "
        + df["sub_category"].astype(str).str.strip(),
    )

    mask_unknown = df["main_category"].eq("")
    df.loc[mask_unknown, "category_path_norm"] = ""

    # 평점 컬럼
    if "score" not in df.columns and "avg_rating_with_text" in df.columns:
        df["score"] = df["avg_rating_with_text"]

    # 뱃지 초기화
    if "badge" not in df.columns:
        df["badge"] = ""
    df["badge"] = df["badge"].fillna("").astype(str)

    # 뱃지 계산
    if "total_reviews" in df.columns:
        tr = pd.to_numeric(df["total_reviews"], errors="coerce").fillna(0)
        need = df["badge"].eq("")
        best = need & (tr >= 200) & (df["score"] >= 4.9)
        reco = need & (tr >= 200) & (df["score"] >= 4.8) & (~best)
        df.loc[best, "badge"] = "BEST"
        df.loc[reco, "badge"] = "추천"

    # 이미지 URL
    if "image_url" not in df.columns:
        if "img_url" in df.columns:
            # img_url이 있으면 image_url로 매핑
            df["image_url"] = df["img_url"].fillna(DEFAULT_IMAGE_URL)
            # 빈 문자열도 기본 이미지로 대체
            mask = df["image_url"].astype(str).str.strip() == ""
            df.loc[mask, "image_url"] = DEFAULT_IMAGE_URL
        else:
            df["image_url"] = DEFAULT_IMAGE_URL
    else:
        # image_url이 있어도 빈 값 처리
        df["image_url"] = df["image_url"].fillna(DEFAULT_IMAGE_URL)
        mask = df["image_url"].astype(str).str.strip() == ""
        df.loc[mask, "image_url"] = DEFAULT_IMAGE_URL

    # 대표 리뷰 ID
    if "representative_review_id_roberta" not in df.columns:
        if "representative_review_id_roberta_sentiment" in df.columns:
            df["representative_review_id_roberta"] = df[
                "representative_review_id_roberta_sentiment"
            ]
        elif "representative_review_id_roberta_semantic" in df.columns:
            df["representative_review_id_roberta"] = df[
                "representative_review_id_roberta_semantic"
            ]
        else:
            df["representative_review_id_roberta"] = np.nan

    # 제품 URL
    if "product_url" not in df.columns:
        df["product_url"] = ""

    # 키워드 문자열
    if "top_keywords_str" not in df.columns:
        if "top_keywords" in df.columns:
            df["top_keywords_str"] = df["top_keywords"].apply(
                lambda x: (
                    ", ".join(map(str, x))
                    if isinstance(x, (list, np.ndarray))
                    else re.sub(r"[\[\]']", "", str(x))
                )
            )
        else:
            df["top_keywords_str"] = ""

    return df


def prepare_dataframe() -> pd.DataFrame:
    """메인 DataFrame 준비 (데이터 소스에 따라 로컬/Athena 자동 분기)"""
    if _get_data_source() == "local":
        product_df = load_products_local()
    else:
        product_df = load_products_from_athena()

    try:
        df = make_df(product_df)
    except Exception:
        df = product_df.copy()

    df = normalize_columns(df)
    return df


def get_options(df: pd.DataFrame) -> tuple:
    """사이드바/검색용 옵션 목록 반환"""
    skin_options = (
        df["skin_type"].dropna().unique().tolist() if "skin_type" in df.columns else []
    )
    product_options = (
        df["product_name"].dropna().unique().tolist()
        if "product_name" in df.columns
        else []
    )
    return skin_options, product_options


def apply_filters(
    df: pd.DataFrame,
    selected_sub_cat: list,
    selected_skin: list,
    min_rating: float,
    max_rating: float,
    min_price: int,
    max_price: int,
    search_text: str = "",
    search_type: str = "키워드",
) -> pd.DataFrame:
    """필터 조건 적용"""
    filtered_df = df.copy()

    # 카테고리 필터
    if selected_sub_cat:
        filtered_df = filtered_df[filtered_df["sub_category"].isin(selected_sub_cat)]

    # 피부 타입 필터
    if selected_skin:
        filtered_df = filtered_df[filtered_df["skin_type"].isin(selected_skin)]

    # 평점 필터
    filtered_df = filtered_df[
        (filtered_df["score"] >= min_rating) & (filtered_df["score"] <= max_rating)
    ]

    # 가격 필터
    filtered_df = filtered_df[
        (filtered_df["price"] >= min_price) & (filtered_df["price"] <= max_price)
    ]

    # 컬럼 보정
    if (
        "score" not in filtered_df.columns
        and "avg_rating_with_text" in filtered_df.columns
    ):
        filtered_df["score"] = filtered_df["avg_rating_with_text"]

    if "image_url" not in filtered_df.columns:
        filtered_df["image_url"] = None
    if "badge" not in filtered_df.columns:
        filtered_df["badge"] = ""
    if "category_path_norm" not in filtered_df.columns:
        filtered_df["category_path_norm"] = (
            filtered_df["category"] if "category" in filtered_df.columns else ""
        )

    # 검색 타입에 따른 필터링
    if search_text:
        s = search_text.strip()

        keyword_series = filtered_df.get(
            "top_keywords",
            pd.Series([""] * len(filtered_df), index=filtered_df.index),
        ).astype(str)
        product_name_series = filtered_df["product_name"].astype(str)
        brand_series = filtered_df["brand"].astype(str)

        if search_type == "상품명":
            # 상품명에만 검색
            mask = product_name_series.str.contains(
                s, case=False, na=False, regex=False
            )

        elif search_type == "문맥":
            # 문맥 검색 (추후 구현 가능)
            mask = (
                product_name_series.str.contains(s, case=False, na=False, regex=False)
                | brand_series.str.contains(s, case=False, na=False, regex=False)
                | keyword_series.str.contains(s, case=False, na=False, regex=False)
            )

        else:  # 키워드 검색 (기본)
            # 키워드, 제품명, 브랜드 모두 검색
            mask = (
                keyword_series.str.contains(s, case=False, na=False, regex=False)
                | brand_series.str.contains(s, case=False, na=False, regex=False)
                | product_name_series.str.contains(s, case=False, na=False, regex=False)
            )

        filtered_df = filtered_df[mask]

    return filtered_df


def sort_products(df: pd.DataFrame, sort_option: str) -> pd.DataFrame:
    """정렬 옵션 적용"""
    df = df.copy()

    # 유사도/추천점수 기본값
    if "reco_score" not in df.columns:
        df["reco_score"] = 0.0
    if "similarity" not in df.columns:
        df["similarity"] = 0.0

    # 뱃지 순서
    badge_order = {"BEST": 0, "추천": 1, "": 2}
    df["badge_rank"] = df.get("badge", "").map(badge_order).fillna(2)

    if sort_option == "추천순":
        df = df.sort_values(
            by=["badge_rank", "score", "total_reviews"],
            ascending=[True, False, False],
        )
    elif sort_option == "평점 높은 순":
        df = df.sort_values(
            by=["score", "total_reviews"],
            ascending=[False, False],
        )
    elif sort_option == "리뷰 많은 순":
        df = df.sort_values(
            by=["total_reviews", "score"],
            ascending=[False, False],
        )
    elif sort_option == "가격 낮은 순":
        df = df.sort_values(
            by=["price", "score"],
            ascending=[True, False],
        )
    elif sort_option == "가격 높은 순":
        df = df.sort_values(
            by=["price", "score"],
            ascending=[False, False],
        )
    else:
        # 기본 정렬
        df = df.sort_values(
            by=["badge_rank", "score", "total_reviews"],
            ascending=[True, False, False],
        )

    return df
