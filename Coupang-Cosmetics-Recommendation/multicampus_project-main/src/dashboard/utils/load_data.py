import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.dataset as ds
import streamlit as st
import re
import ast
import unicodedata


def normalize_text(s):
    if not isinstance(s, str):
        return ""
    # NFKC 정규화 + 공백 제거 + 슬래시 주변 공백 제거
    normalized = unicodedata.normalize("NFKC", s).strip()
    # 슬래시 앞뒤 공백 제거 (예: "헤어 / 바디" → "헤어/바디")
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    return normalized


@st.cache_data
def load_raw_df(parquet_root: Path) -> pd.DataFrame:
    dataset = ds.dataset(parquet_root, format="parquet", partitioning="hive")
    table = dataset.to_table()
    df = table.to_pandas()

    if "category" in df.columns:
        df["category"] = df["category"].str.replace("_", "/", regex=False)

    return df


# 대표 리뷰 로드 & 가독성
@st.cache_data
def load_reviews(product_id: str, review_id: str, category: str, base_dir: Path) -> str:
    category = category.replace("/", "_")
    review_path = base_dir / f"category={category}" / "data.parquet"

    if not review_path.exists():
        return ""
    try:
        df = pd.read_parquet(review_path)
    except Exception:
        return ""

    filtered = df[(df["product_id"] == product_id) & (df["id"] == review_id)]

    if filtered.empty:
        return ""

    text = filtered["full_text"].values[0]

    if isinstance(text, (list, np.ndarray)):
        return text[0] if text else ""
    elif not isinstance(text, str):
        return ""

    def split_text(txt):
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        return "\n".join(sentences)

    pre_text = split_text(text.strip())

    max_len = 600
    if len(pre_text) > max_len:
        return pre_text[:max_len].rstrip() + "···"
    else:
        return pre_text


# 리뷰 작성일, 평점 데이터 로드
@st.cache_data
def load_date_score(product_id: str, category: str, base_dir: Path) -> str:
    category = category.replace("/", "_")
    review_path = base_dir / f"category={category}" / "data.parquet"

    if not review_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(review_path, columns=["product_id", "date", "score"])
    df = df[df["product_id"] == product_id]
    df["date"] = pd.to_datetime(df["date"])

    return df


# 기간별 평점, 이동평균 계산
def rating_trend(
    review_df: pd.DataFrame, freq: str = "W", ma_window: int = 4
) -> pd.DataFrame:
    if review_df.empty:
        return review_df

    df = review_df.copy()
    df = df.set_index("date").sort_index()

    trend_df = (
        df.resample(freq)
        .agg(avg_score=("score", "mean"), review_count=("score", "count"))
        .reset_index()
    )
    trend_df["avg_score"] = trend_df["avg_score"].round(2)
    trend_df["ma"] = (
        trend_df["avg_score"].rolling(window=ma_window, min_periods=1).mean().round(2)
    )

    return trend_df


def make_df(df: pd.DataFrame) -> pd.DataFrame:
    rating_df = df.groupby("product_id", as_index=False).agg(
        {
            "rating_1": "sum",
            "rating_2": "sum",
            "rating_3": "sum",
            "rating_4": "sum",
            "rating_5": "sum",
            "total_reviews": "sum",
        }
    )

    # 상품 평점
    rating_df["score"] = (
        rating_df["rating_1"] * 1
        + rating_df["rating_2"] * 2
        + rating_df["rating_3"] * 3
        + rating_df["rating_4"] * 4
        + rating_df["rating_5"] * 5
    ) / rating_df["total_reviews"]

    rating_df["score"] = rating_df["score"].round(2)
    rating_df["score"] = rating_df["score"].fillna(0)

    image_url = f"https://tr.rbxcdn.com/180DAY-981c49e917ba903009633ed32b3d0ef7/420/420/Hat/Webp/noFilter"

    # 추천 뱃지
    def calc_badge(score, total_reviews):
        if total_reviews >= 200:
            if score >= 4.9:
                return "BEST"
            elif score >= 4.7:
                return "추천"
        return ""

    rating_df["badge"] = rating_df.apply(
        lambda x: calc_badge(x["score"], x["total_reviews"]), axis=1
    )

    # 카테고리 정규화
    main_cats = [
        "스킨케어",
        "클렌징/필링",
        "선케어/태닝",
        "메이크업",
        "헤어/바디",
        "바디",
    ]

    def norm_cat(path):
        if not isinstance(path, str):
            return ""

        # 불필요한 prefix 제거 (정확한 매칭을 위해 정규화 후 비교)
        remove_prefixes = ["쿠팡 홈", "쿠팡홈", "R.LUX", "뷰티", "전체", "화장품"]
        parts = [p.strip() for p in path.split(">")]

        # prefix 제거 (대소문자 구분 없이)
        filtered_parts = []
        for p in parts:
            p_normalized = normalize_text(p)
            if p_normalized not in remove_prefixes:
                filtered_parts.append(p_normalized)

        # main_cats에 있는 카테고리 찾기
        for i, part in enumerate(filtered_parts):
            part_normalized = normalize_text(part)
            for main in main_cats:
                main_normalized = normalize_text(main)
                # "헤어/바디"나 "바디" 등을 찾으면 그 위치부터 반환
                if part_normalized == main_normalized:
                    return " > ".join(filtered_parts[i:])

        # main_cats에 없으면 빈 문자열 (사이드바에 표시 안됨)
        return ""

    def split_category(path: str):
        if not isinstance(path, str):
            return "", "", ""

        parts = [p.strip() for p in path.split(">")]

        main = parts[0] if len(parts) >= 1 else ""
        middle = parts[1] if len(parts) >= 2 else ""

        # depth가 3 이상일 때
        if len(parts) >= 3:
            # 3단계면 그대로, 4단계 이상이면 중간 카테고리 합치기
            if len(parts) == 3:
                sub = parts[2]
            else:
                # 4단계 이상: middle을 2번째부터 마지막 전까지 합치고, sub는 마지막
                middle = " > ".join(parts[1:-1])
                sub = parts[-1]
        else:
            sub = parts[-1] if parts else ""

        return main, middle, sub

    # unhashable값들 문자열 변환
    def _make_hashable_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if (
                df[col]
                .apply(lambda x: isinstance(x, (list, np.ndarray, dict, set)))
                .any()
            ):
                df[col] = df[col].apply(
                    lambda x: (
                        ", ".join(map(str, x))
                        if isinstance(x, (list, np.ndarray))
                        else (str(x) if pd.notna(x) else "")
                    )
                )
        return df

    df = df.copy()
    df["category_path_norm"] = df["category_path"].apply(norm_cat)
    df[["main_category", "middle_category", "sub_category"]] = (
        df["category_path_norm"].apply(split_category).apply(pd.Series)
    )
    df["image_url"] = image_url
    df["top_keywords"] = df["top_keywords"].apply(
        lambda x: (
            ", ".join(x)
            if isinstance(x, (list, np.ndarray))
            else (x if isinstance(x, str) else "")
        )
    )

    product_df = (
        df[
            [
                "product_id",
                "product_name",
                "brand",
                "price",
                "image_url",
                "product_url",
                "representative_review_id_roberta",
                "total_reviews",
                "top_keywords",
                "category",
                "category_path_norm",
                "main_category",
                "middle_category",
                "sub_category",
                "skin_type",
            ]
        ]
        .drop_duplicates("product_id")
        .copy()
    )

    fin_df = product_df.merge(
        rating_df[["product_id", "score", "badge"]], on="product_id", how="left"
    )

    fin_df = _make_hashable_df(fin_df)

    return fin_df
