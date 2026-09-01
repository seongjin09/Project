"""
유사 추천 상품 로직
"""

import streamlit as st
import pandas as pd

from services.recommend_similar_products import recommend_similar_products


def get_recommendations(
    df: pd.DataFrame,
    selected_product: str = None,
    selected_categories: list[str] | None = None,
    query_text: str = None,
    vectorizer=None,
) -> pd.DataFrame:
    """
    선택한 상품과 유사한 추천 상품 조회 또는 문맥 검색

    Args:
        df: 전체 상품 DataFrame
        selected_product: 선택한 상품명 (상품명 검색)
        selected_categories: 선택한 카테고리 목록
        query_text: 검색 문장 (문맥 검색)
        vectorizer: BERTVectorizer 인스턴스 (문맥 검색 시 필요)

    Returns:
        추천 상품 DataFrame (최대 6개)
    """
    # 스키마 고정용 빈 DF (점수는 있지만 값은 없음)
    reco_df_view = df.iloc[0:0].copy()
    reco_df_view["reco_score"] = pd.Series(dtype="float")
    reco_df_view["similarity"] = pd.Series(dtype="float")
    # reco_df_view = pd.DataFrame()

    # 문맥 검색 모드
    if query_text:
        cache_key = (
            "query",
            query_text,
            tuple(selected_categories) if selected_categories else None,
        )

        # 캐시 확인
        if st.session_state.get("reco_cache_key") != cache_key:
            reco_results = recommend_similar_products(
                query_text=query_text,
                categories=selected_categories,
                top_n=5,  # 카테고리별 상위 5개
                vectorizer=vectorizer,
            )

            if isinstance(reco_results, list):
                reco_list = reco_results
            else:
                reco_list = []
                for _, items in reco_results.items():
                    reco_list.extend(items)

            st.session_state["reco_cache"] = reco_list
            st.session_state["reco_cache_key"] = cache_key
        else:
            reco_list = st.session_state.get("reco_cache", [])

        if reco_list:
            tmp_reco_df = pd.DataFrame(reco_list).rename(
                columns={
                    "recommend_score": "reco_score",
                    "cosine_similarity": "similarity",
                }
            )

            merged_df = df.merge(
                tmp_reco_df[["product_id", "reco_score", "similarity"]],
                on="product_id",
                how="inner",
            )
            # merged_df["reco_score"] = merged_df["reco_score"].fillna(0)
            # merged_df["similarity"] = merged_df["similarity"].fillna(0)

            if selected_categories:
                merged_df = merged_df[
                    merged_df["sub_category"].isin(selected_categories)
                ]

            reco_df_view = (
                merged_df.query("reco_score > 0")
                .groupby("sub_category", group_keys=False)
                .head(6)
            )
        if not reco_list:
            return reco_df_view

        return reco_df_view

    # 상품명 검색 모드
    target_product = df[df["product_name"] == selected_product]
    if target_product.empty:
        return reco_df_view

    target_product_id = target_product.iloc[0]["product_id"]

    cache_key = (
        "product",
        target_product_id,
        tuple(selected_categories) if selected_categories else None,
    )

    # 캐시 확인
    if st.session_state.get("reco_cache_key") != cache_key:
        reco_results = recommend_similar_products(
            product_id=target_product_id,
            categories=selected_categories,
            top_n=100,
        )

        if isinstance(reco_results, list):
            reco_list = reco_results
        else:
            reco_list = []
            for _, items in reco_results.items():
                reco_list.extend(items)

        st.session_state["reco_cache"] = reco_list
        st.session_state["reco_cache_key"] = cache_key
    else:
        reco_list = st.session_state.get("reco_cache", [])

    if reco_list:
        tmp_reco_df = pd.DataFrame(reco_list).rename(
            columns={
                "recommend_score": "reco_score",
                "cosine_similarity": "similarity",
            }
        )

        merged_df = df.merge(
            tmp_reco_df[["product_id", "reco_score", "similarity"]],
            on="product_id",
            how="left",
        )
        merged_df["reco_score"] = merged_df["reco_score"].fillna(0)
        merged_df["similarity"] = merged_df["similarity"].fillna(0)

        merged_df = merged_df[merged_df["product_id"] != target_product_id]

        if selected_categories:
            merged_df = merged_df[merged_df["sub_category"].isin(selected_categories)]
        reco_df_view = (
            merged_df.query("reco_score > 0")
            # .sort_values(by=["reco_score", "similarity"], ascending=[False, False])
            .groupby("sub_category", group_keys=False).head(6)
        )

    return reco_df_view
