"""
제품 상세 정보 컴포넌트
"""

import streamlit as st
import pandas as pd
from utils.data_utils import DEFAULT_IMAGE_URL


def render_product_info(product_info: pd.Series):
    """
    선택한 제품의 상세 정보 렌더링

    Args:
        product_info: 제품 정보 Series
    """
    st.subheader("선택한 제품 정보")

    col_img, col_details = st.columns([1, 4])

    with col_img:
        image_url = product_info.get("image_url", DEFAULT_IMAGE_URL)
        if not image_url or (isinstance(image_url, float) and pd.isna(image_url)):
            image_url = DEFAULT_IMAGE_URL
        st.image(image_url, width=280)

    with col_details:
        # 이미지 높이에 맞춰 수직 정렬
        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("제품명", product_info.get("product_name", ""))
        col2.metric(
            "브랜드",
            (
                "-"
                if pd.isna(product_info.get("brand"))
                else str(product_info.get("brand"))
            ),
        )
        col3.metric("피부 타입", product_info.get("skin_type", ""))

        st.markdown("<div style='height:1px;'></div>", unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)
        col4.metric("가격", f"₩{int(product_info.get('price', 0) or 0):,}")
        col5.metric("리뷰 수", f"{int(product_info.get('total_reviews', 0) or 0):,}")
        col6.metric("카테고리", product_info.get("sub_category", ""))

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        if product_info.get("product_url"):
            st.link_button("상품 페이지", str(product_info["product_url"]))
