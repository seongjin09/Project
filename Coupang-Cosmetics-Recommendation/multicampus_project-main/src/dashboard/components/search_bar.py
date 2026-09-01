"""
검색창 컴포넌트
"""

import streamlit as st


def render_search_bar(
    product_options: list,
    on_clear_callback,
    filtered_products: list = None,
    context_search_results: list = None,
):
    """
    검색창 렌더링

    Args:
        product_options: 전체 제품명 옵션 목록
        on_clear_callback: 초기화 버튼 클릭 시 콜백
        filtered_products: 왼쪽 검색으로 필터링된 제품 목록 (없으면 전체 목록 사용)
        context_search_results: 문맥 검색 결과 제품명 목록 (유사도 순)

    Returns:
        selected_product: 선택된 제품명
    """
    with st.container(border=True):
        col_type, col_text, col_sel, col_clear = st.columns(
            [1, 4, 4, 1], vertical_alignment="bottom"
        )

        with col_type:
            st.selectbox(
                "검색 타입",
                options=["상품명", "키워드", "문맥"],
                key="search_type",
                label_visibility="collapsed",
            )

        with col_text:
            search_type = st.session_state.get("search_type", "상품명")

            # 문맥 검색일 때만 placeholder 변경
            if search_type == "문맥":
                placeholder_text = "예: 지성 피부에 좋은 가벼운 로션"
            else:
                placeholder_text = "예: 수분, 촉촉, 진정"

            st.text_input(
                "검색어 입력",
                placeholder=placeholder_text,
                key="search_keyword",
            )

        with col_sel:
            # 검색 타입과 키워드에 따라 제품 목록 필터링
            search_type = st.session_state.get("search_type", "상품명")
            search_keyword = st.session_state.get("search_keyword", "").strip()

            if search_type == "상품명" and search_keyword:
                # 상품명에 검색 키워드가 포함된 제품만 필터링
                display_options = [
                    p for p in product_options if search_keyword.lower() in p.lower()
                ]
            elif search_type == "문맥" and context_search_results:
                # 문맥 검색 결과 사용 (유사도 순)
                display_options = context_search_results
            elif filtered_products is not None:
                # 외부에서 필터링된 제품 목록 사용
                display_options = filtered_products
            else:
                # 전체 목록 사용
                display_options = product_options

            # 문맥 검색일 때는 라벨 변경
            selectbox_label = f"일치 상품 ({len(display_options)}개)"

            st.selectbox(
                selectbox_label,
                options=[""] + display_options,
                key="product_search",
            )
            selected_product = st.session_state.get("product_search", "")

        with col_clear:
            st.button(
                "✕",
                help="검색 초기화",
                on_click=on_clear_callback,
            )

    return selected_product


def get_search_text() -> str:
    """현재 검색어 반환"""
    if st.session_state.get("product_search"):
        return st.session_state.product_search
    return st.session_state.get("search_keyword", "").strip()


def get_search_type() -> str:
    """현재 검색 타입 반환"""
    return st.session_state.get("search_type", "상품명")


def get_search_info() -> tuple[str, str]:
    """검색 타입과 검색어를 함께 반환"""
    search_type = get_search_type()
    search_text = get_search_text()
    return search_type, search_text


def is_initial_state(selected_sub_cat: list, selected_skin: list) -> bool:
    """초기 상태인지 확인"""
    search_text = get_search_text()
    return not search_text and not selected_sub_cat and not selected_skin
