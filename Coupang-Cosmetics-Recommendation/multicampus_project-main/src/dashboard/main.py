"""
화장품 추천 대시보드 - 메인 앱
"""

import streamlit as st
import pandas as pd
import sys
import os

from utils import css
from utils import scroll
from layouts.sidebar import sidebar

# 컴포넌트 임포트
from components.search_bar import (
    render_search_bar,
    get_search_text,
    get_search_type,
    get_search_info,
    is_initial_state,
)
from components.product_info import render_product_info
from components.product_analysis import (
    render_top_keywords,
    render_ai_review_summary,
    load_product_analysis_async,
    render_representative_review,
    render_rating_trend,
)
from components.product_cards import (
    render_popular_products,
    render_search_results_grid,
    render_recommendations_grid,
)
from components.recommendations import get_recommendations
from components.pagination import (
    calculate_pagination,
    init_page_state,
    check_filter_change,
    get_page_slice,
    render_pagination,
)

# 유틸 임포트
from utils.data_utils import (
    prepare_dataframe,
    get_options,
    apply_filters,
    sort_products,
)

sys.path.append(os.path.dirname(__file__))

# =========================
# ✅ 데이터 소스 설정
# =========================
# "local" : 로컬 Parquet 파일 사용 (AWS 연결 불필요)
# "aws"   : AWS Athena 사용 (secrets.toml에 AWS 자격증명 필요)
DATA_SOURCE = "local"


# =========================
# ✅ 세션 상태 초기화
# =========================
def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        "product_search": "",
        "search_keyword": "",
        "page": 1,
        "reco_cache": {},
        "reco_target_product_id": None,
        "_skip_scroll_apply_once": False,
        "last_loaded_product_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================
# ✅ 스크롤 관련
# =========================
def skip_scroll_apply_once():
    """그래프 UI 조작 시 스크롤 스킵"""
    st.session_state["_skip_scroll_apply_once"] = True


def safe_scroll_to_top():
    """안전하게 스크롤 상단 이동"""
    scroll.request_scroll_to_top()


def apply_scroll():
    """스크롤 적용"""
    if not st.session_state.get("_skip_scroll_apply_once", False):
        scroll.apply_scroll_to_top_if_requested()
    else:
        st.session_state["_skip_scroll_apply_once"] = False


# =========================
# ✅ 콜백 함수들
# =========================
def clear_selected_product():
    """제품 선택 해제"""
    st.session_state["product_search"] = ""
    st.session_state["search_keyword"] = ""
    st.session_state["last_loaded_product_id"] = None
    safe_scroll_to_top()


def select_product_from_reco(product_name: str):
    """추천 상품 클릭 시 선택"""
    st.session_state["product_search"] = product_name
    st.session_state["last_loaded_product_id"] = (
        None  # 새 상품이므로 비동기 재로딩 트리거
    )
    safe_scroll_to_top()


def render_recommendation_section(df: pd.DataFrame, selected_product: str):
    """추천 상품 섹션 렌더링"""
    st.markdown("<div style='height:64px;'></div>", unsafe_allow_html=True)
    st.subheader("이 상품과 유사한 추천 상품")

    col_1, col_2, col_3 = st.columns([5, 2, 3])

    with col_2:
        sort_option = st.selectbox(
            "정렬 옵션",
            options=[
                "추천순",
                "평점 높은 순",
                "리뷰 많은 순",
                "가격 낮은 순",
                "가격 높은 순",
            ],
            index=0,
            key="reco_sort_option",
            label_visibility="collapsed",
        )

    with col_3:
        all_categories = sorted(
            c for c in df["sub_category"].unique() if isinstance(c, str) and c.strip()
        )

        # 현재 선택된 상품 카테고리
        current_category = (
            df.loc[df["product_name"] == selected_product, "sub_category"].iloc[0]
            if selected_product in df["product_name"].values
            else None
        )

        # 디폴트
        default_index = (
            all_categories.index(current_category)
            if current_category in all_categories
            else 0
        )

        def on_category_change():
            """추천 카테고리 변경 시 캐시 무효화 및 재검색 트리거"""
            st.session_state["reco_cache_key"] = None
            st.session_state["reco_cache"] = []

        selected_categories = st.selectbox(
            "카테고리 선택",
            all_categories,
            index=default_index,
            key="reco_category_select",
            label_visibility="collapsed",
            on_change=on_category_change,
        )

    # 추천 상품 조회
    product_rows = df[df["product_name"] == selected_product]
    if not product_rows.empty:
        target_product_id = product_rows.iloc[0]["product_id"]

        # 캐시 키 확인
        cache_key = (
            "product",
            target_product_id,
            tuple([selected_categories]) if selected_categories else None,
        )

        # 캐시가 없고, 현재 제품과 다르면 새로 로드
        if st.session_state.get("reco_cache_key") != cache_key:
            # 비동기 작업 자체가 아직 완료되지 않은 경우
            if st.session_state.get("reco_target_product_id") != target_product_id:
                st.info("유사한 상품을 찾고 있습니다...")
                return

            # 카테고리가 변경되어 재검색 필요
            with st.spinner("선택한 카테고리의 유사 상품을 검색 중입니다..."):
                reco_df_view = get_recommendations(
                    df, selected_product, [selected_categories]
                )
        else:
            # 캐시 사용
            reco_df_view = get_recommendations(
                df, selected_product, [selected_categories]
            )
    else:
        st.warning("선택한 제품 정보를 찾을 수 없습니다.")
        return

    # reco_score / similarity 컬럼 방어적 보정
    if "reco_score" not in reco_df_view.columns:
        reco_df_view["reco_score"] = 0.0

    if "similarity" not in reco_df_view.columns:
        reco_df_view["similarity"] = 0.0

    if sort_option == "추천순":
        reco_df_view = reco_df_view.sort_values(
            by=["reco_score", "similarity"],
            ascending=[False, False],
        )
    else:
        reco_df_view = sort_products(reco_df_view, sort_option)

    # Fragment 안에서 그리드 렌더링 (카테고리/정렬 변경 시 함께 재렌더)
    render_recommendations_grid(reco_df_view, select_product_from_reco)


# =========================
# ✅ 메인 앱
# =========================
def main():
    # 초기화
    st.set_page_config(
        page_title="화장품 추천 대시보드",
        page_icon="",
        layout="wide",
    )
    init_session_state()

    # 데이터 소스 설정을 session_state에 저장 (다른 모듈에서 참조)
    st.session_state["data_source"] = DATA_SOURCE

    apply_scroll()

    # 데이터 로드
    df = prepare_dataframe()
    _, product_options = get_options(df)

    # 사이드바
    (
        selected_sub_cat,
        selected_skin,
        min_rating,
        max_rating,
        min_price,
        max_price,
    ) = sidebar(df)

    st.markdown(
        """
        <style>
        .info-icon {
            cursor: help;
            color: #888;
            font-size: 18px;
        }
        </style>
        <span class="info-icon" title="다크 모드에서는 일부 UI가 정상적으로 표시되지 않을 수 있습니다. 원할한 이용을 위해 라이트 모드 사용을 권장합니다.">
        ⓘ
        </span>
        """,
        unsafe_allow_html=True,
    )

    # 메인 타이틀
    st.title("화장품 추천 대시보드")
    st.markdown("---")

    # =========================
    # 문맥 검색 사전 처리
    # =========================
    context_search_results = None
    context_search_df = None  # 문맥 검색 결과 DataFrame
    search_type_pre = st.session_state.get("search_type", "상품명")
    search_keyword_pre = st.session_state.get("search_keyword", "").strip()

    # 문맥 검색일 때 미리 검색 수행
    if search_type_pre == "문맥" and search_keyword_pre:
        try:
            # 환경변수 읽기 헬퍼 (Streamlit Cloud와 로컬 모두 지원)
            def get_config(key: str, default: str = "") -> str:
                """Streamlit secrets 또는 환경변수에서 값 읽기"""
                try:
                    # Streamlit Cloud secrets 우선
                    return st.secrets.get(key, os.getenv(key, default))
                except:
                    # secrets 없으면 환경변수
                    return os.getenv(key, default)

            # 세션에 vectorizer가 없거나 None이면 (재)로드
            if not st.session_state.get("vectorizer"):
                with st.spinner("AI 모델 로딩 중... (최초 1회)"):
                    from services.hf_api_vectorizer import HuggingFaceAPIVectorizer

                    hf_model_id = get_config(
                        "HF_MODEL_ID", "fullfish/multicampus_semantic"
                    )
                    hf_token = get_config("HF_TOKEN")

                    try:
                        st.session_state.vectorizer = HuggingFaceAPIVectorizer(
                            model_id=hf_model_id,
                            api_token=hf_token if hf_token else None,
                        )
                    except Exception as e:
                        st.error(f"⚠️ 모델 로딩 실패: {e}")
                        st.session_state.pop("vectorizer", None)

            # vectorizer가 로드되지 않았으면 문맥 검색 건너뛰기
            if not st.session_state.get("vectorizer"):
                st.warning(
                    "문맥 검색을 사용할 수 없습니다. 다른 검색 타입을 사용해주세요."
                )
            else:
                # 검색어에서 피부 타입 키워드 추출
                skin_type_keywords = ["건성", "지성", "복합성", "민감성", "여드름성"]
                detected_skin_types = [
                    skin for skin in skin_type_keywords if skin in search_keyword_pre
                ]

                # 검색어에서 카테고리 키워드 추출
                all_categories = df["category"].dropna().unique().tolist()
                detected_categories = [
                    cat for cat in all_categories if cat in search_keyword_pre
                ]

                # 캐시 키에 피부 타입 + 카테고리 정보도 포함
                cache_key = (
                    "context_search",
                    search_keyword_pre,
                    tuple(detected_skin_types),
                    tuple(detected_categories),
                )
                if st.session_state.get("context_search_cache_key") != cache_key:
                    with st.spinner("문맥 검색 중..."):
                        from services.recommend_similar_products import (
                            recommend_similar_products,
                        )

                        # 피부 타입이 감지되면 해당 타입으로 필터링
                        # recommend_similar_products는 categories만 받으므로
                        # 전체 데이터를 피부 타입으로 미리 필터링
                        search_data = df
                        filter_messages = []

                        if detected_skin_types:
                            # 복합성 → 복합/혼합으로 매핑
                            skin_filter = []
                            for skin in detected_skin_types:
                                if skin == "복합성":
                                    # 복합/혼합으로 시작하는 모든 피부 타입 포함
                                    skin_filter.extend(
                                        [
                                            s
                                            for s in df["skin_type"].dropna().unique()
                                            if s.startswith("복합/혼합")
                                        ]
                                    )
                                else:
                                    skin_filter.append(skin)

                            search_data = search_data[
                                search_data["skin_type"].isin(skin_filter)
                            ]
                            filter_messages.append(
                                f"피부 타입: {', '.join(detected_skin_types)}"
                            )

                        if detected_categories:
                            search_data = search_data[
                                search_data["category"].isin(detected_categories)
                            ]
                            filter_messages.append(
                                f"카테고리: {', '.join(detected_categories)}"
                            )

                        # if filter_messages:
                        # st.info(
                        #     f"{' | '.join(filter_messages)} 제품 중에서 검색합니다."
                        # )

                        reco_results = recommend_similar_products(
                            query_text=search_keyword_pre,
                            categories=None,
                            top_n=5,  # 카테고리별 상위 5개
                            vectorizer=st.session_state.vectorizer,
                            data=search_data,  # 피부 타입 + 카테고리 필터링된 데이터 전달
                        )

                        # 결과를 product_name 리스트로 변환 (유사도 순)
                        context_products = []
                        if isinstance(reco_results, dict):
                            for _, items in reco_results.items():
                                context_products.extend(items)
                            # reco_score 기준 정렬
                            context_products.sort(
                                key=lambda x: x.get("recommend_score", 0), reverse=True
                            )
                            context_search_results = [
                                p["product_name"]
                                for p in context_products
                                if p.get("product_name")
                            ]

                        st.session_state["context_search_results"] = (
                            context_search_results
                        )
                        st.session_state["context_search_products"] = (
                            context_products  # 전체 결과 저장
                        )
                        st.session_state["context_search_cache_key"] = cache_key
                else:
                    context_search_results = st.session_state.get(
                        "context_search_results", []
                    )
                    context_products = st.session_state.get(
                        "context_search_products", []
                    )

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            st.error(f"문맥 검색 중 오류가 발생했습니다: {str(e)}")
            with st.expander("오류 상세 정보"):
                st.code(error_detail)
            st.info("다른 검색 타입(상품명 또는 키워드)을 사용해주세요.")
            st.session_state.pop("vectorizer", None)

    # 검색창 (문맥 검색 결과 전달)
    selected_product = render_search_bar(
        product_options,
        clear_selected_product,
        context_search_results=context_search_results,
    )
    search_text = get_search_text()
    is_initial = is_initial_state(selected_sub_cat, selected_skin)

    # =========================
    # 인기 상품 TOP 5 (초기 상태)
    # =========================
    if is_initial:
        render_popular_products(df, select_product_from_reco)

    # =========================
    # 제품 상세 정보 (선택 시)
    # =========================
    if selected_product:
        st.caption(
            "상품 선택 상태에서는 검색 모드가 적용되지 않습니다. 재검색하려면 상품 선택을 취소해주세요."
        )
        with st.spinner("정보를 불러오는 중입니다..."):
            product_rows = df[df["product_name"] == selected_product]

        if product_rows.empty:
            st.warning("선택한 제품 정보를 찾을 수 없어요.")
        else:
            product_info = product_rows.iloc[0]

            # 제품 기본 정보
            render_product_info(product_info)

            # 대표 키워드
            render_top_keywords(product_info)

            # 대표 리뷰 & 평점 추이 (비동기 로드)
            product_id = product_info.get("product_id", "")
            review_id = product_info.get("representative_review_id_roberta", None)

            st.subheader("대표 리뷰")
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

            container_pos_review = st.empty()
            container_neg_review = st.empty()

            # AI 리뷰 요약 컨테이너 (비동기 리뷰 로딩 완료 후 렌더링)
            container_ai_summary = st.empty()

            container_trend = st.empty()

            # rerun시 캐시로 AI 요약 복구 렌더
            cache_pid = st.session_state.get("_analysis_cache_product_id")
            same_product_cache = str(product_id) == str(cache_pid)

            # rerun시에도 캐시로 복구 렌더
            if same_product_cache:
                pos_cache = st.session_state.get("_rep_positive_reviews_df_cache")
                neg_cache = st.session_state.get("_rep_negative_reviews_df_cache")
                if pos_cache is not None or neg_cache is not None:
                    # 기존 컨테이너 비우고 다시 렌더링
                    container_pos_review.empty()
                    container_neg_review.empty()
                    render_representative_review(
                        container_pos_review,
                        container_neg_review,
                        pos_cache if pos_cache is not None else pd.DataFrame(),
                        neg_cache if neg_cache is not None else pd.DataFrame(),
                        skip_scroll_apply_once,
                    )

                # AI 요약 복구 렌더링
                container_ai_summary.empty()
                render_ai_review_summary(container_ai_summary, product_info)

                trend_cache = st.session_state.get("_reviews_df_cache")
                if trend_cache is not None:
                    container_trend.empty()
                    render_rating_trend(
                        container_trend, trend_cache, skip_scroll_apply_once
                    )
            else:
                with container_ai_summary.container():
                    st.subheader("AI 리뷰 요약")
                    st.info("리뷰 데이터를 불러오는 중입니다...")

                # 상품이 바뀐 경우만 비동기 재로딩
                if st.session_state.get("last_loaded_product_id") != product_id:
                    # 순간 잔상 제거용
                    st.session_state["_rep_review_df_cache"] = None
                    st.session_state["_reviews_df_cache"] = None
                    st.session_state["_rep_reviews_df_cache"] = None
                    st.session_state["_rep_positive_reviews_df_cache"] = None
                    st.session_state["_rep_negative_reviews_df_cache"] = None
                    # 이전 상품 AI 요약 캐시 제거
                    old_pid = st.session_state.get("_analysis_cache_product_id")
                    if old_pid:
                        st.session_state.pop(f"ai_summary_{old_pid}", None)
                    st.session_state["_analysis_cache_product_id"] = str(product_id)

                    # 제품별 페이지 키 리셋
                    page_key = f"rep_review_page_{st.session_state['_analysis_cache_product_id']}"
                    st.session_state[page_key] = 0

                if st.session_state.get("last_loaded_product_id") != product_id:
                    load_product_analysis_async(
                        product_id,
                        product_info,
                        review_id,
                        container_pos_review,
                        container_neg_review,
                        container_trend,
                        skip_scroll_apply_once,
                        container_ai_summary,
                    )
                    st.session_state["last_loaded_product_id"] = product_id

    # =========================
    # 추천/검색 헤더
    # =========================
    sort_option = "추천순"
    if not is_initial:
        if selected_product:
            # 추천 상품 섹션을 fragment로 렌더링
            render_recommendation_section(df, selected_product)

        else:
            # 문맥 검색일 때 다른 헤더 표시
            search_type_header = st.session_state.get("search_type", "키워드")
            if search_type_header == "문맥" and search_keyword_pre:
                st.markdown("---")
                st.subheader(f'문맥 검색 결과: "{search_keyword_pre}"')

            col_1, col_2 = st.columns([8, 2])
            with col_2:
                sort_option = st.selectbox(
                    "정렬 옵션",
                    options=[
                        "추천순",
                        "평점 높은 순",
                        "리뷰 많은 순",
                        "가격 낮은 순",
                        "가격 높은 순",
                    ],
                    index=0,
                    key="sort_option",
                    label_visibility="collapsed",
                    on_change=skip_scroll_apply_once,
                )

    # =========================
    # 검색 결과 처리
    # =========================
    if is_initial:
        st.info("왼쪽 사이드바 또는 검색어를 입력하여 상품을 찾아보세요.")
    else:
        if not selected_product:
            search_type = st.session_state.get("search_type", "키워드")

            # 문맥 검색일 때는 벡터 유사도 결과 사용
            if search_type == "문맥" and search_keyword_pre:
                # 문맥 검색 결과에서 DataFrame 생성
                context_products = st.session_state.get("context_search_products", [])

                if context_products:
                    # product_id 리스트 추출
                    product_ids = [
                        p["product_id"] for p in context_products if p.get("product_id")
                    ]

                    # df에서 해당 상품들만 필터링
                    search_df_view = df[df["product_id"].isin(product_ids)].copy()

                    # reco_score와 similarity 추가
                    score_map = {
                        p["product_id"]: p.get("recommend_score", 0)
                        for p in context_products
                    }
                    sim_map = {
                        p["product_id"]: p.get("cosine_similarity", 0)
                        for p in context_products
                    }

                    search_df_view["reco_score"] = search_df_view["product_id"].map(
                        score_map
                    )
                    search_df_view["similarity"] = search_df_view["product_id"].map(
                        sim_map
                    )

                    # 추천 점수 순으로 정렬
                    search_df_view = search_df_view.sort_values(
                        "reco_score", ascending=False
                    )

                    # 문맥 검색 결과 직접 렌더링
                    if search_df_view.empty:
                        st.warning("표시할 상품이 없어요.🥺")
                    else:
                        # 카테고리별로 그룹화하여 표시
                        render_search_results_grid(
                            page_df=search_df_view,
                            full_df=search_df_view,
                            category_count=search_df_view["sub_category"].nunique(),
                            on_select_callback=select_product_from_reco,
                        )
                else:
                    st.warning("표시할 상품이 없어요.🥺")
            else:
                # 기존 필터 기반 검색
                filtered_df = apply_filters(
                    df,
                    selected_sub_cat,
                    selected_skin,
                    min_rating,
                    max_rating,
                    min_price,
                    max_price,
                    search_text,
                    search_type,
                )

                # 정렬 적용
                search_df_view = sort_products(filtered_df, sort_option)

                # 페이지네이션 계산
                items_page, total_pages, category_count = calculate_pagination(
                    search_df_view, selected_product
                )
                init_page_state(total_pages)

                # 필터 변경 감지
                check_filter_change(
                    search_text,
                    selected_sub_cat,
                    selected_skin,
                    min_rating,
                    max_rating,
                    min_price,
                    max_price,
                    sort_option,
                    safe_scroll_to_top,
                )

                # 페이지 슬라이스
                page_df = get_page_slice(
                    search_df_view, selected_product, items_page, category_count
                )

                # =========================
                # 상품 출력
                # =========================
                if page_df.empty:
                    st.warning("표시할 상품이 없어요.🥺")
                else:
                    render_search_results_grid(
                        page_df=page_df,
                        full_df=search_df_view,
                        category_count=category_count,
                        on_select_callback=select_product_from_reco,
                    )
                    # =========================
                    # 페이지네이션
                    # =========================
                    show_pagination = selected_product or selected_sub_cat
                    if show_pagination and total_pages > 1:
                        render_pagination(total_pages, safe_scroll_to_top)

    st.markdown(
        """
        <style>
        .footer {
            font-size: 12px;
            color: #777;
            text-align: center;
            padding: 16px 0;
        }
        </style>

        <div class="footer">
            <br><br><br>
            ⓒ 2026 Team Tensor · Multicampus team project
        </div>
        """,
        unsafe_allow_html=True,
    )

    # CSS 적용
    css.set_css()


if __name__ == "__main__":
    main()
