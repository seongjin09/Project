"""
제품 분석 컴포넌트 (대표 키워드, 대표 리뷰, 평점 추이)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.load_data import rating_trend
from utils.data_utils import load_reviews_athena, load_top_reviews_athena
from services.recommend_similar_products import recommend_similar_products
from services.review_summarizer import get_cached_summary


def render_top_keywords(product_info: pd.Series):
    """대표 키워드 렌더링"""
    st.markdown("<div style='height:64px;'></div>", unsafe_allow_html=True)
    st.subheader("대표 키워드")
    top_kw = product_info.get("top_keywords_str", [])
    if isinstance(top_kw, str):
        top_kw = [k.strip() for k in top_kw.split(",") if k.strip()]
    cols = st.columns(5)

    for col, kw in zip(cols, top_kw):
        with col:
            st.markdown(
                f""" 
                    <div style=" 
                    padding:12px; 
                    border-radius:12px; 
                    background:#f5f7fa; 
                    text-align:center; 
                    font-weight:600; 
                "> 
                #{kw} 
                </div> 
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:64px;'></div>", unsafe_allow_html=True)


def _extract_review_texts(reviews_df: pd.DataFrame) -> list[str]:
    """리뷰 DataFrame에서 텍스트 추출"""
    texts = []
    if reviews_df is None or reviews_df.empty:
        return texts
    for _, row in reviews_df.iterrows():
        text = ""
        if "full_text" in reviews_df.columns and pd.notna(row.get("full_text")):
            text = str(row["full_text"])
        if not text:
            title = str(row.get("title") or "") if "title" in reviews_df.columns else ""
            content = (
                str(row.get("content") or "") if "content" in reviews_df.columns else ""
            )
            text = (title + " " + content).strip()
        if text:
            texts.append(text)
    return texts


def render_ai_review_summary(container, product_info: pd.Series):
    """AI 리뷰 요약 렌더링 (컨테이너 기반, 캐시 활용)"""
    product_name = product_info.get("product_name", "")
    product_id = str(product_info.get("product_id", ""))

    # 세션에 이미 요약이 있으면 바로 렌더링
    summary_cache_key = f"ai_summary_{product_id}"
    summary = st.session_state.get(summary_cache_key)

    if summary:
        with container.container():
            st.subheader("AI 리뷰 요약")
            st.markdown(
                f"""
                <div style="
                    padding: 20px;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #f0f4ff 0%, #f5f0ff 100%);
                    border-left: 4px solid #6366f1;
                    margin: 8px 0;
                    line-height: 1.7;
                ">
                    {summary}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
    else:
        with container.container():
            st.subheader("AI 리뷰 요약")
            st.info("💬 리뷰 데이터를 불러오는 중입니다...")


def _generate_ai_summary(container, product_info: pd.Series):
    """
    리뷰 로딩 완료 후 AI 요약 생성 및 렌더링
    (load_product_analysis_async에서 호출)
    """
    product_name = product_info.get("product_name", "")
    product_id = str(product_info.get("product_id", ""))
    summary_cache_key = f"ai_summary_{product_id}"

    # 이미 생성된 요약이 있으면 스킵
    if summary_cache_key in st.session_state:
        render_ai_review_summary(container, product_info)
        return

    # 대표 키워드 추출
    top_kw = product_info.get("top_keywords_str", "")
    if isinstance(top_kw, list):
        top_kw = ",".join(top_kw)

    # 캐시된 리뷰 가져오기
    pos_cache = st.session_state.get("_rep_positive_reviews_df_cache")
    neg_cache = st.session_state.get("_rep_negative_reviews_df_cache")

    pos_texts = _extract_review_texts(pos_cache)
    neg_texts = _extract_review_texts(neg_cache)

    if not pos_texts and not neg_texts:
        with container.container():
            st.subheader("AI 리뷰 요약")
            st.info("요약할 리뷰 데이터가 없습니다.")
        return

    # AI 요약 생성
    with container.container():
        st.subheader("AI 리뷰 요약")
        with st.spinner("🤖 AI가 리뷰를 분석하고 있습니다..."):
            summary = get_cached_summary(
                product_name=product_name,
                keywords_str=top_kw,
                positive_reviews_str="\n".join(pos_texts),
                negative_reviews_str="\n".join(neg_texts),
            )
            st.session_state[summary_cache_key] = summary

    # 생성 완료 후 렌더링
    render_ai_review_summary(container, product_info)


def render_representative_review(
    container_pos,
    container_neg,
    positive_reviews_df: pd.DataFrame,
    negative_reviews_df: pd.DataFrame,
    skip_scroll_callback,
):
    """대표 리뷰 렌더링 (긍정/부정 독립 컨테이너)"""
    pid = st.session_state.get("_analysis_cache_product_id", "unknown")

    _render_single_review_section(
        container_pos,
        positive_reviews_df,
        "positive",
        "긍정 대표 리뷰",
        pid,
        skip_scroll_callback,
    )
    _render_single_review_section(
        container_neg,
        negative_reviews_df,
        "negative",
        "부정 대표 리뷰",
        pid,
        skip_scroll_callback,
    )


def _render_single_review_section(
    container, reviews_df, review_type, title, pid, skip_scroll_callback
):
    """단일 리뷰 타입 섹션 렌더링"""
    with container.container():
        st.markdown(f"#### {title}")
        if reviews_df is None or reviews_df.empty:
            st.info(f"{title}가 없습니다.")
        else:
            _render_review_pagination(
                reviews_df, review_type, pid, skip_scroll_callback
            )


@st.fragment
def _render_review_pagination(
    reviews_df: pd.DataFrame, review_type: str, product_id: str, skip_scroll_callback
):
    """개별 리뷰 페이지네이션 렌더링 (fragment로 독립 실행)"""
    # 캐시 ID를 키에 포함하여 중복 방지
    cache_id = st.session_state.get("_analysis_cache_product_id", product_id)
    page_key = f"rep_review_page_{review_type}_{cache_id}"

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    total = len(reviews_df)
    page = int(st.session_state[page_key])
    page = max(0, min(page, total - 1))
    st.session_state[page_key] = page

    # 현재 페이지 리뷰 표시
    row = reviews_df.iloc[page]

    meta = []
    if "date" in reviews_df.columns and pd.notna(row.get("date")):
        meta.append(str(row.get("date")))
    if "score" in reviews_df.columns and pd.notna(row.get("score")):
        score = row.get("score")
        stars = "⭐" * int(score) if pd.notna(score) else ""
        meta.append(f"{stars} {score}점")
    if "sentiment_score" in reviews_df.columns and pd.notna(row.get("sentiment_score")):
        sentiment = row.get("sentiment_score")
        if pd.notna(sentiment):
            sentiment_pct = f"{float(sentiment) * 100:.1f}%"
            emoji = "😊" if float(sentiment) >= 0.5 else "😟"
            meta.append(f"{emoji} {sentiment_pct}")

    if meta:
        st.caption(" · ".join(meta))

    # full_text 우선, 없으면 title+content
    text = ""
    if "full_text" in reviews_df.columns and pd.notna(row.get("full_text")):
        text = str(row.get("full_text") or "")
    if not text:
        title = str(row.get("title") or "") if "title" in reviews_df.columns else ""
        content = (
            str(row.get("content") or "") if "content" in reviews_df.columns else ""
        )
        text = (title + "\n\n" + content).strip()

    if text:
        st.text(text)
    else:
        st.info("표시할 리뷰 텍스트가 없습니다.")

    # 페이지네이션 버튼
    col_l, col_m, col_r = st.columns([2, 6, 2])

    def prev_page():
        skip_scroll_callback()
        st.session_state[page_key] = max(0, st.session_state[page_key] - 1)

    def next_page():
        skip_scroll_callback()
        st.session_state[page_key] = min(total - 1, st.session_state[page_key] + 1)

    with col_l:
        st.button(
            "◀ 이전",
            on_click=prev_page,
            disabled=(page == 0),
            use_container_width=True,
            key=f"rep_prev_{review_type}_{cache_id}",
        )
    with col_m:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px;'>({page+1} / {total})</div>",
            unsafe_allow_html=True,
        )
    with col_r:
        st.button(
            "다음 ▶",
            on_click=next_page,
            disabled=(page >= total - 1),
            use_container_width=True,
            key=f"rep_next_{review_type}_{cache_id}",
        )


def render_rating_trend(container, reviews_df: pd.DataFrame, skip_scroll_callback):
    """평점 추이 렌더링"""
    with container.container():
        st.markdown("<div style='height:64px;'></div>", unsafe_allow_html=True)
        st.subheader("평점 추이")

        if (
            reviews_df.empty
            or "date" not in reviews_df.columns
            or "score" not in reviews_df.columns
        ):
            st.info("평점 추이를 그릴 리뷰 데이터가 없습니다.")
            return

        review_df = reviews_df[["date", "score"]].copy()
        review_df["date"] = pd.to_datetime(review_df["date"], errors="coerce")
        review_df["score"] = pd.to_numeric(review_df["score"], errors="coerce")
        review_df = review_df.dropna(subset=["date", "score"]).sort_values("date")

        if review_df.empty:
            st.info("평점 추이를 그릴 수 있는 날짜/평점 데이터가 없습니다.")
            return

        min_date = review_df["date"].min().date()
        max_date = review_df["date"].max().date()

        pid = st.session_state.get("_analysis_cache_product_id", "unknown")

        # fragment로 UI 부분만 분리
        _render_rating_trend_ui(
            review_df, min_date, max_date, pid, skip_scroll_callback
        )


@st.fragment
def _render_rating_trend_ui(
    review_df: pd.DataFrame, min_date, max_date, pid: str, skip_scroll_callback
):
    """평점 추이 UI 렌더링 (fragment로 독립 실행)"""
    freq_key = f"rating_freq_{pid}"
    date_key = f"rating_date_{pid}"
    reset_key = f"rating_reset_{pid}"

    col_left, col_mid, col_right, _ = st.columns([1, 1, 1, 1])

    with col_left:
        freq_label = st.selectbox(
            "평균 기준",
            ["일간", "주간", "월간"],
            index=2,
            key=freq_key,
        )

    freq_map = {
        "일간": ("D", 7),
        "주간": ("W", 4),
        "월간": ("ME", 3),
    }
    freq, ma_window = freq_map[freq_label]

    DATE_RANGE_KEY = f"rating_date_range_{pid}"
    default_date_range = (min_date, max_date)

    # 저장된 날짜 범위가 있으면 사용, 없으면 기본값
    if DATE_RANGE_KEY not in st.session_state:
        st.session_state[DATE_RANGE_KEY] = default_date_range

    with col_mid:
        date_range = st.date_input(
            "기간 선택",
            value=st.session_state[DATE_RANGE_KEY],
            min_value=min_date,
            max_value=max_date,
            key=date_key,
        )

    def reset_date_range():
        skip_scroll_callback()
        st.session_state[DATE_RANGE_KEY] = default_date_range

    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
            "↺",
            key=reset_key,
            help="날짜 초기화",
            on_click=reset_date_range,
        )

    trend_df = pd.DataFrame()
    is_date_range_ready = False

    if isinstance(date_range, tuple) and len(date_range) == 2:
        is_date_range_ready = True
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # 날짜 범위 저장
        st.session_state[DATE_RANGE_KEY] = (start_date.date(), end_date.date())

        date_df = review_df.loc[
            (review_df["date"] >= start_date) & (review_df["date"] <= end_date)
        ]
        if not date_df.empty:
            trend_df = rating_trend(date_df, freq=freq, ma_window=ma_window)
    else:
        st.info("마지막 날짜를 선택해주세요.📆")

    if is_date_range_ready and not trend_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=trend_df["date"],
                y=trend_df["avg_score"],
                name=f"{freq_label} 평균",
                marker_color="slateblue",
                opacity=0.4,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trend_df["date"],
                y=trend_df["ma"],
                mode="lines",
                name=f"추세 ({ma_window}개{freq_label} 이동평균)",
                line=dict(color="royalblue", width=3),
            )
        )
        fig.update_layout(
            yaxis=dict(range=[1, 5.1]),
            xaxis_title="날짜",
            yaxis_title="평균 평점",
            hovermode="x unified",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif is_date_range_ready and trend_df.empty:
        st.info("선택한 기간에 대한 평점 데이터가 없습니다.")


def load_product_analysis_async(
    product_id: str,
    product_info: pd.Series,
    review_id,
    container_pos_review,
    container_neg_review,
    container_trend,
    skip_scroll_callback,
    container_ai_summary=None,
):
    """
    비동기로 대표 리뷰, 평점 추이, 추천 상품 로드
    각 컨테이너에 도착 즉시 렌더링

    Args:
        product_id: 제품 ID
        product_info: 제품 정보 Series
        review_id: 대표 리뷰 ID
        container_pos_review: 긍정 리뷰 placeholder
        container_neg_review: 부정 리뷰 placeholder
        container_trend: 평점 추이 placeholder
        skip_scroll_callback: 스크롤 스킵 콜백
        container_ai_summary: AI 요약 placeholder
    """
    # 초기 로딩 메시지 표시
    with container_pos_review.container():
        st.markdown("긍정 대표 리뷰")
        st.info("긍정 대표 리뷰를 불러오는 중...")

    with container_neg_review.container():
        st.markdown("부정 대표 리뷰")
        st.info("부정 대표 리뷰를 불러오는 중...")

    with container_trend.container():
        st.markdown("평점 추이")
        st.info("평점 데이터를 불러오는 중...")

    pid = str(product_id)

    reviews_loaded = {"positive": False, "negative": False}

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_type = {}

        if product_id:
            f_pos = executor.submit(
                load_top_reviews_athena, pid, product_info, 5, "positive"
            )
            future_to_type[f_pos] = "REVIEW_POSITIVE"

            f_neg = executor.submit(
                load_top_reviews_athena, pid, product_info, 5, "negative"
            )
            future_to_type[f_neg] = "REVIEW_NEGATIVE"

            f_trend = executor.submit(load_reviews_athena, pid)
            future_to_type[f_trend] = "TREND"

        if product_id and st.session_state.get("reco_target_product_id") != product_id:
            # 현재 상품의 카테고리로만 초기 검색 (성능 최적화)
            current_category = (
                product_info.get("sub_category")
                if pd.notna(product_info.get("sub_category"))
                else None
            )
            f_reco = executor.submit(
                recommend_similar_products,
                product_id=product_id,
                categories=[current_category] if current_category else None,
                top_n=100,
            )
            future_to_type[f_reco] = "RECO"

        # 도착 즉시 세션 상태 업데이트 + 컨테이너 렌더링

        for future in as_completed(future_to_type):
            task_type = future_to_type[future]

            try:
                result = future.result()

                if task_type == "REVIEW_POSITIVE":
                    st.session_state["_rep_positive_reviews_df_cache"] = result
                    st.session_state["_analysis_cache_product_id"] = pid
                    reviews_loaded["positive"] = True
                    _render_single_review_section(
                        container_pos_review,
                        result,
                        "positive",
                        "긍정 대표 리뷰",
                        pid,
                        skip_scroll_callback,
                    )

                elif task_type == "REVIEW_NEGATIVE":
                    st.session_state["_rep_negative_reviews_df_cache"] = result
                    st.session_state["_analysis_cache_product_id"] = pid
                    reviews_loaded["negative"] = True
                    _render_single_review_section(
                        container_neg_review,
                        result,
                        "negative",
                        "부정 대표 리뷰",
                        pid,
                        skip_scroll_callback,
                    )

                elif task_type == "TREND":
                    st.session_state["_reviews_df_cache"] = result
                    st.session_state["_analysis_cache_product_id"] = pid
                    render_rating_trend(container_trend, result, skip_scroll_callback)

                elif task_type == "RECO":
                    reco_list = (
                        result
                        if isinstance(result, list)
                        else [item for items in result.values() for item in items]
                    )
                    # 즉시 세션 업데이트 → fragment가 바로 감지
                    st.session_state["reco_cache"] = reco_list
                    st.session_state["reco_target_product_id"] = product_id
                    # 추천 섹션 fragment가 사용하는 cache_key도 동기화 (현재 상품 카테고리 기준)
                    current_category = (
                        product_info.get("sub_category")
                        if pd.notna(product_info.get("sub_category"))
                        else None
                    )
                    st.session_state["reco_cache_key"] = (
                        "product",
                        product_id,
                        tuple([current_category]) if current_category else None,
                    )

            except Exception as e:
                if task_type == "REVIEW_POSITIVE":
                    st.session_state["_rep_positive_reviews_df_cache"] = pd.DataFrame()
                    reviews_loaded["positive"] = True
                    with container_pos_review.container():
                        st.markdown("긍정 대표 리뷰")
                        st.error(f"로드 실패: {e}")
                elif task_type == "REVIEW_NEGATIVE":
                    st.session_state["_rep_negative_reviews_df_cache"] = pd.DataFrame()
                    reviews_loaded["negative"] = True
                    with container_neg_review.container():
                        st.markdown("부정 대표 리뷰")
                        st.error(f"로드 실패: {e}")
                elif task_type == "TREND":
                    with container_trend.container():
                        st.markdown("평점 추이")
                        st.error(f"평점 추이 로드 실패: {e}")
                elif task_type == "RECO":
                    st.error(f"추천 상품 로드 실패: {e}")

    # 긍부정 리뷰 로딩 완료 후 AI 요약 생성
    if (
        container_ai_summary
        and reviews_loaded["positive"]
        and reviews_loaded["negative"]
    ):
        _generate_ai_summary(container_ai_summary, product_info)
