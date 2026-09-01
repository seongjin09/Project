# athena_queries.py
from services.athena_client import athena_read, quote_list

# athena_read를 athena_read_cached로 사용
athena_read_cached = athena_read


# quote_str 함수 추가 (없으면)
def quote_str(s):
    """문자열을 SQL 쿼리용으로 이스케이프"""
    return str(s).replace("'", "''")


# 테이블명
PRODUCT_TABLE = "coupang_db.integrated_products_final"
REVIEWS_TABLE = "coupang_db.reviews"


# =========================
# 1) 전체 상품 조회 쿼리
# =========================
SQL_ALL_PRODUCTS = f"""
SELECT
  product_id,
  product_name,
  brand,
  category_path,
  path,
  price,
  delivery_type,
  product_url,
  img_url,
  skin_type,
  top_keywords,
  sentiment_analysis,
  avg_rating_with_text,
  avg_rating_without_text,
  text_review_ratio,
  total_reviews,
  rating_1,
  rating_2,
  rating_3,
  rating_4,
  rating_5,
  product_vector_roberta_sentiment,
  product_vector_roberta_semantic,
  sentiment_score,
  positive_representative_ids,
  positive_representative_scores,
  negative_representative_ids,
  negative_representative_scores,
  category
FROM {PRODUCT_TABLE}
"""


def fetch_all_products():
    """
    전체 상품 조회 (LIMIT 없음)
    """
    return athena_read_cached(SQL_ALL_PRODUCTS)


# =========================
# 2) 상품별 리뷰 조회
# =========================
def fetch_reviews_by_product(product_id: str):
    pid = quote_str(product_id)

    sql = f"""
    SELECT
      category,
      product_id,
      id,
      full_text,
      title,
      content,
      has_text,
      score,
      label,
      tokens,
      char_length,
      token_count,
      date,
      collected_at,
      nickname,
      has_image,
      helpful_count,
      sentiment_score,
      roberta_sentiment,
      roberta_semantic
    FROM {REVIEWS_TABLE}
    WHERE product_id = '{pid}'
    ORDER BY date DESC
    """
    return athena_read_cached(sql)


# =========================
# 3) 조건 기반 상품 검색
# =========================
def search_products_flexible(
    categories,
    skin_types,
    min_rating,
    max_rating,
    min_price,
    max_price,
):
    """
    categories / skin_types가 비어있으면 해당 조건 제거
    """
    where_parts = ["1=1"]

    if categories:
        categories_in = quote_list(categories)
        where_parts.append(f"category IN ({categories_in})")

    if skin_types:
        skin_types_in = quote_list(skin_types)
        where_parts.append(
            f"""
            (
              CASE
                WHEN skin_type LIKE '복합/혼합%' THEN '복합/혼합'
                ELSE skin_type
              END
            ) IN ({skin_types_in})
            """.strip()
        )

    where_parts.append(
        f"avg_rating_with_text BETWEEN {float(min_rating)} AND {float(max_rating)}"
    )
    where_parts.append(f"price BETWEEN {int(min_price)} AND {int(max_price)}")

    where_sql = "\n  AND ".join(where_parts)

    sql = f"""
    SELECT
      product_id,
      product_name,
      brand,
      category,
      price,
      skin_type,
      total_reviews,
      avg_rating_with_text,
      rating_1,
      rating_2,
      rating_3,
      rating_4,
      rating_5,
      sentiment_score,
      top_keywords,
      product_url
    FROM {PRODUCT_TABLE}
    WHERE {where_sql}
    ORDER BY total_reviews DESC, avg_rating_with_text DESC
    """
    return athena_read_cached(sql)


# =========================
# 4) 대표 리뷰 텍스트 조회
# =========================
def fetch_representative_review_text(product_id: str, review_id: int):
    """
    특정 상품의 특정 리뷰 1개만 조회
    """
    pid = quote_str(product_id)

    sql = f"""
    SELECT full_text, title, content
    FROM {REVIEWS_TABLE}
    WHERE product_id = '{pid}' AND id = {int(review_id)}
    LIMIT 1
    """
    return athena_read_cached(sql)


# =========================
# 5) 벡터 검색용 상품 데이터 로드
# =========================
def load_products_data_from_athena(
    categories=None,
    vector_type: str = "roberta_semantic",
    table_name: str = "coupang_db.integrated_products_final",
):
    """
    문맥 검색용 상품 데이터 로드
    """
    import json

    vector_col = f"product_vector_{vector_type}"

    where_clause = ""
    if categories:
        # SUB_ALIAS 역매핑: "리무버" → ["립메이크업/포인트리무버", "립/아이리무버"]
        ALIAS_REVERSE = {
            "리무버": [
                "립메이크업/포인트리무버",
                "립/아이리무버",
                "아이메이크업/포인트리무버",
                "립 메이크업/포인트리무버",
            ],
            "알로에/수딩/애프터선": ["알로에/수딩/에프터선"],
        }

        expanded_cats = []
        for c in categories:
            # 원본 추가
            expanded_cats.append(c)
            # alias가 있으면 원본 카테고리들도 추가
            if c in ALIAS_REVERSE:
                expanded_cats.extend(ALIAS_REVERSE[c])

        # sub_category(정규화된 값: /구분)와 Athena category(_구분, 공백 포함) 모두 매칭
        all_variants = []
        for c in expanded_cats:
            all_variants.append(c)  # 슬래시 버전
            all_variants.append(c.replace("/", "_"))  # 언더스코어 버전
            all_variants.append(c.replace("/", " "))  # 공백 버전

        all_cats = list(set(all_variants))
        cat_list = quote_list(all_cats)
        where_clause = f"WHERE category IN ({cat_list})"

    sql = f"""
    SELECT
        product_id,
        product_name,
        brand,
        category,
        sentiment_score,
        avg_rating_with_text,
        total_reviews,
        product_url,
        price,
        top_keywords,
        {vector_col}
    FROM {table_name}
    {where_clause}
    """

    df = athena_read_cached(sql)

    # 벡터 JSON 파싱
    if (
        not df.empty
        and vector_col in df.columns
        and df[vector_col].dtype == object
        and isinstance(df[vector_col].iloc[0], str)
    ):
        df[vector_col] = df[vector_col].apply(json.loads)

    return df


# =========================
# 6) 상위 리뷰 텍스트 조회 (여러 개)
# =========================
def fetch_top_reviews_text(product_id: str, review_ids: list):
    """
    특정 상품의 여러 리뷰를 한 번에 조회

    Args:
        product_id: 상품 ID
        review_ids: 리뷰 ID 리스트

    Returns:
        pd.DataFrame: 리뷰 데이터
    """
    if not review_ids:
        import pandas as pd

        return pd.DataFrame()

    pid = quote_str(product_id)
    ids_str = ", ".join(str(int(rid)) for rid in review_ids)

    sql = f"""
    SELECT 
        id,
        full_text, 
        title, 
        content,
        score,
        sentiment_score
    FROM {REVIEWS_TABLE}
    WHERE product_id = '{pid}' AND id IN ({ids_str})
    """
    return athena_read_cached(sql)
