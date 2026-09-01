"""
전처리 후 Parquet 파일 기반 기본 통계 분석
- integrated_products_final.parquet
- category_summary.parquet
- partitioned_reviews_category_*.parquet
- JSON 형태로 data/ 에 저장 (old 파일 형식 준수)
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List

DATA_DIR = Path("data/processed_data")
PRODUCTS_DIR = DATA_DIR / "integrated_products_final"
CATEGORY_SUMMARY_DIR = DATA_DIR / "category_summary"
REVIEWS_DIR = DATA_DIR / "partitioned_reviews"
OUTPUT_JSON = Path("data") / "basic_stats_summary.json"


def load_all_data():
    """모든 Parquet 파일 로드"""
    print("=" * 60)
    print("데이터 로딩 중...")
    print("=" * 60)

    # 상품 데이터 (Hive 파티셔닝: category=*/data.parquet)
    product_files = sorted(PRODUCTS_DIR.glob("category=*/data.parquet"))
    all_products = []
    for f in product_files:
        df_p = pd.read_parquet(f)
        all_products.append(df_p)
    df_products = (
        pd.concat(all_products, ignore_index=True) if all_products else pd.DataFrame()
    )
    print(f"✓ 상품 데이터: {len(df_products):,}개")

    # 카테고리 통계
    df_category = pd.read_parquet(CATEGORY_SUMMARY_DIR / "data.parquet")
    print(f"✓ 카테고리: {len(df_category):,}개")

    # 리뷰 데이터 (Hive 파티셔닝: category=*/data.parquet)
    review_files = sorted(REVIEWS_DIR.glob("category=*/data.parquet"))
    all_reviews = []
    for f in review_files:
        df_r = pd.read_parquet(f)
        all_reviews.append(df_r)
        category = f.parent.name.replace("category=", "")
        print(f"  - {category}: {len(df_r):,}개 리뷰")

    df_reviews = (
        pd.concat(all_reviews, ignore_index=True) if all_reviews else pd.DataFrame()
    )
    print(f"✓ 전체 리뷰: {len(df_reviews):,}개")

    return df_products, df_category, df_reviews


def analyze_basic_stats(df_products, df_category, df_reviews) -> Dict[str, Any]:
    """기본 통계 분석 및 JSON 저장용 데이터 생성"""
    print("\n" + "=" * 60)
    print("기본 통계 분석")
    print("=" * 60)

    stats_result = {}

    # 메타 정보
    meta = {
        "total_products_seen": len(df_products),
        "total_reviews_collected": int(df_products["total_reviews"].sum()),
        "n_categories": len(df_category),
        "n_files": len(list(REVIEWS_DIR.glob("*.parquet"))),
    }

    # 1. 상품 통계
    print("\n[1] 상품 통계")
    print(f"  - 총 상품 수: {len(df_products):,}개")
    print(f"  - 평균 리뷰 수: {df_products['total_reviews'].mean():.1f}개")
    print(f"  - 최대 리뷰 수: {df_products['total_reviews'].max():,}개")
    print(f"  - 평균 평점(텍스트): {df_products['avg_rating_with_text'].mean():.2f}")
    print(
        f"  - 평균 평점(비텍스트): {df_products['avg_rating_without_text'].mean():.2f}"
    )
    print(f"  - 평균 텍스트 비율: {df_products['text_review_ratio'].mean():.1%}")

    # 2. 카테고리별 통계
    print("\n[2] 카테고리별 통계")
    category_stats = []
    for _, row in df_category.iterrows():
        print(f"\n  [{row['category']}]")
        print(f"    - 전체 리뷰: {row['total_reviews']:,}개")
        print(
            f"    - 텍스트 리뷰: {row['text_reviews']:,}개 ({row['text_reviews']/row['total_reviews']*100:.1f}%)"
        )
        print(f"    - 비텍스트 리뷰: {row['no_text_reviews']:,}개")
        print(
            f"    - 평점 분포: 5점({row['rating_5']:,}) 4점({row['rating_4']:,}) 3점({row['rating_3']:,}) 2점({row['rating_2']:,}) 1점({row['rating_1']:,})"
        )

        category_stats.append(
            {
                "category": row["category"],
                "total_reviews": int(row["total_reviews"]),
                "text_reviews": int(row["text_reviews"]),
                "no_text_reviews": int(row["no_text_reviews"]),
                "rating_5": int(row["rating_5"]),
                "rating_4": int(row["rating_4"]),
                "rating_3": int(row["rating_3"]),
                "rating_2": int(row["rating_2"]),
                "rating_1": int(row["rating_1"]),
            }
        )

    # 3. 리뷰 통계
    print("\n[3] 리뷰 통계")
    print(f"  - 전체 리뷰 수: {len(df_reviews):,}개")
    print(
        f"  - 텍스트 리뷰: {df_reviews['has_text'].sum():,}개 ({df_reviews['has_text'].sum()/len(df_reviews)*100:.1f}%)"
    )
    print(f"  - 평균 평점: {df_reviews['score'].mean():.2f}")

    # 평점 분포
    score_dist = df_reviews["score"].value_counts().sort_index()
    print("\n  평점 분포:")
    score_count_list = []
    for score, count in score_dist.items():
        print(f"    {score}점: {count:,}개 ({count/len(df_reviews)*100:.1f}%)")
        score_count_list.append({"score": int(score), "cnt": int(count)})

    # 4. 브랜드 통계
    print("\n[4] 상위 브랜드 (리뷰 수 기준)")
    brand_reviews = (
        df_products.groupby("brand")["total_reviews"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    brand_stats = []
    for brand, count in brand_reviews.items():
        print(f"  - {brand}: {count:,}개")
        brand_stats.append({"brand": brand, "total_reviews": int(count)})

    # 5. 카테고리별 상품 수
    category_product_counts = (
        df_products.groupby("category").size().reset_index(name="product_cnt")
    )
    category_product_counts = category_product_counts.sort_values(
        "product_cnt", ascending=False
    )

    # 6. 상품별 리뷰 수
    product_review_counts = df_products[["product_id", "total_reviews"]].copy()
    product_review_counts.columns = ["product_id", "review_cnt"]
    product_review_counts = product_review_counts.sort_values(
        "review_cnt", ascending=False
    )

    # 7. 상품별 리뷰 수 통계 요약
    review_cnt_summary = (
        df_products["total_reviews"]
        .describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
        .to_dict()
    )

    # 8. 리뷰 수 구간별 상품 수
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf")]
    labels = [
        "1~1",
        "2~2",
        "3~4",
        "5~9",
        "10~19",
        "20~49",
        "50~99",
        "100~199",
        "200~499",
        "500~999",
        "1000+",
    ]
    df_products["review_bin"] = pd.cut(
        df_products["total_reviews"], bins=bins, labels=labels, right=False
    )
    review_bins = df_products["review_bin"].value_counts().sort_index()
    review_bins_list = [
        {"bin": str(idx), "product_cnt": int(val)} for idx, val in review_bins.items()
    ]

    # 9. 카테고리별 평점 분포 (category_score_distribution)
    category_score_dist = []
    for _, row in df_category.iterrows():
        cat = row["category"]
        for score in [1, 2, 3, 4, 5]:
            category_score_dist.append(
                {"category": cat, "score": score, "cnt": int(row[f"rating_{score}"])}
            )

    # 10. 카테고리별 평균 평점 (category_score_summary)
    category_score_summary = []
    for _, row in df_category.iterrows():
        total_cnt = int(row["total_reviews"])
        score_sum = sum(
            score * int(row[f"rating_{score}"]) for score in [1, 2, 3, 4, 5]
        )
        mean_score = score_sum / total_cnt if total_cnt > 0 else 0
        category_score_summary.append(
            {
                "category": row["category"],
                "total_cnt": total_cnt,
                "score_sum": score_sum,
                "mean_score": mean_score,
            }
        )

    # 결과 구조화 (old 파일 형식)
    stats_result = {
        "meta": meta,
        "category_count": category_product_counts.to_dict(orient="records"),
        "category_score_part_count": category_score_dist,
        "category_mean_score": category_score_summary,
        "total_review_bins_count": review_bins_list,
        "summary": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in review_cnt_summary.items()
        },
        "product_review_descending": product_review_counts.head(100).to_dict(
            orient="records"
        ),  # 상위 100개만
        "score_count": score_count_list,
        "category_details": category_stats,
        "brand_top10": brand_stats,
    }

    return stats_result


def analyze_rating_distribution(df_products) -> Dict[str, Any]:
    """평점 분포 분석"""
    print("\n" + "=" * 60)
    print("평점 분포 상세 분석")
    print("=" * 60)

    total_ratings = {
        "1점": int(df_products["rating_1"].sum()),
        "2점": int(df_products["rating_2"].sum()),
        "3점": int(df_products["rating_3"].sum()),
        "4점": int(df_products["rating_4"].sum()),
        "5점": int(df_products["rating_5"].sum()),
    }

    total = sum(total_ratings.values())
    print("\n전체 평점 분포:")
    for rating, count in total_ratings.items():
        print(f"  {rating}: {count:,}개 ({count/total*100:.1f}%)")

    return {"rating_distribution": total_ratings, "total_ratings": total}


def save_json(data: Dict[str, Any], output_path: Path):
    """JSON 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ JSON 저장 완료: {output_path}")
    return str(output_path)


def main():
    """메인 실행 함수"""
    # 데이터 로드
    df_products, df_category, df_reviews = load_all_data()

    # 기본 통계
    basic_stats = analyze_basic_stats(df_products, df_category, df_reviews)

    # 평점 분포
    rating_dist = analyze_rating_distribution(df_products)

    # 최종 결과 통합
    final_result = {**basic_stats, "rating_distribution_detail": rating_dist}

    # JSON 저장
    save_json(final_result, OUTPUT_JSON)

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)
    print(f"저장 위치: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
