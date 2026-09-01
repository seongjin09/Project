import ast
import re
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from itertools import chain
import seaborn as sns
import random
import matplotlib.gridspec as gridspec
import platform
from pathlib import Path

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

# 운영체제별 한글 폰트 설정
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False
    FONT_PATH = r"C:\WINDOWS\FONTS\MALGUNSL.TTF"
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
    plt.rcParams["axes.unicode_minus"] = False
    FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
else:  # Linux
    plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False
    FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

DATA_DIR = os.path.join("data", "processed_data")
PRODUCTS_DIR = os.path.join(DATA_DIR, "integrated_products_final")
CATEGORY_SUMMARY_DIR = os.path.join(DATA_DIR, "category_summary")
REVIEWS_DIR = os.path.join(DATA_DIR, "partitioned_reviews")


# Parquet 파일 로딩
print(f"상품 데이터: {PRODUCTS_DIR}")
print(f"카테고리 통계: {CATEGORY_SUMMARY_DIR}")
print(f"리뷰 데이터: {REVIEWS_DIR}")

# 상품 데이터 로딩 (Hive 파티셔닝: category=*/data.parquet)
product_files = sorted(Path(PRODUCTS_DIR).glob("category=*/data.parquet"))
all_products_list = []
for f in product_files:
    df_p = pd.read_parquet(f)
    all_products_list.append(df_p)
df_product = (
    pd.concat(all_products_list, ignore_index=True)
    if all_products_list
    else pd.DataFrame()
)

# 카테고리 통계 로딩
df_category_summary = pd.read_parquet(
    os.path.join(CATEGORY_SUMMARY_DIR, "data.parquet")
)

# 모든 리뷰 데이터 로딩 (Hive 파티셔닝: category=*/data.parquet)
review_files = sorted(Path(REVIEWS_DIR).glob("category=*/data.parquet"))
all_reviews_list = []
for f in review_files:
    df_r = pd.read_parquet(f)
    all_reviews_list.append(df_r)

if all_reviews_list:
    df_review_all = pd.concat(all_reviews_list, ignore_index=True)
else:
    df_review_all = pd.DataFrame()

# 텍스트 있는 리뷰만 분리
df_review = df_review_all[df_review_all["has_text"] == True].copy()

print("\n===== 데이터 로드 완료 =====")
print(f"상품 수: {len(df_product)}")
print(f"카테고리 수: {len(df_category_summary)}")
print(f"전체 리뷰 수: {len(df_review_all)}")
print(f"텍스트 리뷰 수: {len(df_review)}")
print(f"\n상품 데이터 컬럼: {df_product.columns.tolist()[:10]}...")
print(f"리뷰 데이터 컬럼: {df_review_all.columns.tolist()}")

print("\n===== 카테고리 통계 =====")
print(df_category_summary)

# 상품별 통계 계산
product_stats = (
    df_review_all.groupby("product_id")
    .agg(
        mean_score=("score", "mean"),
        mean_helpful=("helpful_count", "mean"),
        review_count=("score", "count"),
    )
    .reset_index()
)

print("\n===== 상품별 통계 (상위 10개) =====")
print(product_stats.head(10))

# 리뷰 많은 상품 TOP 6
top_6_products = (
    df_review_all.groupby("product_id")
    .size()
    .reset_index(name="total_reviews")
    .merge(df_product[["product_id", "product_name"]], on="product_id", how="left")
    .sort_values("total_reviews", ascending=False)
    .head(6)
)

print("\n===== 리뷰 많은 상품 TOP 6 =====")
print(top_6_products)


# 리뷰 많은 상품 TOP 6 - 평점 분포 차트
# (date 필드가 없으므로 월별이 아닌 전체 평점 분포만 표시)
top_6_product_ids = top_6_products["product_id"].tolist()

# 리뷰 많은 상품 TOP 6 - 월별 시계열 분석 차트
top_6_product_ids = top_6_products["product_id"].tolist()

# date 컬럼을 datetime 객체로 변환하고 월 단위 주기 컬럼 생성
df_top6 = df_review_all[df_review_all["product_id"].isin(top_6_product_ids)].copy()
df_top6["date"] = pd.to_datetime(df_top6["date"], errors="coerce")
df_top6 = df_top6.dropna(subset=["date", "score"])
df_top6["year_month"] = df_top6["date"].dt.to_period("M").astype(str)

fig = plt.figure(figsize=(18, 15))
gs = gridspec.GridSpec(3, 2, figure=fig)

# 평점별 고정 색상 설정 (빨강~초록 계열)
color_map = {1: "#d32f2f", 2: "#ff6f00", 3: "#fbc02d", 4: "#7cb342", 5: "#388e3c"}

for i, pid in enumerate(top_6_product_ids):
    df_p = df_top6[df_top6["product_id"] == pid]

    if len(df_p) == 0:
        continue

    product_name = (
        df_product.loc[df_product["product_id"] == pid, "product_name"].values[0]
        if pid in df_product["product_id"].values
        else pid
    )

    # 월별 평점 분포 데이터 집계 (unstack을 통해 막대용 행렬 생성)
    rating_dist = df_p.groupby(["year_month", "score"]).size().unstack(fill_value=0)
    rating_dist = rating_dist.reindex(columns=[1, 2, 3, 4, 5], fill_value=0)

    # 월별 평균 평점 계산
    monthly_mean = df_p.groupby("year_month")["score"].mean()

    ax1 = fig.add_subplot(gs[i // 2, i % 2])

    # Y1: 누적 막대 그래프 (평점 분포)
    rating_dist.plot(
        kind="bar",
        stacked=True,
        ax=ax1,
        color=[color_map[c] for c in rating_dist.columns],
        alpha=0.8,
    )
    ax1.set_title(f"{product_name[:40]}...\n월별 평점 분포 및 추이", pad=20)
    ax1.set_xlabel("조회 월")
    ax1.set_ylabel("리뷰 수 (막대)")
    ax1.legend(title="평점", loc="upper left", ncol=5, fontsize=8)

    # x축 레이블 최적화 (데이터가 많을 경우 간격 조절)
    xticks = ax1.get_xticks()
    if len(rating_dist.index) > 10:
        ax1.set_xticks(xticks[::2])
        ax1.set_xticklabels(rating_dist.index[::2], rotation=45, ha="right")
    else:
        ax1.set_xticklabels(rating_dist.index, rotation=45, ha="right")

    # Y2: 이중축 선 그래프 (평균 평점)
    ax2 = ax1.twinx()
    monthly_mean.plot(
        kind="line",
        ax=ax2,
        color="blue",
        marker="o",
        markersize=4,
        linewidth=2,
        label="평균 평점",
    )
    ax2.set_ylabel("평균 평점 (선)")
    ax2.set_ylim(1, 5.2)  # 평점 범위 고정
    ax2.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.show()


# ===== 시각화 1 (6개 차트) =====
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 전체 상품 평점 분포 (상단 왼쪽 - Horizontal Bar)
rating_sum = df_category_summary[
    ["rating_1", "rating_2", "rating_3", "rating_4", "rating_5"]
].sum()
rating_sum.index = ["1점", "2점", "3점", "4점", "5점"]
rating_sum.plot(kind="barh", ax=axes[0, 0], color=sns.color_palette("YlOrRd", 5))
axes[0, 0].set_title("전체 상품 평점 분포", weight="bold")
axes[0, 0].set_ylabel("평점")
axes[0, 0].set_xlabel("리뷰 수")
axes[0, 0].grid(axis="x", alpha=0.3)

# 2. 리뷰 길이 분포 (상단 가운데 - Histogram)
if "char_length" in df_review.columns:
    axes[0, 1].hist(
        df_review["char_length"], bins=50, color="pink", edgecolor="grey", alpha=0.7
    )
    axes[0, 1].set_title("리뷰 길이 분포", weight="bold")
    axes[0, 1].set_xlim(0, 2000)
    axes[0, 1].set_xlabel("리뷰 길이")
    axes[0, 1].set_ylabel("빈도")

# 3. 리뷰 길이 & Helpful_count (상단 오른쪽 - Scatter with log scale)
axes[0, 2].scatter(
    df_review.get("char_length", [0] * len(df_review)),
    df_review.get("helpful_count", [0] * len(df_review)),
    alpha=0.3,
    s=10,
    color="green",
)
axes[0, 2].set_xscale("log")
axes[0, 2].set_yscale("log")
axes[0, 2].set_title("리뷰 길이 & Helpful_count", weight="bold")
axes[0, 2].set_xlabel("리뷰 길이 (log)")
axes[0, 2].set_ylabel("Helpful_count (log)")

# 4. 평점별 리뷰 길이 (하단 왼쪽 - Violin Plot)
if not df_review.empty and "char_length" in df_review.columns:
    sns.violinplot(
        x="score", y="char_length", data=df_review, palette="Set2", ax=axes[1, 0]
    )
    axes[1, 0].set_title("평점별 리뷰 길이", weight="bold")
    axes[1, 0].set_xlabel("평점")
    axes[1, 0].set_ylabel("리뷰 길이")
    axes[1, 0].set_ylim(0, 1500)

# 5. 평점별 helpful_count (하단 가운데 - Box Plot with log scale)
if not df_review_all.empty:
    sns.boxplot(
        x="score",
        y="helpful_count",
        data=df_review_all,
        palette="Pastel1",
        ax=axes[1, 1],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("평점별 helpful_count", weight="bold")
    axes[1, 1].set_xlabel("평점")
    axes[1, 1].set_ylabel("Helpful_count (log)")

# 6. 상품 평균 평점 & Helpful_count (하단 오른쪽 - Scatter)
if not product_stats.empty:
    axes[1, 2].scatter(
        product_stats["mean_score"],
        product_stats["mean_helpful"],
        s=60,
        alpha=0.7,
        color="skyblue",
        edgecolor="blue",
    )
    axes[1, 2].set_title("상품 평균 평점 & Helpful_count", weight="bold")
    axes[1, 2].set_xlabel("상품 평균 평점")
    axes[1, 2].set_ylabel("평균 Helpful_count")

plt.tight_layout()
plt.show()


# ===== 시각화 2 (추가 차트) =====
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 첫 번째 줄 - 카테고리별 평점 분포, 상품별 리뷰 수 분포
category_ratings = []
for _, row in df_category_summary.iterrows():
    category_ratings.append(
        {
            "category": row["category"],
            "1점": row["rating_1"],
            "2점": row["rating_2"],
            "3점": row["rating_3"],
            "4점": row["rating_4"],
            "5점": row["rating_5"],
        }
    )

df_cat_ratings = pd.DataFrame(category_ratings).set_index("category")
df_cat_ratings.plot(
    kind="barh", stacked=True, ax=axes[0, 0], color=sns.color_palette("YlOrRd", 5)
)
axes[0, 0].set_title("카테고리별 평점 분포", weight="bold")
axes[0, 0].set_ylabel("카테고리")
axes[0, 0].set_xlabel("리뷰 수")
axes[0, 0].grid(axis="x", alpha=0.3)

# 상품별 리뷰 수 분포
axes[0, 1].hist(df_product["total_reviews"], bins=30, color="purple")
axes[0, 1].set_title("상품별 리뷰 수 분포", weight="bold")
axes[0, 1].set_xlabel("리뷰 수")
axes[0, 1].set_ylabel("상품 수")

# 빈 차트 제거
fig.delaxes(axes[0, 2])

# 두 번째 줄 - 카테고리별 텍스트 유무, 카테고리별 평균 평점, 평균 평점 분포
# 1. 카테고리별 텍스트 유무
text_stats = df_category_summary[
    ["category", "text_reviews", "no_text_reviews"]
].set_index("category")
text_stats.plot(kind="bar", stacked=True, ax=axes[1, 0])
axes[1, 0].set_title("카테고리별 텍스트 유무", weight="bold")
axes[1, 0].set_xlabel("카테고리")
axes[1, 0].set_ylabel("리뷰 수")
axes[1, 0].tick_params(axis="x", rotation=15)

# 2. 카테고리별 평균 평점 (텍스트 유무별)
category_avg_ratings = []
for category in df_category_summary["category"]:
    cat_products = df_product[df_product["category"] == category]

    # 텍스트 있는 리뷰의 평균 평점 (가중 평균)
    total_text_reviews = (
        cat_products["total_reviews"] * cat_products["text_review_ratio"] / 100
    )
    weighted_text_rating = (
        (cat_products["avg_rating_with_text"] * total_text_reviews).sum()
        / total_text_reviews.sum()
        if total_text_reviews.sum() > 0
        else 0
    )

    # 텍스트 없는 리뷰의 평균 평점 (가중 평균)
    total_no_text_reviews = (
        cat_products["total_reviews"] * (100 - cat_products["text_review_ratio"]) / 100
    )
    weighted_no_text_rating = (
        (cat_products["avg_rating_without_text"] * total_no_text_reviews).sum()
        / total_no_text_reviews.sum()
        if total_no_text_reviews.sum() > 0
        else 0
    )

    category_avg_ratings.append(
        {
            "category": category,
            "텍스트 있음": weighted_text_rating,
            "텍스트 없음": weighted_no_text_rating,
        }
    )

df_cat_avg = pd.DataFrame(category_avg_ratings).set_index("category")
df_cat_avg.plot(kind="bar", ax=axes[1, 1], color=["blue", "red"], alpha=0.7)
axes[1, 1].set_title("카테고리별 평균 평점 (텍스트 유무별)", weight="bold")
axes[1, 1].set_xlabel("카테고리")
axes[1, 1].set_ylabel("평균 평점")
axes[1, 1].set_ylim([0, 5])
axes[1, 1].tick_params(axis="x", rotation=15)
axes[1, 1].axhline(y=3, color="gray", linestyle="--", alpha=0.5)
axes[1, 1].legend()

# 3. 평균 평점 분포 (텍스트 유무별)
axes[1, 2].hist(
    df_product["avg_rating_with_text"],
    bins=20,
    alpha=0.7,
    label="텍스트 있음",
    color="blue",
)
axes[1, 2].hist(
    df_product["avg_rating_without_text"],
    bins=20,
    alpha=0.7,
    label="텍스트 없음",
    color="red",
)
axes[1, 2].set_title("평균 평점 분포 (텍스트 유무별)", weight="bold")
axes[1, 2].set_xlabel("평균 평점")
axes[1, 2].set_ylabel("상품 수")
axes[1, 2].legend()

plt.tight_layout()
plt.show()


# ===== 시각화 3 (추가 분석) =====
df_product["product_name_short"] = df_product["product_name"].str.slice(
    0, 30
)  # 상품명 30자로 제한

fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 2.3])

rating_cols = ["rating_1", "rating_2", "rating_3", "rating_4", "rating_5"]

# 1. TOP 10 상품 평균 평점
ax1 = fig.add_subplot(gs[0, 0])
df_product["avg_rating"] = (
    df_product["rating_1"] * 1
    + df_product["rating_2"] * 2
    + df_product["rating_3"] * 3
    + df_product["rating_4"] * 4
    + df_product["rating_5"] * 5
) / (df_product[rating_cols].sum(axis=1)).replace(0, pd.NA)

top10 = (
    df_product.dropna(subset=["avg_rating"])
    .sort_values("avg_rating", ascending=False)
    .head(10)
)

top10.set_index("product_name_short")["avg_rating"].plot(
    kind="barh", color="slateblue", ax=ax1
)
ax1.set_title("TOP 10 상품 평균 평점", weight="bold")
ax1.set_xlim(4.5, 5)
ax1.invert_yaxis()

# 2. 상품별 평점 분포 히트맵
ax2 = fig.add_subplot(gs[0, 1])

# 상품명 merge
df_review_all_with_name = df_review_all.merge(
    df_product[["product_id", "product_name_short"]], on="product_id", how="left"
)

rating_dist = (
    df_review_all_with_name.groupby(["product_name_short", "score"])
    .size()
    .unstack(fill_value=0)
)
rating_dist = rating_dist.loc[
    rating_dist.sum(axis=1).sort_values(ascending=False).head(10).index
]
rating_dist.columns = rating_dist.columns.astype(str)

sns.heatmap(
    rating_dist,
    cmap="YlOrRd",
    annot=True,
    fmt=".0f",
    annot_kws={"size": 8},
    linewidths=0.5,
    linecolor="white",
    cbar=True,
    ax=ax2,
)
ax2.set_title("상품별 평점 분포 히트맵", weight="bold")
ax2.set_aspect("auto")
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=10)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=9)

# 3. 월별 평균 판매량 추이
ax3 = fig.add_subplot(gs[1, :])

df_review_all["review_date"] = pd.to_datetime(df_review_all["date"], errors="coerce")
monthly_review_cnt = (
    df_review_all.dropna(subset=["review_date"])
    .set_index("review_date")
    .resample("ME")["id"]
    .count()
)
monthly_review_cnt.plot(ax=ax3, linewidth=2, color="salmon")
ax3.set_title("월별 평균 판매량 추이 (리뷰 수 기반)", weight="bold")
ax3.set_xlabel("월")
ax3.set_ylabel("리뷰 수")

plt.tight_layout()
plt.show()


# ===== 워드클라우드 =====
def normalize_tokens(x):
    # list
    if isinstance(x, list):
        return x
    # numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()
    # 문자열
    if isinstance(x, str):
        # 단어 추출
        tokens = re.findall(r"'([^']+)'", x)
        return tokens
    return []


df_review["tokens"] = df_review["tokens"].apply(normalize_tokens)

df_wc = df_review[
    (df_review["label"].isin([0, 1])) & (df_review["tokens"].apply(len) > 0)
]

pos_tokens = list(chain.from_iterable(df_wc[df_wc["label"] == 1]["tokens"]))
neg_tokens = list(chain.from_iterable(df_wc[df_wc["label"] == 0]["tokens"]))

print(f"\n긍정 리뷰 수: {(df_wc['label'] == 1).sum()}")
print(f"부정 리뷰 수: {(df_wc['label'] == 0).sum()}")
print(f"긍정 토큰 수: {len(pos_tokens)}")
print(f"부정 토큰 수: {len(neg_tokens)}")


def wc_color(palette):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        r, g, b = random.choice(palette)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    return color_func


def wc(tokens, palette):
    if not tokens:
        return None
    text = " ".join(tokens)
    wc_obj = WordCloud(
        font_path=FONT_PATH,
        background_color="white",
        width=800,
        height=600,
        max_words=100,
        color_func=wc_color(palette),
    ).generate(text)
    return wc_obj


wc_pos = wc(pos_tokens, sns.color_palette("OrRd", 10))
wc_neg = wc(neg_tokens, sns.color_palette("cool", 10))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

if wc_pos is not None:
    axes[0].imshow(wc_pos)
axes[0].set_title("긍정 리뷰 워드클라우드", weight="bold")
axes[0].axis("off")

if wc_neg is not None:
    axes[1].imshow(wc_neg)
axes[1].set_title("부정 리뷰 워드클라우드", weight="bold")
axes[1].axis("off")

plt.tight_layout()
plt.show()


print("\n===== EDA 완료 =====")
print(f"총 {len(df_product)}개 상품, {len(df_review_all)}개 리뷰 분석 완료")
