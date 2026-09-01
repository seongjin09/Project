"""
TF-IDF 기본 공식 vs 변형 공식 비교

기본 공식: score = diff (= 긍정 평균 TF - 부정 평균 TF)
변형 공식: score = abs(diff) × log1p(support) × balanced_ratio
"""

import math
import pandas as pd
import glob
from collections import Counter, defaultdict

# ========== 데이터 로드 ==========
print("=" * 80)
print("TF-IDF 기본 공식 vs 변형 공식 비교")
print("=" * 80)

# 리뷰가 많은 단일 상품 선택 (로션 카테고리)
files = glob.glob(
    "data/new_processed_data/partitioned_reviews/category=로션/data.parquet"
)
df = pd.read_parquet(files[0])

# 텍스트 + 라벨 있는 리뷰만
mask = df["full_text"].notna() & (df["full_text"] != "") & df["label"].notna()
df = df[mask].copy()

# 리뷰 수가 가장 많은 상품 선택
product_counts = df["product_id"].value_counts()
target_product = product_counts.index[3]
product_df = df[df["product_id"] == target_product].copy()

total_pos = int((product_df["label"] == 1).sum())
total_neg = int((product_df["label"] == 0).sum())

print(f"\n분석 상품: {target_product}")
print(f"전체 리뷰: {len(product_df)}개 (긍정: {total_pos}개, 부정: {total_neg}개)")
print(f"긍정:부정 비율 = {total_pos/total_neg:.1f}:1 (불균형)")


# ========== TF 계산 (공통) ==========
pos_sum = defaultdict(float)
pos_cnt = defaultdict(int)
neg_sum = defaultdict(float)
neg_cnt = defaultdict(int)

for _, row in product_df.iterrows():
    tokens = row["tokens"]
    if tokens is None or len(tokens) == 0:
        continue
    tokens = list(tokens)
    token_freq = Counter(tokens)
    total_tokens = len(tokens)

    if row["label"] == 1:
        for word, freq in token_freq.items():
            tf = freq / total_tokens
            pos_sum[word] += tf
            pos_cnt[word] += 1
    elif row["label"] == 0:
        for word, freq in token_freq.items():
            tf = freq / total_tokens
            neg_sum[word] += tf
            neg_cnt[word] += 1

MIN_DOC_FREQ = 10

# ========== 방법 1: 기본 TF-IDF diff (단순 차이) ==========
basic_rows = []
for w in set(pos_sum.keys()) | set(neg_sum.keys()):
    pc = pos_cnt.get(w, 0)
    nc = neg_cnt.get(w, 0)
    support = pc + nc
    if support < MIN_DOC_FREQ:
        continue

    pos_mean = (pos_sum[w] / pc) if pc else 0.0
    neg_mean = (neg_sum[w] / nc) if nc else 0.0
    diff = pos_mean - neg_mean

    basic_rows.append(
        {
            "word": w,
            "score": diff,  # 기본: 단순 diff
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "pos_n": pc,
            "neg_n": nc,
            "support": support,
        }
    )

df_basic = pd.DataFrame(basic_rows)

# ========== 방법 2: 변형 공식 (balanced_ratio + log support) ==========
modified_rows = []
for w in set(pos_sum.keys()) | set(neg_sum.keys()):
    pc = pos_cnt.get(w, 0)
    nc = neg_cnt.get(w, 0)
    support = pc + nc
    if support < MIN_DOC_FREQ:
        continue

    # 클래스 불균형 보정
    pos_rate = pc / total_pos if total_pos > 0 else 0
    neg_rate = nc / total_neg if total_neg > 0 else 0
    if (pos_rate + neg_rate) == 0:
        balanced_ratio = 0
    else:
        balanced_ratio = (pos_rate - neg_rate) / (pos_rate + neg_rate)

    pos_mean = (pos_sum[w] / pc) if pc else 0.0
    neg_mean = (neg_sum[w] / nc) if nc else 0.0
    diff = pos_mean - neg_mean

    # 변형 공식
    score = abs(diff) * math.log1p(support) * balanced_ratio

    modified_rows.append(
        {
            "word": w,
            "score": score,
            "diff": diff,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "pos_n": pc,
            "neg_n": nc,
            "support": support,
            "balanced_ratio": balanced_ratio,
        }
    )

df_modified = pd.DataFrame(modified_rows)

# ========== 결과 비교 ==========
TOP_N = 10

print("\n" + "=" * 80)
print("📊 [긍정 키워드 Top 10 비교]")
print("=" * 80)

basic_pos = df_basic.nlargest(TOP_N, "score")
modified_pos = df_modified.nlargest(TOP_N, "score")

print(f"\n{'─'*38} 기본 공식 {'─'*38}")
print(
    f"{'순위':>4} {'키워드':<10} {'score':>10} {'긍정TF':>10} {'부정TF':>10} {'긍정N':>7} {'부정N':>7} {'총N':>6}"
)
print("─" * 86)
for i, (_, r) in enumerate(basic_pos.iterrows(), 1):
    print(
        f"{i:>4} {r['word']:<10} {r['score']:>10.6f} {r['pos_mean']:>10.6f} {r['neg_mean']:>10.6f} {r['pos_n']:>7} {r['neg_n']:>7} {r['support']:>6}"
    )

print(f"\n{'─'*38} 변형 공식 {'─'*38}")
print(
    f"{'순위':>4} {'키워드':<10} {'score':>10} {'diff':>10} {'bal_ratio':>10} {'긍정N':>7} {'부정N':>7} {'총N':>6}"
)
print("─" * 86)
for i, (_, r) in enumerate(modified_pos.iterrows(), 1):
    print(
        f"{i:>4} {r['word']:<10} {r['score']:>10.6f} {r['diff']:>10.6f} {r['balanced_ratio']:>10.4f} {r['pos_n']:>7} {r['neg_n']:>7} {r['support']:>6}"
    )

print("\n" + "=" * 80)
print("📊 [부정 키워드 Top 10 비교]")
print("=" * 80)

basic_neg = df_basic.nsmallest(TOP_N, "score")
modified_neg = df_modified.nsmallest(TOP_N, "score")

print(f"\n{'─'*38} 기본 공식 {'─'*38}")
print(
    f"{'순위':>4} {'키워드':<10} {'score':>10} {'긍정TF':>10} {'부정TF':>10} {'긍정N':>7} {'부정N':>7} {'총N':>6}"
)
print("─" * 86)
for i, (_, r) in enumerate(basic_neg.iterrows(), 1):
    print(
        f"{i:>4} {r['word']:<10} {r['score']:>10.6f} {r['pos_mean']:>10.6f} {r['neg_mean']:>10.6f} {r['pos_n']:>7} {r['neg_n']:>7} {r['support']:>6}"
    )

print(f"\n{'─'*38} 변형 공식 {'─'*38}")
print(
    f"{'순위':>4} {'키워드':<10} {'score':>10} {'diff':>10} {'bal_ratio':>10} {'긍정N':>7} {'부정N':>7} {'총N':>6}"
)
print("─" * 86)
for i, (_, r) in enumerate(modified_neg.iterrows(), 1):
    print(
        f"{i:>4} {r['word']:<10} {r['score']:>10.6f} {r['diff']:>10.6f} {r['balanced_ratio']:>10.4f} {r['pos_n']:>7} {r['neg_n']:>7} {r['support']:>6}"
    )

# ========== 문제점 분석 ==========
print("\n" + "=" * 80)
print("🔍 [기본 공식의 문제점 사례]")
print("=" * 80)

# 기본 공식 긍정 Top10에서 support가 작은 키워드 찾기
print("\n▶ 문제 1: 표본 부족 키워드가 상위에 올라옴")
basic_pos_low_support = df_basic.nlargest(30, "score")
low_support_examples = basic_pos_low_support[basic_pos_low_support["support"] < 30]
if not low_support_examples.empty:
    for _, r in low_support_examples.head(5).iterrows():
        print(
            f"  '{r['word']}': score={r['score']:.6f}, 총 {r['support']}개 리뷰에서만 등장 (신뢰도 낮음)"
        )
else:
    print("  (이 상품에서는 해당 사례 없음)")

print("\n▶ 문제 2: 클래스 불균형 무시")
print(
    f"  긍정 리뷰: {total_pos}개 vs 부정 리뷰: {total_neg}개 (비율 {total_pos/max(total_neg,1):.1f}:1)"
)
print(f"  → 기본 공식은 이 불균형을 고려하지 않음")
print(f"  → 변형 공식은 balanced_ratio로 보정")

# 구체적 사례: 같은 키워드의 두 공식 비교
print(f"\n▶ 문제 3: 구체적 비교 사례")
common_words = set(basic_pos["word"].tolist()) | set(modified_pos["word"].tolist())
for w in list(common_words)[:5]:
    b_row = df_basic[df_basic["word"] == w]
    m_row = df_modified[df_modified["word"] == w]
    if not b_row.empty and not m_row.empty:
        b = b_row.iloc[0]
        m = m_row.iloc[0]
        basic_rank = df_basic.sort_values("score", ascending=False).reset_index()
        basic_rank["rank"] = range(1, len(basic_rank) + 1)
        b_rank = basic_rank[basic_rank["word"] == w]["rank"].values[0]

        mod_rank = df_modified.sort_values("score", ascending=False).reset_index()
        mod_rank["rank"] = range(1, len(mod_rank) + 1)
        m_rank = mod_rank[mod_rank["word"] == w]["rank"].values[0]

        direction = (
            "↑ 상승" if m_rank < b_rank else ("↓ 하락" if m_rank > b_rank else "= 유지")
        )
        print(
            f"  '{w}': 기본={b_rank}위 → 변형={m_rank}위 ({direction}) "
            f"[support={int(b['support'])}, 긍정N={int(b['pos_n'])}, 부정N={int(b['neg_n'])}]"
        )

print("\n" + "=" * 80)
print("✅ 결론")
print("=" * 80)
print(
    """
기본 공식 (score = diff):
  - 단순히 긍정TF - 부정TF 차이만 봄
  - 소수 리뷰에서만 등장한 단어가 과대평가됨
  - 긍정/부정 리뷰 수 불균형을 반영하지 못함

변형 공식 (score = |diff| × log1p(support) × balanced_ratio):
  - log1p(support): 많은 리뷰에서 등장한 키워드일수록 신뢰도 높게 평가
  - balanced_ratio: 긍정/부정 리뷰 수 비율 차이를 보정
  - abs(diff): 부호 반전(음수×음수=양수) 방지
"""
)
