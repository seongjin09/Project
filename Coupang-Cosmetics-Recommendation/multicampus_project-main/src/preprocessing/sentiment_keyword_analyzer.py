import json
import math
import pandas as pd
from collections import Counter, defaultdict

# =========================
# 0) 설정
# =========================
JSON_PATH = "파일경로"

TOP_N_FREQ = 20  # 빈도 기반 분석에서 상위 몇 개 단어를 볼 것인지
TOP_N_SENTIMENT = 30  # 감성 특화 키워드를 상위 몇 개까지 뽑을 것인지
MIN_DOC_FREQ = 20  # 너무 적은 리뷰에서만 등장한 단어를 분석 대상에서 제외

OUTPUT_JSON = "저장할 파일 이름"

CATEGORY_MODE = "leaf"

SKIN_TYPES = {
    "건성": ["건성"],
    "지성": ["지성", "지성인"],
    "복합성": ["복합", "복합성"],
    "민감성": ["민감", "민감성"],
    "여드름성": ["여드름", "여드름성"],
}


# =========================
# 1) 유틸 함수
# =========================
def detect_skin_types(tokens):
    token_set = set(tokens)
    found = []
    for skin, keys in SKIN_TYPES.items():
        if any(k in token_set for k in keys):
            found.append(skin)
    return found


def normalize_tfidf(tfidf):
    if isinstance(tfidf, dict):
        return {str(k): float(v) for k, v in tfidf.items()}
    if isinstance(tfidf, list):
        out = {}
        for item in tfidf:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                w, s = item
                out[str(w)] = float(s)
        return out
    return {}


def category_from_path(category_path: str) -> str:
    if not isinstance(category_path, str) or not category_path.strip():
        return "UNKNOWN_CATEGORY"
    if CATEGORY_MODE == "path":
        return category_path.strip()
    parts = [p.strip() for p in category_path.split(">")]
    return parts[-1] if parts else category_path.strip()


def make_product_key(product_id, product_name):
    pid = (
        "UNKNOWN_ID"
        if product_id is None or (isinstance(product_id, float) and pd.isna(product_id))
        else str(product_id)
    )
    pname = (
        "UNKNOWN_NAME"
        if not isinstance(product_name, str) or not product_name.strip()
        else product_name.strip()
    )
    return f"{pid}__{pname}"


def load_and_flatten(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for product in raw.get("data", []):
        pinfo = product.get("product_info", {})
        category_path = pinfo.get("category_path", "")
        category = category_from_path(category_path)

        product_id = pinfo.get("product_id")
        product_name = pinfo.get("product_name_clean", pinfo.get("product_name"))
        brand = pinfo.get("brand")

        for r in product.get("reviews", {}).get("data", []):
            tokens = r.get("tokens", [])
            if not isinstance(tokens, list) or len(tokens) == 0:
                continue

            rows.append(
                {
                    "product_id": product_id,
                    "product_name": product_name,
                    "product_key": make_product_key(product_id, product_name),
                    "brand": brand,
                    "category": category,
                    "category_path": category_path,
                    "label": r.get("label"),
                    "score": r.get("score"),
                    "date": r.get("date"),
                    "tokens": tokens,
                    "tfidf": r.get("tfidf", None),
                }
            )

    return pd.DataFrame(rows)


# =========================
# 2) 빈도 분석 (피부 타입별)
# =========================
def top_words_by_skin(df_reviews, top_n=30):
    df = df_reviews.copy()
    df["skin_types"] = df["tokens"].apply(detect_skin_types)

    df_skin = (
        df[df["skin_types"].map(len) > 0]
        .explode("skin_types")
        .rename(columns={"skin_types": "skin_type"})
    )

    result = {}
    for skin, part in df_skin.groupby("skin_type"):
        all_tokens = []
        for toks in part["tokens"].tolist():
            all_tokens.extend(toks)
        result[skin] = Counter(all_tokens).most_common(top_n)

    return df_skin, result


# =========================
# 3) 감성 특화 키워드 (대안 1: 가중 diff)
# =========================
def sentiment_tfidf_diff(df_reviews, top_n=50, min_doc_freq=5):
    """
    (긍정 평균 TF-IDF) - (부정 평균 TF-IDF) 를 구한 뒤,
    support(=pos_n+neg_n)로 가중치를 줘서(로그) 표본이 작은 단어가 과도하게 뜨는 현상을 완화.

    score = diff * log1p(pos_n + neg_n)

    - score > 0 : 긍정 특화(강도 + 신뢰도 반영)
    - score < 0 : 부정 특화(강도 + 신뢰도 반영)
    """
    empty_cols = [
        "word",
        "diff",
        "pos_tfidf_mean",
        "neg_tfidf_mean",
        "pos_doc_count",
        "neg_doc_count",
        "support",
        "score",
    ]

    df = df_reviews.copy()
    df = df[df["label"].isin([0, 1])].copy()
    df = df[df["tfidf"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    pos_sum = defaultdict(float)
    pos_cnt = defaultdict(int)
    neg_sum = defaultdict(float)
    neg_cnt = defaultdict(int)

    for _, row in df.iterrows():
        tfidf_dict = normalize_tfidf(row["tfidf"])
        if not tfidf_dict:
            continue

        if row["label"] == 1:
            for w, s in tfidf_dict.items():
                pos_sum[w] += float(s)
                pos_cnt[w] += 1
        else:
            for w, s in tfidf_dict.items():
                neg_sum[w] += float(s)
                neg_cnt[w] += 1

    if not pos_sum and not neg_sum:
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    # 전체 긍정/부정 리뷰 수 계산 (클래스 불균형 보정용)
    total_pos = int((df["label"] == 1).sum())
    total_neg = int((df["label"] == 0).sum())

    rows = []
    for w in set(pos_sum.keys()) | set(neg_sum.keys()):
        pc = pos_cnt.get(w, 0)
        nc = neg_cnt.get(w, 0)

        # 전체 support가 최소 빈도를 만족해야 함
        support = pc + nc
        if support < min_doc_freq:
            continue

        # 1. 문서 출현 비율 계산 (클래스 불균형 보정)
        pos_rate = pc / total_pos if total_pos > 0 else 0
        neg_rate = nc / total_neg if total_neg > 0 else 0

        # 2. 균형 잡힌 비율 (Balanced Ratio)
        # 긍정에서 더 잘 나오면 양수, 부정에서 더 잘 나오면 음수
        if (pos_rate + neg_rate) == 0:
            balanced_ratio = 0
        else:
            balanced_ratio = (pos_rate - neg_rate) / (pos_rate + neg_rate)

        # 3. 평균 TF-IDF 차이
        pos_mean = (pos_sum[w] / pc) if pc else 0.0
        neg_mean = (neg_sum[w] / nc) if nc else 0.0
        diff = pos_mean - neg_mean

        # 4. 최종 점수: abs(diff)로 강도만 반영, balanced_ratio가 방향성 결정
        # 이렇게 하면 음수 × 음수 = 양수가 되는 부호 반전 현상 방지
        score = abs(diff) * math.log1p(support) * balanced_ratio

        rows.append(
            {
                "word": w,
                "diff": diff,
                "pos_tfidf_mean": pos_mean,
                "neg_tfidf_mean": neg_mean,
                "pos_doc_count": pc,
                "neg_doc_count": nc,
                "support": support,
                "balanced_ratio": balanced_ratio,
                "score": score,
            }
        )

    if not rows:
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    df_diff = pd.DataFrame(rows, columns=empty_cols)

    # ✅ 정렬 기준을 diff가 아니라 score로 변경(표본 반영)
    pos_special = df_diff.sort_values("score", ascending=False).head(top_n)
    neg_special = df_diff.sort_values("score", ascending=True).head(top_n)
    return pos_special, neg_special


def df_to_diff_list(df_part):
    return [
        {
            "word": row["word"],
            "diff": float(row["diff"]),
            "pos": float(row["pos_tfidf_mean"]),
            "neg": float(row["neg_tfidf_mean"]),
            "pos_n": int(row["pos_doc_count"]),
            "neg_n": int(row["neg_doc_count"]),
            "support": int(
                row.get("support", row["pos_doc_count"] + row["neg_doc_count"])
            ),
            "balanced_ratio": float(row.get("balanced_ratio", 0.0)),
            "score": float(row.get("score", row["diff"])),
        }
        for _, row in df_part.iterrows()
    ]


def print_diff_block(title, df_pos, df_neg, max_print=10):
    print(f"\n{title}")

    if df_pos.empty and df_neg.empty:
        print(
            "⚠️ 키워드를 만들 수 없습니다. (tfidf 없음 / label 한쪽만 있음 / MIN_DOC_FREQ로 모두 제외 등)"
        )
        return

    print("✅ 긍정 특화 키워드 (score 큰 순)")
    for _, r in df_pos.head(max_print).iterrows():
        # 기존 요청 포맷 + score/support 같이 보여줌(원치 않으면 score/support 부분 삭제)
        print(
            f"{r['word']}\t"
            f"diff={r['diff']:.6f} pos={r['pos_tfidf_mean']:.6f} neg={r['neg_tfidf_mean']:.6f} "
            f"(pos_n={int(r['pos_doc_count'])}, neg_n={int(r['neg_doc_count'])}) "
            f"support={int(r['support'])} score={r['score']:.6f}"
        )

    print("❌ 부정 특화 키워드 (score 작은 순)")
    for _, r in df_neg.head(max_print).iterrows():
        print(
            f"{r['word']}\t"
            f"diff={r['diff']:.6f} pos={r['pos_tfidf_mean']:.6f} neg={r['neg_tfidf_mean']:.6f} "
            f"(pos_n={int(r['pos_doc_count'])}, neg_n={int(r['neg_doc_count'])}) "
            f"support={int(r['support'])} score={r['score']:.6f}"
        )


# =========================
# 4) 메인
# =========================
def main():
    df_reviews = load_and_flatten(JSON_PATH)

    if len(df_reviews) == 0:
        print("리뷰 데이터가 비어있습니다. JSON_PATH 또는 JSON 구조를 확인하세요.")
        return

    # (A) 피부 타입별 빈도 분석
    df_skin, skin_top = top_words_by_skin(df_reviews, top_n=TOP_N_FREQ)

    print("====================================")
    print(f"총 리뷰 수: {len(df_reviews)}")
    print(f"피부타입 언급 리뷰 수(중복 포함, explode 후): {len(df_skin)}")
    print("====================================")
    for skin, pairs in skin_top.items():
        print(f"\n[{skin}] TOP {TOP_N_FREQ}")
        for w, c in pairs:
            print(f"{w}\t{c}")

    # (B) 전체 감성 특화 키워드(가중 diff)
    pos_all, neg_all = sentiment_tfidf_diff(
        df_reviews, top_n=TOP_N_SENTIMENT, min_doc_freq=MIN_DOC_FREQ
    )
    print_diff_block(
        "[전체 리뷰 집합 - 감성 특화 키워드(가중 diff)]", pos_all, neg_all, max_print=10
    )

    # (C) 카테고리별
    category_results = {}
    for cat, part in df_reviews.groupby("category"):
        pos_cat, neg_cat = sentiment_tfidf_diff(
            part, top_n=TOP_N_SENTIMENT, min_doc_freq=MIN_DOC_FREQ
        )
        category_results[str(cat)] = {
            "positive_special": df_to_diff_list(pos_cat),
            "negative_special": df_to_diff_list(neg_cat),
            "review_count": int(len(part)),
            "pos_count": int((part["label"] == 1).sum()),
            "neg_count": int((part["label"] == 0).sum()),
            "tfidf_notna_count": int(part["tfidf"].notna().sum()),
        }

    # (D) 상품별
    product_results = {}
    for pkey, part in df_reviews.groupby("product_key"):
        pid, pname = (pkey.split("__", 1) + [""])[:2]

        pos_p, neg_p = sentiment_tfidf_diff(
            part, top_n=TOP_N_SENTIMENT, min_doc_freq=MIN_DOC_FREQ
        )

        print_diff_block(f"[상품] {pid} | {pname}", pos_p, neg_p, max_print=5)

        product_results[str(pkey)] = {
            "product_id": pid,
            "product_name": pname,
            "positive_special": df_to_diff_list(pos_p),
            "negative_special": df_to_diff_list(neg_p),
            "review_count": int(len(part)),
            "pos_count": int((part["label"] == 1).sum()),
            "neg_count": int((part["label"] == 0).sum()),
            "tfidf_notna_count": int(part["tfidf"].notna().sum()),
        }

    # 저장
    output = {
        "meta": {
            "top_n_freq": TOP_N_FREQ,
            "top_n_sentiment": TOP_N_SENTIMENT,
            "min_doc_freq": MIN_DOC_FREQ,
            "category_mode": CATEGORY_MODE,
            "sentiment_scoring": "score = diff * log1p(pos_n + neg_n)",
        },
        "results": {
            "skin_type_word_frequency": {
                "top_n": TOP_N_FREQ,
                "skin_types": {
                    skin: [{"word": w, "count": c} for w, c in pairs]
                    for skin, pairs in skin_top.items()
                },
            },
            "overall_sentiment_special_words_weighted_diff": {
                "positive_special": df_to_diff_list(pos_all),
                "negative_special": df_to_diff_list(neg_all),
                "review_count": int(len(df_reviews)),
                "pos_count": int((df_reviews["label"] == 1).sum()),
                "neg_count": int((df_reviews["label"] == 0).sum()),
                "tfidf_notna_count": int(df_reviews["tfidf"].notna().sum()),
            },
            "category_sentiment_special_words_weighted_diff": category_results,
            "product_sentiment_special_words_weighted_diff": product_results,
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[저장 완료] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
