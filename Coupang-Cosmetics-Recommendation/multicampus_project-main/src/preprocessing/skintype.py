import json
from collections import Counter
from typing import Dict, List, Any, Optional

TYPE_KEYWORDS = {
    "건성": ["건성"],
    "지성": ["지성", "지성인"],
    "복합성": ["복합", "복합성"],
    "민감성": ["민감", "민감성"],
    "여드름성": ["여드름", "여드름성"],
}

def _count_from_tokens(tokens: List[str], type_keywords: Dict[str, List[str]]) -> Dict[str, int]:
    token_counter = Counter(tokens)
    return {t: sum(token_counter.get(kw, 0) for kw in kws) for t, kws in type_keywords.items()}

def _count_from_text(text: str, type_keywords: Dict[str, List[str]]) -> Dict[str, int]:
    return {t: sum(text.count(kw) for kw in kws) for t, kws in type_keywords.items()}

def _pick_skin_type_from_counts(counts: Dict[str, int]) -> str:
    """리뷰 기반 최다 스킨타입 1개(동점이면 혼합, 전부 0이면 미분류)"""
    if not counts:
        return "미분류"
    max_count = max(counts.values())
    if max_count <= 0:
        return "미분류"
    top_types = [t for t, v in counts.items() if v == max_count]
    if len(top_types) == 1:
        return top_types[0]
    return "복합/혼합(" + ",".join(sorted(top_types)) + ")"

def _find_skin_type_in_product_name(product_name: str, type_keywords: Dict[str, List[str]]) -> Optional[str]:
    """
    상품명에 키워드가 들어있으면 해당 타입 반환
    - 여러 타입이 동시에 매칭되면 혼합으로 반환
    - 매칭 없으면 None
    """
    if not product_name:
        return None

    matched_types = []
    for t, kws in type_keywords.items():
        for kw in kws:
            if kw and kw in product_name:
                matched_types.append(t)
                break

    matched_types = sorted(set(matched_types))
    if not matched_types:
        return None
    if len(matched_types) == 1:
        return matched_types[0]
    return "복합/혼합(" + ",".join(matched_types) + ")"

def classify_product(product_obj: Dict[str, Any]) -> Dict[str, Any]:
    pinfo = product_obj.get("product_info") or {}
    product_name = (pinfo.get("product_name_clean") or pinfo.get("product_name") or "").strip()

    # ✅ 1) 상품명 우선 룰
    name_based = _find_skin_type_in_product_name(product_name, TYPE_KEYWORDS)
    if name_based:
        skin_type = name_based
    else:
        # ✅ 2) 리뷰 기반 룰
        reviews = (product_obj.get("reviews") or {}).get("data") or []
        total_counts = Counter({t: 0 for t in TYPE_KEYWORDS})

        for r in reviews:
            tokens = r.get("tokens")
            if isinstance(tokens, list) and tokens:
                c = _count_from_tokens(tokens, TYPE_KEYWORDS)
            else:
                text = r.get("full_text") or r.get("content") or ""
                c = _count_from_text(text, TYPE_KEYWORDS)
            total_counts.update(c)

        skin_type = _pick_skin_type_from_counts(dict(total_counts))

    category = (pinfo.get("category_norm") or pinfo.get("category_path") or "").strip()

    return {
        "product_name": product_name,
        "category": category,
        "skin_type": skin_type,
    }

def make_product_skin_type_json(input_json_path: str, output_json_path: str):
    with open(input_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    products = raw.get("data") or []
    results = [classify_product(p) for p in products]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"저장 완료 → {output_json_path}")

if __name__ == "__main__":
    in_path = r"C:\Users\user\Downloads\processed_클렌징 폼_with_text.json"  # 🔥 여기에 네 파일 경로로 변경
    out_path = "product_skin_type.json"
    make_product_skin_type_json(in_path, out_path)
