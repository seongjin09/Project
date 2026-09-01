import json
import re
from typing import Dict, List
from collections import Counter


# =========================
# 기본 유틸리티 함수
# =========================
def to_lower(text):
    """텍스트를 소문자로 변환하고 앞뒤 공백을 제거함"""
    if not isinstance(text, str):
        return text
    return text.lower().strip()


# =========================
# 브랜드 이름 표준화 (통합)
# =========================
# 한글/영문으로 혼용되는 브랜드 이름을 하나의 대표 영문명으로 통일함
BRAND_MAPPING = {
    "토코보": "tocobo",
    "tocobo": "tocobo",
    "라네즈": "laneige",
    "laneige": "laneige",
    "이니스프리": "innisfree",
    "innisfree": "innisfree",
}


def normalize_brand(brand: str) -> str:
    """브랜드 이름을 소문자로 만들고 매핑 테이블에 따라 표준화함"""
    brand = to_lower(brand)
    return BRAND_MAPPING.get(brand, brand)


# =========================
# 카테고리 이름 표준화 (통합)
# =========================
# 세부 카테고리 명칭들을 대표 카테고리 명칭으로 그룹화함
CATEGORY_GROUPS = {
    "스킨": ["스킨", "토너", "skin", "toner"],
    "로션": ["로션", "lotion", "emulsion", "에멀전"],
    "에센스": ["에센스", "essence"],
    "세럼": ["세럼", "serum"],
    "크림": ["크림", "cream"],
    "선스틱": ["선스틱", "sun stick", "sunstick"],
    "선크림": ["선크림", "썬크림", "sun cream", "sunscreen"],
    "클렌저": ["클렌저", "cleanser"],
}

# 그룹을 매핑 딕셔너리로 변환
CATEGORY_MAPPING = {
    variant: standard
    for standard, variants in CATEGORY_GROUPS.items()
    for variant in variants
}


def normalize_category(category_path: str) -> str:
    """카테고리 경로에서 마지막 잎 노드(세부 분류)를 추출하여 표준화함"""
    if not isinstance(category_path, str):
        return category_path
    leaf = category_path.split(">")[-1].strip().lower()
    return CATEGORY_MAPPING.get(leaf, leaf)


# =========================
# 동의어 및 정규화 처리
# =========================
# 영문/국문 혼용이나 오타, 유사 의미를 가진 단어들을 정규표현식으로 통합함
SYNONYM_PATTERNS = {
    r"machine\s*learning|머신\s*learning|머신\s*러닝|ml": "머신러닝",
    r"skin\s*care|스킨\s*케어": "스킨케어",
    r"skin|toner|토너": "스킨",
    r"lotion|emulsion|에멀전": "로션",
    r"essence|에센스": "에센스",
    r"serum|세럼": "세럼",
    r"cream|크림": "크림",
    r"sun\s*stick|sunstick|선\s*스틱": "선스틱",
    r"sun\s*cream|sunscreen|썬\s*크림|선\s*크림": "선크림",
    r"cleanser|cleansing\s*foam|폼\s*클렌저": "클렌저",
}


def normalize_synonyms(text: str) -> str:
    """정규표현식 패턴을 순회하며 동의어를 표준어로 치환함"""
    if not isinstance(text, str):
        return text
    text = text.lower()
    for pattern, replacement in SYNONYM_PATTERNS.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


# =========================
# 브랜드 패턴 생성 및 상품명 내 브랜드 제거
# =========================
def build_brand_patterns(brands):
    """각 브랜드명에 대해 공백이 섞인 경우까지 탐지할 수 있는 정규표현식 패턴을 생성함"""
    patterns = {}
    for b in brands:
        if not b:
            continue
        # 글자 사이에 공백이 있는 경우도 매칭 (ex: 토 코 보)
        spaced = r"\s*".join(list(b))
        patterns[b] = rf"{b}|{spaced}"
    return patterns


def remove_brand_from_name(name: str, brand: str, brand_patterns: dict) -> str:
    """상품명에 중복으로 포함된 브랜드 이름을 삭제하여 상품 특징만 남김"""
    if not isinstance(name, str) or not isinstance(brand, str):
        return name
    pattern = brand_patterns.get(brand)
    if not pattern:
        return name
    name = re.sub(pattern, " ", name.lower())
    return re.sub(r"\s+", " ", name).strip()


# =========================
# 상품명 노이즈 제거
# =========================
# 용량, 수량, 차단 지수, 특수 기호 등 핵심 키워드 분석에 방해가 되는 정보 정의
NOISE_PATTERNS = [
    r"\[.*?\]",  # 대괄호 및 그 내용
    r"\d+\s?(ml|g)",  # 용량 (ex: 50ml, 100 g)
    r"\d+\s?개(입)?",  # 수량 (ex: 3개입)
    r"spf\s?\d+\+?",  # 자외선 차단 지수
    r"pa\+{1,4}",  # PA 등급
    r"[★☆■◆▶️,]",  # 주요 특수 기호 및 쉼표
]


def clean_product_name(name: str, brand_normal: str, brand_patterns: dict) -> str:
    """소문자 변환, 노이즈 패턴 제거, 브랜드명 삭제, 동의어 처리를 순차적으로 수행함"""
    name = to_lower(name)
    for pattern in NOISE_PATTERNS:
        name = re.sub(pattern, " ", name)
    name = remove_brand_from_name(name, brand_normal, brand_patterns)
    name = normalize_synonyms(name)
    return re.sub(r"\s+", " ", name).strip()


# =========================
# 토큰화 및 후처리
# =========================
def tokenize(text: str) -> List[str]:
    """공백을 기준으로 텍스트를 단어 단위(토큰)로 분리함"""
    if not isinstance(text, str):
        return []
    return text.split()


# 토큰 단위에서 발생하는 중복/유사 의미 처리 및 무의미한 단어 제거
TOKEN_MAPPING = {
    "썬스틱": "선스틱",
    "스틱": "선스틱",
    "유브이": "선케어",
    "선": "선케어",
}

DROP_TOKENS = {
    "퍼펙션",
    "프레쉬",
    "마스터즈",
    "내추럴",
}


def normalize_tokens(tokens):
    """불용어를 제거하고 매핑 테이블에 따라 각 토큰을 표준화함"""
    normalized = []
    for t in tokens:
        if t in DROP_TOKENS:
            continue
        t = TOKEN_MAPPING.get(t, t)
        normalized.append(t)
    return normalized


# =========================
# 개별 상품 정보 전처리 실행
# =========================
def preprocess_product_info(product_info: Dict, brand_patterns: dict) -> Dict:
    """상품 정보 딕셔너리를 받아 브랜드, 카테고리, 상품명을 정제하고 토큰을 생성함"""
    product_info = product_info.copy()

    # 브랜드 표준화 (원본 필드에 덮어씌움)
    brand_normal = normalize_brand(product_info.get("brand"))
    product_info["brand"] = brand_normal
    product_info["category_normal"] = normalize_category(
        product_info.get("category_path")
    )

    # 상품명 정제 및 토큰 추출
    product_info["product_name_clean"] = clean_product_name(
        product_info.get("product_name"), brand_normal, brand_patterns
    )
    tokens = tokenize(product_info["product_name_clean"])
    product_info["product_tokens"] = normalize_tokens(tokens)

    return product_info


# =========================
# 전체 JSON 데이터 구조 전처리
# =========================
def brand_standardizer(raw_json: Dict) -> Dict:
    """JSON 데이터 전체를 순회하며 모든 상품에 대해 전처리를 적용함"""
    # 현재 데이터 세트에 존재하는 모든 브랜드 목록 추출 및 패턴 구축
    brands = {
        normalize_brand(item.get("product_info", {}).get("brand"))
        for item in raw_json.get("data", [])
    }
    brand_patterns = build_brand_patterns(brands)

    # 각 상품 아이템의 product_info를 정제된 버전으로 업데이트
    for item in raw_json.get("data", []):
        if "product_info" in item:
            item["product_info"] = preprocess_product_info(
                item["product_info"], brand_patterns
            )
    return raw_json


# =========================
# 결과 분석용 함수
# =========================
def analyze_tokens(processed_json: Dict, top_n: int = 30):
    """전체 데이터에서 가장 빈번하게 등장하는 단어 상위 N개를 집계함"""
    counter = Counter()
    for item in processed_json.get("data", []):
        counter.update(item.get("product_info", {}).get("product_tokens", []))
    return counter.most_common(top_n)


# =========================
# 프로그램 실행 메인 루틴
# =========================
if __name__ == "__main__":
    # 로컬 경로의 JSON 파일 로드
    with open("result_오일.json", "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    # 전처리 파이프라인 실행
    processed_json = brand_standardizer(raw_json)

    # 정제된 데이터를 새로운 파일로 저장
    with open("cleaned_data.json", "w", encoding="utf-8") as f:
        json.dump(processed_json, f, ensure_ascii=False, indent=2)

    print("전처리 전체 파이프라인 완료\n")

    # 빈도 분석 결과 출력
    print("상위 토큰 TOP 30:")
    for token, cnt in analyze_tokens(processed_json, 30):
        print(token, cnt)
