import json
import copy
from collections import Counter


def drop_missing_val_splitter(data: dict) -> tuple:
    """
    데이터를 받아서 결측치를 제거하고 텍스트 유무에 따라 분할합니다.

    Args:
        data: 전처리할 JSON 데이터

    Returns:
        (with_text, without_text): 텍스트가 있는 데이터와 없는 데이터 튜플
    """
    # 용량 최적화를 위해 삭제할 기본값 필드 정의
    DROP_0 = {"helpful_count"}
    DROP_FALSE = {"has_image"}

    def has_text(review: dict) -> bool:
        """리뷰 객체 내에 실제 본문 내용이 존재하는지 확인"""
        for k in ["content", "full_text"]:
            v = review.get(k)
            if isinstance(v, str) and v.strip():
                return True
        return False

    def drop_missing_fields(obj: dict):
        """결측값이거나 의미 없는 기본값인 경우 해당 key를 삭제하여 용량 최적화"""
        for k in list(obj.keys()):
            v = obj[k]

            # 1. 값이 None이거나 빈 문자열인 경우 삭제
            if v is None or (isinstance(v, str) and v.strip() == ""):
                del obj[k]
                continue

            # 2. 도움수(helpful_count)가 0인 경우 삭제
            if k in DROP_0 and v == 0:
                del obj[k]
                continue

            # 3. 이미지 여부(has_image)가 False인 경우 삭제
            if k in DROP_FALSE and v is False:
                del obj[k]
                continue

    def init_metadata(data: dict):
        """결과 데이터 객체의 통계 메타데이터 초기화"""
        data["total_collected_reviews"] = 0
        data["total_text_reviews"] = 0
        data["total_product"] = 0
        data["total_rating_distribution"] = {}

    # 원본 구조 유지를 위해 전체 데이터 복사 (Deep Copy)
    with_text = copy.deepcopy(data)
    without_text = copy.deepcopy(data)

    init_metadata(with_text)
    init_metadata(without_text)

    if "search_name" in data:
        with_text["search_name"] = data["search_name"]
        without_text["search_name"] = data["search_name"]

    # 분리된 데이터별로 별점 분포를 다시 계산하기 위한 카운터
    rating_with = Counter()
    rating_without = Counter()

    data_with = []
    data_without = []

    # 전체 상품 리스트 순회
    for product in data["data"]:
        # 상품 정보 영역 결측치 정제
        drop_missing_fields(product["product_info"])

        reviews = product["reviews"]["data"]

        # 리뷰 분리를 위해 상품 객체 틀 복사
        p_with = copy.deepcopy(product)
        p_without = copy.deepcopy(product)

        reviews_with = []
        reviews_without = []

        # 개별 리뷰 순회 및 분류
        for r in reviews:
            # 리뷰 데이터 영역 결측치 및 기본값 정제
            drop_missing_fields(r)

            if has_text(r):
                # 텍스트가 있는 리뷰 리스트에 추가
                reviews_with.append(r)
                rating_with[str(r.get("score"))] += 1
            else:
                # 텍스트가 없는 리뷰 리스트에 추가
                reviews_without.append(r)
                rating_without[str(r.get("score"))] += 1

        # 텍스트 리뷰가 포함된 상품의 경우 메타데이터 갱신 및 리스트 추가
        if reviews_with:
            p_with["reviews"]["data"] = reviews_with
            p_with["reviews"]["total_count"] = len(reviews_with)
            p_with["reviews"]["text_count"] = len(reviews_with)

            data_with.append(p_with)
            with_text["total_product"] += 1
            with_text["total_collected_reviews"] += len(reviews_with)
            with_text["total_text_reviews"] += len(reviews_with)

        # 텍스트가 없는 리뷰가 포함된 상품의 경우 메타데이터 갱신 및 리스트 추가
        if reviews_without:
            p_without["reviews"]["data"] = reviews_without
            p_without["reviews"]["total_count"] = len(reviews_without)
            p_without["reviews"]["text_count"] = 0

            data_without.append(p_without)
            without_text["total_product"] += 1
            without_text["total_collected_reviews"] += len(reviews_without)

    # 최종 분리된 데이터셋에 집계된 결과 반영
    with_text["data"] = data_with
    with_text["total_rating_distribution"] = dict(rating_with)

    without_text["data"] = data_without
    without_text["total_rating_distribution"] = dict(rating_without)
    without_text["total_text_reviews"] = 0

    return with_text, without_text


if __name__ == "__main__":
    # 파일 경로 설정
    INPUT_PATH = "result_아이라이너.json"
    OUTPUT_WITH_TEXT = "with_text_drop_missing.json"
    OUTPUT_WITHOUT_TEXT = "without_text_drop_missing.json"

    try:
        # 원본 데이터 로드
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 전처리 및 분리 로직 실행
        with_text, without_text = drop_missing_val_splitter(data)

        # 텍스트가 있는 정제 데이터 저장
        with open(OUTPUT_WITH_TEXT, "w", encoding="utf-8") as f:
            json.dump(with_text, f, ensure_ascii=False, indent=2)

        # 텍스트가 없는 정제 데이터 저장
        with open(OUTPUT_WITHOUT_TEXT, "w", encoding="utf-8") as f:
            json.dump(without_text, f, ensure_ascii=False, indent=2)

        print("데이터 전처리 및 분리 작업이 완료되었습니다.")
        print(f"- 텍스트 포함 리뷰 파일: {OUTPUT_WITH_TEXT}")
        print(f"- 텍스트 미포함 리뷰 파일: {OUTPUT_WITHOUT_TEXT}")

    except FileNotFoundError:
        print(f"오류: {INPUT_PATH} 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
