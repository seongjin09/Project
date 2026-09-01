import json
import re
import os
import unicodedata
from datetime import datetime


def normalize_text(text):
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s.,!?~❤️]", "", text)
    text = re.sub(r"([ㄱ-ㅎㅏ-ㅣ])\1+", r"\1\1", text)
    text = re.sub(r"([ㄱ-ㅎㅏ-ㅣ]{2})\1+", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_format(input_filename):
    if not os.path.exists(input_filename):
        print(f"파일을 찾을 수 없습니다: {input_filename}")
        return

    with open(input_filename, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    cleaned_products = []

    for product in json_data.get("data", []):
        seen_reviews = set()  # 상품별로 (날짜, 닉네임, 내용) 중복 체크용

        # 1. 상품 정보 변환
        info = product.get("product_info", {})

        # product_id 정수형 변환
        product_id_raw = str(info.get("product_id", "0")).replace(",", "")
        info["product_id"] = int(product_id_raw) if product_id_raw.isdigit() else 0

        # 가격 정수형 변환 (쉼표 제거 후 정수형)
        price_raw = str(info.get("price", "0")).replace(",", "")
        info["price"] = int(price_raw) if price_raw.isdigit() else 0

        # 총 리뷰 수 정수형 변환
        rev_count_raw = str(info.get("total_reviews", "0")).replace(",", "")
        info["total_reviews"] = int(rev_count_raw) if rev_count_raw.isdigit() else 0

        # rating_distribution 정수형 변환
        rating_dist = info.get("rating_distribution", {})
        for key in ["5", "4", "3", "2", "1"]:
            if key in rating_dist:
                rating_dist[key] = (
                    int(rating_dist[key])
                    if isinstance(rating_dist[key], (int, float, str))
                    and str(rating_dist[key]).replace(",", "").isdigit()
                    else 0
                )
        info["rating_distribution"] = rating_dist

        # 2. 리뷰 데이터 변환 및 정제
        reviews_container = product.get("reviews", {})
        original_reviews = reviews_container.get("data", [])
        cleaned_reviews = []

        for review in original_reviews:
            # 날짜 변환 (datetime 객체 변환 후 덮어씌움)
            date_str = review.get("date", "").strip(".")
            try:
                # 2025.12.12. 형태 파싱
                dt_obj = datetime.strptime(date_str, "%Y.%m.%d")
                review["date"] = dt_obj.strftime(
                    "%Y-%m-%d"
                )  # ISO 형식 문자열로 덮어씌움
            except:
                review["date"] = date_str

            # collected_at 변환 (시분초 포함, datetime 형식으로 덮어씌움)
            collected_str = review.get("collected_at", "")
            try:
                # 2025.12.20 03:32:03 형태 파싱
                dt_obj = datetime.strptime(collected_str, "%Y.%m.%d %H:%M:%S")
                review["collected_at"] = dt_obj.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )  # ISO 형식으로 덮어씌움
            except:
                review["collected_at"] = collected_str

            # 별점 정수형 변환
            review["score"] = int(review.get("score", 0))

            # id 정수형 변환
            review["id"] = int(review.get("id", 0))

            # helpful_count 정수형 변환
            review["helpful_count"] = int(review.get("helpful_count", 0))

            # 텍스트 정규화 (이모지 제거 및 자모음 반복 축소)
            review["title"] = normalize_text(review.get("title", ""))
            review["content"] = normalize_text(review.get("content", ""))
            review["full_text"] = normalize_text(review.get("full_text", ""))

            # 중복 검사 (날짜, 닉네임, 전체 텍스트 기준)
            review_fingerprint = (
                review.get("date"),
                review.get("nickname"),
                review.get("full_text"),
            )
            if review_fingerprint not in seen_reviews:
                seen_reviews.add(review_fingerprint)
                cleaned_reviews.append(review)

        # 업데이트된 리뷰 리스트 저장
        reviews_container["data"] = cleaned_reviews
        reviews_container["total_count"] = len(cleaned_reviews)
        reviews_container["text_count"] = sum(
            1 for r in cleaned_reviews if r.get("content")
        )

        product["product_info"] = info
        product["reviews"] = reviews_container
        cleaned_products.append(product)

    # 전체 통계 업데이트
    json_data["data"] = cleaned_products
    json_data["total_product"] = len(cleaned_products)
    json_data["total_collected_reviews"] = sum(
        p["reviews"]["total_count"] for p in cleaned_products
    )
    json_data["total_text_reviews"] = sum(
        p["reviews"]["text_count"] for p in cleaned_products
    )

    # total_rating_distribution 정수형 변환
    total_rating = json_data.get("total_rating_distribution", {})
    for key in ["5", "4", "3", "2", "1"]:
        if key in total_rating:
            total_rating[key] = (
                int(total_rating[key])
                if isinstance(total_rating[key], (int, float, str))
                and str(total_rating[key]).replace(",", "").isdigit()
                else 0
            )
    json_data["total_rating_distribution"] = total_rating

    # 전처리된 데이터 반환
    return json_data


if __name__ == "__main__":
    # main.py 실행 후 생성된 파일명을 여기에 입력하세요
    target_file = "result_오일.json"
    result = preprocess_format(target_file)

    # 결과 저장
    output_filename = target_file.replace(".json", "_formatted.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"전처리 완료: {output_filename}")
