from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import random
import urllib.parse


# [기존] 키워드 검색 URL 수집 함수
def get_product_urls(driver, keyword, max_products=5, min_reviews=0):
    """키워드 기반 검색으로 상품 URL 수집"""
    encoded_keyword = urllib.parse.quote(keyword)
    search_url = (
        f"https://www.coupang.com/np/search?component=&q={encoded_keyword}&channel=user"
    )
    return _collect_urls(
        driver, search_url, max_products, f"검색어: {keyword}", min_reviews
    )


# [신규] 카테고리 ID 기반 URL 수집 함수
def get_category_product_urls(driver, category_id, max_products=5, min_reviews=0):
    """카테고리 ID 기반으로 상품 URL 수집"""
    # listSize=60 (60개씩 보기), sorter=saleCountDesc (판매량순)
    category_url = f"https://www.coupang.com/np/categories/{category_id}?listSize=60&sorter=saleCountDesc"
    return _collect_urls(
        driver, category_url, max_products, f"카테고리ID: {category_id}", min_reviews
    )


# [공통] 내부 수집 로직
def _collect_urls(driver, target_url, max_products, log_prefix, min_reviews=0):
    """실제 URL 수집을 수행하는 내부 함수"""
    product_urls = []
    current_page = 1

    try:
        print(f"[get_urls] {log_prefix} 페이지 접속 중... (최소 리뷰: {min_reviews}개)")
        driver.get(target_url)
        time.sleep(random.uniform(2.5, 3.5))

        while len(product_urls) < max_products:
            # 1. 페이지 로딩 대기
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "product-list"))
                )
            except TimeoutException:
                print("   -> 상품 리스트를 찾을 수 없습니다.")
                break

            driver.execute_script("window.scrollBy(0, 1500);")
            time.sleep(1.5)

            # 2. 파싱
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            links = soup.select("ul#product-list > li > a")

            print(f"   -> {current_page}페이지 발견 상품: {len(links)}개")

            # 3. URL 추출 및 리뷰 개수 필터링
            for link in links:
                if len(product_urls) >= max_products:
                    break
                href = link.get("href")
                if not href or "javascript" in href or href == "#":
                    continue

                # 리뷰 개수 추출
                review_count = 0
                try:
                    # ProductUnit_productInfo__1l0il 클래스 div 찾기
                    product_info_div = link.select_one(
                        "div.ProductUnit_productInfo__1l0il"
                    )
                    if product_info_div:
                        # ProductRating_productRating__jjf7W 클래스 div 찾기
                        rating_div = product_info_div.select_one(
                            "div.ProductRating_productRating__jjf7W"
                        )
                        if rating_div:
                            # ProductRating_ratingCount__R0Vhz 클래스 span 찾기
                            rating_count_span = rating_div.select_one(
                                "span.ProductRating_ratingCount__R0Vhz"
                            )
                            if rating_count_span:
                                rating_text = rating_count_span.text.strip()
                                # "(" "숫자" ")" 형태에서 숫자만 추출
                                # 예: "(123)" -> "123"
                                import re

                                match = re.search(r"\((\d+)\)", rating_text)
                                if match:
                                    review_count = int(match.group(1))
                except Exception as e:
                    pass

                # 최소 리뷰 개수 필터링
                if review_count < min_reviews:
                    continue

                if href.startswith("/"):
                    full_url = f"https://www.coupang.com{href}"
                else:
                    full_url = href

                if full_url not in product_urls:
                    product_urls.append(full_url)
                    print(
                        f"   ✓ URL 수집: {len(product_urls)}번째 | 리뷰: {review_count}개 | {href[:50]}..."
                    )

            print(f"   -> 현재 확보: {len(product_urls)}/{max_products}개")

            if len(product_urls) >= max_products:
                break

            # 4. 다음 페이지 이동
            next_page = current_page + 1
            print(f"   -> {next_page}페이지로 이동 시도...")

            try:
                next_btn_xpath = (
                    f"//div[contains(@class, 'Pagination')]//a[text()='{next_page}']"
                )
                next_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, next_btn_xpath))
                )
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", next_btn
                )
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", next_btn)

                current_page += 1
                time.sleep(random.uniform(3, 4))

            except Exception:
                print("   -> 다음 페이지가 없습니다.")
                break

    except Exception as e:
        print(f"[get_urls] 에러: {e}")

    return product_urls
