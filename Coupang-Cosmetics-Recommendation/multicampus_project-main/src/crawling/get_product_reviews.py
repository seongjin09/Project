import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime


def clean_text(text):
    """비정상적인 줄 종결자, 특수 공백 및 특수 문자 제거"""
    if not text:
        return text

    # NBSP(\u00A0)를 가장 먼저 일반 공백으로 치환해야 합니다.
    text = text.replace("\u00a0", " ")

    # LS(U+2028), PS(U+2029) 등 특수 줄바꿈 문자를 일반 공백으로 치환
    text = text.replace("\u2028", " ").replace("\u2029", " ")

    # 기타 제어 문자 제거 (이제 일반 공백이 된 NBSP는 isprintable()을 통과함)
    text = "".join(char for char in text if char.isprintable() or char in "\n\r\t")

    return text.strip()


def get_product_reviews(
    driver, url, rank_num, target_review_count=100, driver_collected_count=0
):
    # 최종 결과를 담을 구조
    result_data = {
        "product_info": {},
        "reviews": {"total_count": 0, "text_count": 0, "data": []},
    }

    print(f"[Reviewer] 상품 페이지 접속: {url}")

    # 1. 페이지 접속
    driver.set_page_load_timeout(30)
    driver.get(url)

    # 접속 직후 알림창 처리
    try:
        alert = driver.switch_to.alert
        print(f"   -> 접속 직후 경고창 감지: {alert.text}")
        alert.accept()
    except:
        pass

    time.sleep(random.uniform(5, 6))

    # -------------------------------------------------------
    # [기본 정보 파싱]
    # -------------------------------------------------------
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    product_id = str(rank_num)
    product_name = "Unknown"
    time.sleep(1)
    try:
        product_name_h1 = soup.select_one("h1.product-title.twc-text-lg.twc-text-black")
        if product_name_h1:
            product_name_span = product_name_h1.select_one("span.twc-font-bold")
            if product_name_span:
                product_name = clean_text(product_name_span.text.strip())
    except:
        pass

    if product_name == "Unknown":
        try:
            product_name = clean_text(
                soup.select_one("h2.prod-buy-header__title").text.strip()
            )
        except:
            pass

    price = "0"
    try:
        price_tag = soup.select_one("div.price-amount.final-price-amount")
        if not price_tag:
            price_tag = soup.select_one(
                "div.option-table-list__option--selected div.option-table-list__option-price"
            )
        if price_tag:
            price = (
                price_tag.text.strip()
                .replace("원", "")
                .replace(",", "")
                .split()[0]
                .strip()
            )
    except:
        pass

    delivery_type = "일반배송"
    try:
        badge_img = soup.select_one("div.price-badge img")
        if badge_img:
            src = badge_img.get("src", "")
            if "rocket-fresh" in src:
                delivery_type = "로켓프레시"
            elif "badge_1998ab96bf7" in src:
                delivery_type = "로켓배송(쿠팡)"
            elif "badge_1998ab98cb6" in src:
                delivery_type = "로켓배송(파트너사)"
            elif "badge_199559e56f7" in src:
                delivery_type = "판매자 로켓"
            elif "global_b" in src:
                delivery_type = "로켓 직구"
    except:
        pass

    total_reviews = "0"
    try:
        review_count_text = soup.select_one("span.rating-count-txt").text.strip()
        total_reviews = review_count_text.split("개")[0].replace(",", "").strip()
    except:
        pass

    category_str = ""
    try:
        crumb_links = soup.select("ul.breadcrumb li a")
        category_str = " > ".join(
            [clean_text(link.text.strip()) for link in crumb_links if link.text.strip()]
        )
    except:
        pass

    # 브랜드 이름 추출
    brand_name = ""
    try:
        # 첫 번째 시도: twc-font-bold twc-text-[14px] ... 클래스
        brand_div = soup.select_one(
            "div.twc-font-bold.twc-text-\\[14px\\].twc-text-\\[\\#111\\].twc-leading-\\[17px\\].twc-max-w-\\[130px\\].md\\:twc-max-w-\\[328px\\].twc-overflow-hidden.twc-text-ellipsis.twc-whitespace-nowrap"
        )
        if brand_div:
            brand_name = clean_text(brand_div.text.strip())
        else:
            # 두 번째 시도: twc-mb-[12px] ... 클래스
            brand_div_alt = soup.select_one(
                "div.twc-mb-\\[12px\\].twc-text-\\[14px\\].twc-leading-\\[17px\\].twc-text-\\[\\#346AFF\\]"
            )
            if brand_div_alt:
                brand_name = clean_text(brand_div_alt.text.strip())
            else:
                # 세 번째 시도: twc-text-sm twc-text-blue-600 클래스
                brand_div_third = soup.select_one("div.twc-text-sm.twc-text-blue-600")
                if brand_div_third:
                    brand_name = clean_text(brand_div_third.text.strip())
    except:
        pass

    # 상품 이미지 URL 추출
    img_url = ""
    try:
        # class가 "twc-w-full twc-max-h-[546px]"인 img 태그 찾기
        img_tag = soup.select_one("img.twc-w-full.twc-max-h-\[546px\]")
        if img_tag and img_tag.get("src"):
            src = img_tag.get("src")
            # //로 시작하면 https: 붙이기
            if src.startswith("//"):
                img_url = f"https:{src}"
            else:
                img_url = src
    except:
        pass

    # -------------------------------------------------------
    # [브랜드 본사 정품 체크 - 수집 제외]
    # -------------------------------------------------------
    try:
        brand_official_div = soup.select_one(
            "div.twc-flex.twc-items-center.twc-justify-center.twc-text-\\[14px\\]\\/\\[17px\\].twc-text-\\[\\#FFF\\].twc-font-\\[600\\].twc-bg-bluegray-1000.twc-h-\\[32px\\].twc-p-\\[0_16px\\].twc-mb-\\[32px\\].max-md\\:twc-m-\\[0_-16px_32px\\]"
        )
        if brand_official_div:
            brand_official_span = brand_official_div.select_one("span.twc-ml-\\[3px\\]")
            if brand_official_span:
                brand_official_text = clean_text(brand_official_span.text.strip())
                if brand_official_text == "로켓배송 · 브랜드 본사 정품":
                    print(
                        "   -> [수집 제외] 브랜드 본사 정품 상품입니다. 수집을 건너뜁니다."
                    )
                    return {
                        "skip_official_product": True,
                        "product_info": {},
                        "reviews": {"total_count": 0, "text_count": 0, "data": []},
                    }
    except:
        pass

    print(
        f"   -> 상품ID: {product_id} / 브랜드: {brand_name} / 상품명: {product_name} / 상품 img: {img_url}"
    )
    print(f"   -> 가격: {price}원 / 배송: {delivery_type} / 총리뷰: {total_reviews}")

    if product_name == "Unknown":
        print(
            "   -> [접근 거절] 상품명을 가져올 수 없습니다. 드라이버 재시작이 필요합니다."
        )
        return {
            "product_info": {},
            "reviews": {"total_count": 0, "text_count": 0, "data": []},
        }

    # 총 리뷰 수가 0이면 바로 리턴
    if int(total_reviews) == 0:
        print("   -> [리뷰 없음] 총 리뷰 수가 0개입니다. 수집을 건너뜁니다.")
        result_data["product_info"] = {
            "product_id": product_id,
            "brand": brand_name,
            "category_path": category_str,
            "product_name": product_name,
            "img_url": img_url,
            "price": price,
            "delivery_type": delivery_type,
            "total_reviews": total_reviews,
            "product_url": url,
            "rating_distribution": {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0},
        }
        return result_data

    # -------------------------------------------------------
    # [리뷰 섹션 준비]
    # -------------------------------------------------------
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.3);")
    time.sleep(random.uniform(1, 1.5))

    try:
        review_section = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "sdpReview"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", review_section)
        driver.execute_script("window.scrollBy(0, -200);")
        time.sleep(random.uniform(2, 2.5))
    except:
        print("   -> 리뷰 섹션을 찾을 수 없습니다. (리뷰 없음 추정)")
        result_data["product_info"] = {
            "product_id": product_id,
            "brand": brand_name,
            "category_path": category_str,
            "product_name": product_name,
            "img_url": img_url,
            "price": price,
            "delivery_type": delivery_type,
            "total_reviews": total_reviews,
            "product_url": url,
            "rating_distribution": {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0},
        }
        return result_data

    # -------------------------------------------------------
    # ★ [핵심] 별점별 순회 수집 로직 적용 (각 별점당 target_review_count개씩)
    # -------------------------------------------------------

    STAR_RATINGS = [
        {"score": 5, "text": "최고"},
        {"score": 4, "text": "좋음"},
        {"score": 3, "text": "보통"},
        {"score": 2, "text": "별로"},
        {"score": 1, "text": "나쁨"},
    ]

    all_reviews_list = []
    total_text_collected = 0

    rating_distribution = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}

    # 첫 번째 별점 처리 전에 드롭다운에서 실제 별점별 개수 추출
    try:
        # 리뷰 섹션 상단으로 스크롤
        review_section = driver.find_element(By.ID, "sdpReview")
        driver.execute_script("arguments[0].scrollIntoView(true);", review_section)
        driver.execute_script("window.scrollBy(0, -200);")
        time.sleep(random.uniform(1.2, 1.7))

        dropdown_trigger = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//div[contains(@class, 'twc-flex') and contains(@class, 'twc-items-center') and contains(@class, 'twc-cursor-pointer')]//div[contains(@class, 'twc-text-[14px]')]",
                )
            )
        )
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});", dropdown_trigger
        )
        time.sleep(random.uniform(0.5, 1))
        driver.execute_script("arguments[0].click();", dropdown_trigger)
        time.sleep(random.uniform(1.2, 1.7))

        import re

        for star_info in STAR_RATINGS:
            target_text = star_info["text"]
            target_score = star_info["score"]
            try:
                option_xpath = f"//*[@data-radix-popper-content-wrapper]//*[text()='{target_text}']"
                text_element = driver.find_element(By.XPATH, option_xpath)

                parent_element = text_element.find_element(By.XPATH, "./..")
                full_text = parent_element.text.strip()

                match = re.search(r"\n([0-9,]+)", full_text)
                if not match:
                    match = re.search(r"\(([0-9,]+)\)", full_text)

                if match:
                    count_str = match.group(1).replace(",", "")
                    rating_distribution[str(target_score)] = int(count_str)
                    # print(f"   -> {target_text}({target_score}점) 리뷰: {count_str}개")
                else:
                    print(
                        f"   -> {target_text} 개수 추출 실패 - 전체 텍스트: '{full_text}'"
                    )
            except Exception as e:
                print(f"   -> {target_text} 개수 추출 실패: {e}")

        driver.execute_script("document.body.click();")
        time.sleep(random.uniform(0.5, 1))

    except Exception as e:
        print(f"   -> 별점별 개수 추출 실패: {e}")

    # -------------------------------------------------------
    # 드라이버 생명주기 체크 (수집 시작 전)
    # -------------------------------------------------------
    # 각 별점의 예상 수집량 계산
    total_expected_collection = 0
    for star_info in STAR_RATINGS:
        target_score = star_info["score"]
        actual_count = rating_distribution.get(str(target_score), 0)
        ten_percent = int(actual_count * 0.25)
        dynamic_target = max(target_review_count, ten_percent)
        total_expected_collection += dynamic_target

    # 현재 드라이버 수집량 + 예상 수집량이 5500 초과하면 재시작 필요
    if driver_collected_count + total_expected_collection > 5500:
        print(
            f"   -> [드라이버 한계] 현재: {driver_collected_count}개 + 예상: {total_expected_collection}개 = {driver_collected_count + total_expected_collection}개 > 5500"
        )
        print("   -> 드라이버 재시작을 위해 빈 데이터 반환")
        return {
            "product_info": {},
            "reviews": {"total_count": 0, "text_count": 0, "data": []},
        }
    else:
        # print(
        #     f"   -> [드라이버 체크] 현재: {driver_collected_count}개 + 예상: {total_expected_collection}개 = {driver_collected_count + total_expected_collection}개 ≤ 5500 → 진행"
        # )
        pass
    for star_info in STAR_RATINGS:
        target_score = star_info["score"]
        target_text = star_info["text"]

        # 이 별점의 실제 개수 확인
        actual_count = rating_distribution.get(str(target_score), 0)

        # 해당 별점의 리뷰가 0개면 드롭다운 클릭하지 않고 스킵
        if actual_count == 0:
            # print(f"\n   >>> [별점 스킵] '{target_text}' 리뷰 0개 - 수집하지 않음")
            continue

        # 25% 계산
        ten_percent = int(actual_count * 0.25)

        # 최종 수집 목표: max(REVIEW_TARGET, 25%)
        dynamic_target = max(target_review_count, ten_percent)

        # print(f"\n   >>> [별점 변경] '{target_text}' 리뷰 수집 시작")
        # print(
        #     f"       실제 리뷰: {actual_count}개 | 10%: {ten_percent}개 | 기본 목표: {target_review_count}개 → 수집 목표: {dynamic_target}개"
        # )

        # 리뷰 섹션 상단으로 스크롤 (드롭다운 버튼이 보이도록)
        try:
            review_section = driver.find_element(By.ID, "sdpReview")
            driver.execute_script("arguments[0].scrollIntoView(true);", review_section)
            driver.execute_script("window.scrollBy(0, -200);")  # 헤더 공간 확보
            time.sleep(random.uniform(1, 1.2))
        except:
            # 리뷰 섹션을 못 찾으면 페이지 상단으로
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(random.uniform(1, 1.2))
        # 1. 별점 드롭다운 열기 (Test Script의 XPath 사용)
        try:
            dropdown_trigger = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//div[contains(@class, 'twc-flex') and contains(@class, 'twc-items-center') and contains(@class, 'twc-cursor-pointer')]//div[contains(@class, 'twc-text-[14px]')]",
                    )
                )
            )
            # 드롭다운 버튼을 화면 중앙으로 스크롤
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", dropdown_trigger
            )
            time.sleep(random.uniform(0.7, 1.2))

            # 현재 선택된 텍스트 확인 (디버깅용)
            # print(f"     -> 현재 드롭다운 상태: {dropdown_trigger.text.strip()}")

            driver.execute_script("arguments[0].click();", dropdown_trigger)
            time.sleep(random.uniform(0.6, 1))  # 팝업 애니메이션 대기

        except Exception as e:
            print(f"     -> [SKIP] 드롭다운 버튼 클릭 실패: {e}")
            continue

        # 2. 팝업 내 옵션 선택 (Test Script의 XPath 및 로직 사용)
        try:
            # 텍스트가 정확히 일치하는 요소를 찾음 (text()='...')
            option_xpath = (
                f"//*[@data-radix-popper-content-wrapper]//*[text()='{target_text}']"
            )

            star_option = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, option_xpath))
            )

            # 클릭 전 스크롤 및 클릭 (안정성 확보)
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", star_option
            )
            time.sleep(random.uniform(0.7, 1.2))
            driver.execute_script("arguments[0].click();", star_option)

            # print(f"     -> 필터 적용 완료: {target_text}")
            time.sleep(random.uniform(0.8, 1.2))  # 리스트 갱신 대기 (중요)
        except Exception as e:
            print(f"     -> [SKIP] 옵션('{target_text}') 클릭 실패: {e}")
            try:
                driver.execute_script("document.body.click();")
            except:
                pass
            continue

        # ---------------------------------------------------
        # [페이지네이션] 해당 별점 내에서 페이지 넘기며 수집
        # ---------------------------------------------------
        current_page_num = 1
        star_collected_count = (
            0  # 이 별점에서 수집한 전체 리뷰 개수 (내용 유무 상관없이)
        )
        star_text_count = 0  # 이 별점에서 수집한 텍스트 리뷰 개수
        STAR_LIMIT = dynamic_target  # 동적으로 계산된 목표 개수

        while star_collected_count < STAR_LIMIT:
            # 리뷰 파싱
            curr_soup = BeautifulSoup(driver.page_source, "html.parser")
            review_articles = curr_soup.select("article.twc-border-bluegray-200")

            if not review_articles:
                # print(
                #     f"     -> 더 이상 표시할 리뷰가 없습니다. ('{target_text}' 수집: {star_collected_count}개)"
                # )
                break

            collected_timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")

            for article in review_articles:
                if star_collected_count >= STAR_LIMIT:
                    break

                try:
                    content_span = article.select_one("span.twc-bg-white")
                    content = content_span.text.strip() if content_span else ""
                    content = clean_text(content)  # 특수 문자 제거

                    rating = 0
                    rating_div = article.select_one(
                        r"div.twc-inline-flex.twc-items-center.twc-gap-\[2px\]"
                    )
                    if rating_div:
                        rating = len(rating_div.select("i.twc-bg-full-star"))

                    date_div = article.select_one("div.twc-text-bluegray-700")
                    date = date_div.text.strip() if date_div else ""
                    date = clean_text(date)  # 특수 문자 제거

                    # 닉네임 추출
                    nickname = ""
                    try:
                        nickname_span = article.select_one(
                            "span.twc-text-\\[16px\\]\\/\\[19px\\].twc-font-bold.twc-text-bluegray-900"
                        )
                        if nickname_span:
                            nickname = clean_text(nickname_span.text.strip())
                    except:
                        pass

                    title_div = article.select_one(
                        "div.twc-font-bold.twc-text-bluegray-900"
                    )
                    title = title_div.text.strip() if title_div else ""
                    title = clean_text(title)  # 특수 문자 제거

                    has_image = False
                    img_container = article.select_one(
                        "div.twc-overflow-x-auto.twc-scrollbar-hidden"
                    )
                    if img_container and img_container.select_one("img"):
                        has_image = True

                    # 도움이 된 수 추출
                    helpful_count = 0
                    try:
                        helpful_span = article.select_one(
                            "span:-soup-contains('명에게 도움이 됐어요')"
                        )
                        if helpful_span:
                            helpful_text = helpful_span.text.strip()
                            import re

                            match = re.search(r"(\d+)명에게", helpful_text)
                            if match:
                                helpful_count = int(match.group(1))
                    except:
                        pass

                    review_obj = {
                        "id": len(all_reviews_list) + 1,
                        "score": rating,
                        "date": date,
                        "collected_at": collected_timestamp,
                        "nickname": nickname,
                        "has_image": has_image,
                        "helpful_count": helpful_count,
                        "title": title,
                        "content": content,
                        "full_text": f"{title} {content}",
                    }

                    all_reviews_list.append(review_obj)
                    star_collected_count += 1  # 전체 리뷰 개수 증가

                    if content:
                        star_text_count += 1
                        total_text_collected += 1

                except:
                    continue

            # 이 별점의 목표량을 달성했으면 다음 별점으로
            if star_collected_count >= STAR_LIMIT:
                # print(
                #     f"     -> '{target_text}' 별점 목표 달성 ({star_collected_count}개)"
                # )
                break

            # 페이지 이동 로직
            if current_page_num % 10 == 0:
                try:
                    next_arrow_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "//div[contains(@class, 'twc-mt-[24px]') and contains(@class, 'twc-flex-wrap')]//button[last()]",
                            )
                        )
                    )
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});",
                        next_arrow_btn,
                    )
                    driver.execute_script("arguments[0].click();", next_arrow_btn)
                    time.sleep(random.uniform(1.1, 1.2))

                    next_page_number = current_page_num + 1
                    next_block_first_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                f"//button[.//span[text()='{next_page_number}']]",
                            )
                        )
                    )
                    driver.execute_script("arguments[0].click();", next_block_first_btn)
                    time.sleep(random.uniform(1.1, 1.2))

                    current_page_num = next_page_number
                    continue
                except:
                    # print(
                    #     f"     -> 다음 페이지 블록(화살표)이 없습니다. ('{target_text}' 수집: {star_collected_count}개)"
                    # )
                    break
            else:
                next_num = current_page_num + 1
                try:
                    next_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, f"//button[.//span[text()='{next_num}']]")
                        )
                    )
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", next_btn
                    )
                    driver.execute_script("arguments[0].click();", next_btn)
                    time.sleep(random.uniform(1.1, 1.2))
                    current_page_num += 1
                except:
                    # print(
                    #     f"     -> 마지막 페이지 도달 ({current_page_num}페이지, '{target_text}' 수집: {star_collected_count}개)"
                    # )
                    break

    result_data["product_info"] = {
        "product_id": product_id,
        "brand": brand_name,
        "category_path": category_str,
        "product_name": product_name,
        "img_url": img_url,
        "price": price,
        "delivery_type": delivery_type,
        "total_reviews": total_reviews,
        "product_url": url,
        "rating_distribution": rating_distribution,
    }

    result_data["reviews"] = {
        "total_count": len(all_reviews_list),
        "text_count": total_text_collected,
        "data": all_reviews_list,
    }

    return result_data
