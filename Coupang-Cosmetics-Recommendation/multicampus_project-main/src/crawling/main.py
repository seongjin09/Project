import json
import time
import random
import os  # 파일 존재 여부 확인용
import undetected_chromedriver as uc
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    NoSuchWindowException,
    WebDriverException,
)
import gc

from get_product_urls import get_product_urls, get_category_product_urls
from get_product_reviews import get_product_reviews


def main():
    # [방법 1] 키워드 검색을 원할 때
    # MODE = "KEYWORD"
    # TARGETS = ["사과"]

    # [방법 2] 카테고리 수집을 원할 때
    MODE = "CATEGORY"
    TARGETS = {
        # "스킨": "486248",
        # "로션": "486249",
        # "에센스_세럼_앰플": "486250",
        # "미스트": "486251",
        # "오일": "486252",
        # "페이셜크림": "486269",
        # "아이_넥크림": "486270",
        # "멀티밤_스틱": "508961",
        # "올인원": "486271",
        # "알로에_수딩_애프터선": "486272",
        # "기초세트": "486254",
        # "틴트_립글로스": "176582",
        # "립스틱": "176583",
        # "립케어": "176584",
        # "포인트리무버": "176586",
        # "블러셔": "176595",
        # "하이라이터": "403010",
        "셰이딩": "403011",
    }
    PRODUCT_LIMIT = 200
    REVIEW_TARGET = 200
    MAX_REVIEWS_PER_SEARCH = 50000

    print(">>> 전체 작업을 시작합니다...")

    try:
        # 반복문 시작 부분 수정
        if MODE == "KEYWORD":
            iterable = enumerate(TARGETS)
        else:
            iterable = enumerate(TARGETS.items())  # (0, ("스킨케어", "486248"))

        for k_idx, item in iterable:
            # 이 키워드/카테고리 시작 시간 기록
            search_start_time = time.time()

            # 모드에 따라 변수 할당
            if MODE == "KEYWORD":
                search_key = item  # "사과"
                search_id = None
            else:
                search_key = item[0]  # "스킨케어"
                search_id = item[1]  # "486248"

            # ---------------------------------------------------------
            # [이어하기 기능] 기존 데이터 로드
            # ---------------------------------------------------------
            crawled_data_list = []
            keyword_total_collected = 0
            keyword_total_text = 0
            total_rating_distribution = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}
            processed_urls = set()  # 이미 수집한 URL 집합

            # 읽어올 파일명 후보 (중단된 파일 우선, 없으면 완료 파일 확인)
            resume_filenames = [
                f"result_{search_key}_interrupted.json",
                f"result_{search_key}.json",
            ]

            for fname in resume_filenames:
                if os.path.exists(fname):
                    try:
                        with open(fname, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)

                        # 데이터 복원
                        if "data" in existing_data:
                            crawled_data_list = existing_data["data"]
                            # URL 추출하여 집합에 추가
                            for row in crawled_data_list:
                                p_url = row.get("product_info", {}).get("product_url")
                                if p_url:
                                    processed_urls.add(p_url)

                            # 통계 복원
                            keyword_total_collected = existing_data.get(
                                "total_collected_reviews", 0
                            )
                            keyword_total_text = existing_data.get(
                                "total_text_reviews", 0
                            )
                            if "total_rating_distribution" in existing_data:
                                total_rating_distribution = existing_data[
                                    "total_rating_distribution"
                                ]

                        print(
                            f"\n>>> [이어하기] '{fname}' 파일 발견! 기존 데이터 {len(crawled_data_list)}개 로드 완료."
                        )
                        print(
                            f">>> [이어하기] 이미 수집된 URL {len(processed_urls)}개는 건너뜁니다."
                        )
                        break
                    except Exception as e:
                        print(
                            f">>> [이어하기] 파일 읽기 실패 (무시하고 새로 시작): {e}"
                        )

            # 전체 별점 분포 집계 (기존 코드에서는 여기 위치했으나 위로 이동함)
            # total_rating_distribution = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}

            # ---------------------------------------------------------
            # [단계 1] URL 수집
            # ---------------------------------------------------------
            total_count = len(TARGETS) if MODE == "KEYWORD" else len(TARGETS.items())
            print(f"\n{'='*50}")
            print(f">>> [{k_idx+1}/{total_count}] '{search_key}' URL 수집 시작")
            print(f"{'='*50}")

            # URL 수집 재시도 로직
            URL_COLLECT_MAX_RETRIES = 3
            urls = []
            consecutive_failures = (
                0  # 연속 실패 카운터 (URL 실패 + 리뷰 수집 실패 통합)
            )
            CONSECUTIVE_FAIL_LIMIT = 10  # 연속 실패 허용 횟수
            WAIT_TIME_ON_CONSECUTIVE_FAIL = 25 * 60  # 20분 (초 단위)

            for url_attempt in range(URL_COLLECT_MAX_RETRIES):
                print(
                    f"\n>>> URL 수집 시도 [{url_attempt+1}/{URL_COLLECT_MAX_RETRIES}]"
                )

                options = uc.ChromeOptions()
                options.add_argument("--no-first-run")
                options.add_argument("--no-service-autorun")
                options.add_argument("--password-store=basic")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--blink-settings=imagesEnabled=false")

                driver = uc.Chrome(
                    options=options, use_subprocess=False, version_main=144
                )
                try:
                    # 모드에 따라 호출 함수 분기
                    if MODE == "KEYWORD":
                        urls = get_product_urls(
                            driver,
                            search_key,
                            max_products=PRODUCT_LIMIT,
                            min_reviews=100,
                        )
                    else:
                        # 카테고리 ID로 수집 함수 호출
                        urls = get_category_product_urls(
                            driver,
                            search_id,
                            max_products=PRODUCT_LIMIT,
                            min_reviews=100,
                        )

                    print(f">>> [{search_key}] URL {len(urls)}개 확보 완료")
                    # 중복 urls 제거
                    urls = list(set(urls))
                    print(
                        f">>> [{search_key}] 중복 제거 후 URL {len(urls)}개 확보 완료"
                    )

                    # ---------------------------------------------------------
                    # [이어하기 기능] 이미 수집된 URL 필터링
                    # ---------------------------------------------------------
                    if processed_urls:
                        original_count = len(urls)
                        # 기존에 없는 URL만 남김
                        urls = [u for u in urls if u not in processed_urls]
                        skipped_count = original_count - len(urls)
                        print(
                            f">>> [이어하기] 기존 {skipped_count}개 URL 제외 -> 남은 URL {len(urls)}개"
                        )

                    time.sleep(1)
                    # 성공하면 루프 탈출
                    if urls or processed_urls:
                        consecutive_failures = 0  # 성공 시 연속 실패 카운터 리셋
                        break
                    else:
                        print(f">>> URL 수집 실패 (0개) - 재시도합니다.")
                        consecutive_failures += 1

                except Exception as e:
                    print(f">>> URL 수집 중 에러: {e}")
                    urls = []
                    consecutive_failures += 1
                finally:
                    print(">>> URL 수집 브라우저 종료 및 메모리 정리 중...")
                    driver = driver_cleanup(driver)

                    # 연속 실패 5번 체크
                    if consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
                        print(f"\n!!! 연속 {consecutive_failures}번 실패 감지 !!!")
                        print(
                            f"!!! {WAIT_TIME_ON_CONSECUTIVE_FAIL // 60}분 대기 후 재시도합니다..."
                        )
                        time.sleep(WAIT_TIME_ON_CONSECUTIVE_FAIL)
                        consecutive_failures = 0  # 대기 후 카운터 리셋

                # 마지막 시도가 아니면 대기
                if url_attempt < URL_COLLECT_MAX_RETRIES - 1 and not urls:
                    print(">>> 20초 대기 후 재시도...")
                    time.sleep(20)

            if not urls and not processed_urls:
                print(
                    f">>> [{search_key}] {URL_COLLECT_MAX_RETRIES}번 시도 후에도 URL을 수집하지 못했습니다. 넘어갑니다."
                )
                continue

            if not urls and processed_urls:
                print(
                    f">>> [{search_key}] 이미 모든 URL을 수집했습니다. 다음으로 넘어갑니다."
                )
                # 저장 로직을 거치도록 urls=[] 상태로 진행

            # ---------------------------------------------------------
            # [단계 2] 개별 상품 리뷰 수집 (리뷰 수에 따라 드라이버 재사용/재시작)
            # ---------------------------------------------------------
            print(f">>> [{search_key}] 상세 리뷰 수집 시작")

            # 첫 상품을 위한 드라이버 생성
            driver = None
            driver_collected_count = 0  # 현재 드라이버가 수집한 총 리뷰 개수

            for idx, url in enumerate(urls):
                # 타겟 리뷰 개수 도달 체크 (기존 데이터 포함)
                if keyword_total_collected >= MAX_REVIEWS_PER_SEARCH:
                    print(
                        f"\n>>> [{search_key}] 타겟 리뷰 개수({MAX_REVIEWS_PER_SEARCH}개) 도달!"
                    )
                    print(f">>> 수집을 종료하고 저장합니다.")
                    break

                print(f"\n   [{idx+1}/{len(urls)}] 상품 처리 시작... ({search_key})")

                MAX_RETRIES = 3
                success = False

                for attempt in range(MAX_RETRIES):
                    print(
                        f"     -> [시도 {attempt+1}/{MAX_RETRIES}] 브라우저 실행 중..."
                    )

                    try:
                        # 드라이버가 없으면 새로 생성
                        if driver is None:
                            options = uc.ChromeOptions()
                            options.add_argument("--no-first-run")
                            options.add_argument("--no-service-autorun")
                            options.add_argument("--password-store=basic")
                            options.add_argument("--window-size=1920,1080")
                            options.add_argument("--blink-settings=imagesEnabled=false")
                            driver = uc.Chrome(
                                options=options, use_subprocess=False, version_main=144
                            )
                            driver_collected_count = 0

                        # 수집 함수 호출 (현재 드라이버 수집량 전달)
                        data = get_product_reviews(
                            driver,
                            url,
                            idx + 1,
                            target_review_count=REVIEW_TARGET,
                            driver_collected_count=driver_collected_count,
                        )

                        # 브랜드 본사 정품 상품인 경우 드라이버 재시작 없이 스킵
                        if data and data.get("skip_official_product"):
                            print(
                                "     -> [브랜드 본사 정품] 드라이버 재시작 없이 다음 상품으로 넘어갑니다."
                            )
                            consecutive_failures = 0  # 성공으로 간주하고 카운터 리셋
                            success = True
                            break

                        if (
                            data
                            and data.get("product_info")
                            and data["product_info"].get("product_id")
                        ):
                            r_data = data.get("reviews", {})
                            current_collected = r_data.get("total_count", 0)

                            # 리뷰가 0개면 저장하지 않고 다음으로
                            if current_collected == 0:
                                print(
                                    "     -> [스킵] 리뷰가 0개입니다. 저장하지 않고 다음 상품으로 넘어갑니다."
                                )
                                consecutive_failures = (
                                    0  # 성공으로 간주하고 카운터 리셋
                                )
                                success = True
                                break

                            keyword_total_collected += current_collected
                            keyword_total_text += r_data.get("text_count", 0)
                            driver_collected_count += current_collected

                            # 상품별 rating_distribution을 전체에 합산
                            product_rating = data.get("product_info", {}).get(
                                "rating_distribution", {}
                            )
                            for score, count in product_rating.items():
                                total_rating_distribution[score] += count

                            crawled_data_list.append(data)
                            print(
                                f"     -> [성공] 수집 완료 (전체: {current_collected}개, 글 포함: {r_data.get('text_count', 0)}개)"
                            )
                            print(
                                f"     -> 키워드 누적: {keyword_total_collected}개 / 현 드라이버: {driver_collected_count}개"
                            )

                            consecutive_failures = 0  # 성공 시 연속 실패 카운터 리셋
                            success = True

                            # 타겟 리뷰 개수 도달 체크
                            if keyword_total_collected >= MAX_REVIEWS_PER_SEARCH:
                                print(
                                    f"\n>>> [{search_key}] 타겟 리뷰 개수({MAX_REVIEWS_PER_SEARCH}개) 도달!"
                                )
                                print(
                                    f">>> 현재 수집: {keyword_total_collected}개 / URL 남음: {len(urls) - idx - 1}개"
                                )
                                print(f">>> 수집을 종료하고 저장합니다.")
                                break

                            break
                        else:
                            print("     -> [실패] 데이터가 비어있습니다. 재시도합니다.")
                            consecutive_failures += 1
                            # 드라이버 재시작
                            driver = driver_cleanup(driver)
                            driver_collected_count = 0

                            # 연속 실패 5번 체크
                            if consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
                                print(
                                    f"\n!!! 연속 {consecutive_failures}번 실패 감지 !!!"
                                )
                                print(
                                    f"!!! {WAIT_TIME_ON_CONSECUTIVE_FAIL // 60}분 대기 후 재시도합니다..."
                                )
                                time.sleep(WAIT_TIME_ON_CONSECUTIVE_FAIL)
                                consecutive_failures = 0  # 대기 후 카운터 리셋

                    except Exception as e:
                        print(f"     -> [에러 발생] {e}")
                        consecutive_failures += 1
                        # 에러 발생 시 드라이버 재시작
                        if driver:
                            driver = driver_cleanup(driver)
                            driver_collected_count = 0

                        # 연속 실패 5번 체크
                        if consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
                            print(f"\n!!! 연속 {consecutive_failures}번 실패 감지 !!!")
                            print(
                                f"!!! {WAIT_TIME_ON_CONSECUTIVE_FAIL // 60}분 대기 후 재시도합니다..."
                            )
                            time.sleep(WAIT_TIME_ON_CONSECUTIVE_FAIL)
                            consecutive_failures = 0  # 대기 후 카운터 리셋

                        continue

                # 2번 다 실패했을 경우
                if not success:
                    print(
                        f"     -> [최종 실패] {MAX_RETRIES}번 시도했으나 수집 실패. 다음 상품으로 넘어갑니다."
                    )

                # 타겟 리뷰 개수 도달 시 URL 루프 탈출
                if keyword_total_collected >= MAX_REVIEWS_PER_SEARCH:
                    break

                # print(f"     -> 다음 상품 대기중...(2초)")
                time.sleep(2)

            # 키워드/카테고리 처리 완료 후 드라이버가 남아있으면 종료
            if driver:
                print(f">>> [{search_key}] 모든 상품 처리 완료 - 드라이버 종료")
                driver = driver_cleanup(driver)

            # ---------------------------------------------------------
            # [단계 3] 키워드/카테고리 완료 후 저장
            # ---------------------------------------------------------
            result_json = {
                "search_name": search_key,
                "total_collected_reviews": keyword_total_collected,
                "total_text_reviews": keyword_total_text,
                "total_product": len(crawled_data_list),
                "total_rating_distribution": total_rating_distribution,
                "data": crawled_data_list,
            }

            if crawled_data_list:
                filename = f"result_{search_key}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, indent=2, ensure_ascii=False)
                print(f"\n [{search_key}] 저장 완료: {filename}")

                # [이어하기 기능] 완료되었으므로 임시 중단 파일 삭제
                interrupted_file = f"result_{search_key}_interrupted.json"
                if os.path.exists(interrupted_file):
                    try:
                        os.remove(interrupted_file)
                        print(f"     -> 임시 중단 파일 삭제됨: {interrupted_file}")
                    except:
                        pass
            else:
                print(f"\n[{search_key}] 수집된 데이터가 없습니다.")

            search_end_time = time.time()
            search_elapsed = search_end_time - search_start_time
            search_hours = int(search_elapsed // 3600)
            search_minutes = int((search_elapsed % 3600) // 60)
            search_seconds = int(search_elapsed % 60)
            print(
                f"\n[{search_key}] 처리 시간: {search_hours}시간 {search_minutes}분 {search_seconds}초"
            )

            # 다음 키워드/카테고리 준비 중 (마지막이 아닐 때만)
            if k_idx < total_count - 1:  # 마지막이 아닐 때만
                print(f">>> 다음 항목 준비 중 (20.0초)...")
                time.sleep(20)

    except KeyboardInterrupt:
        print("\n>>> 사용자에 의해 작업이 중단되었습니다.")

        # 드라이버 정리
        try:
            if driver:
                print(">>> 드라이버 정리 중...")
                driver = driver_cleanup(driver)
        except:
            pass

        # 현재까지 수집된 데이터가 있으면 저장
        try:
            if crawled_data_list:
                result_json = {
                    "search_name": search_key,
                    "total_collected_reviews": keyword_total_collected,
                    "total_text_reviews": keyword_total_text,
                    "total_product": len(crawled_data_list),
                    "total_rating_distribution": total_rating_distribution,
                    "data": crawled_data_list,
                }

                filename = f"result_{search_key}_interrupted.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, indent=2, ensure_ascii=False)
                print(f">>> 중단 시점까지의 데이터 저장 완료: {filename}")
                print(
                    f">>> 저장된 데이터: 상품 {len(crawled_data_list)}개, 리뷰 {keyword_total_collected}개"
                )
        except Exception as e:
            print(f">>> 데이터 저장 중 오류: {e}")


def driver_cleanup(driver):
    try:
        driver.quit()
        print("driver 종료 및 30초 대기")
        try:
            del driver
        except Exception as e:
            print(f"드라이버 삭제 중 에러(무시됨): {e}")
        gc.collect()
        driver = None
        time.sleep(30)
    except Exception as e:
        print(f"드라이버 종료 중 에러(무시됨): {e}")

    return None


if __name__ == "__main__":
    main()
