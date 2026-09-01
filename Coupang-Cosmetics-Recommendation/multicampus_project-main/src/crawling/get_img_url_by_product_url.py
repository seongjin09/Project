import os
import json
import random
import re
import time
from pathlib import Path
from curl_cffi import requests
from tqdm import tqdm

# 설정
INPUT_DIR = "./data/pre_data"
OUTPUT_DIR = "./data/new_pre_data"
COOKIE = "x-coupang-target-market=KR; x-coupang-accept-language=ko-KR; PCID=17664634680114475919418; MARKETID=17664634680114475919418; sid=d0170f68140f4305b6d630c289b26f3a3b23b0a0; baby-isWide=wide; ak_bmsc=1F6440FC19D82C4E415B375C4F9F12BB~000000000000000000000000000000~YAAQheQ1F5Jp9dibAQAAevbj/R7K5YJdHMxFpVXsj8Baf1nv31Mhb4XYirO1zcYST5qmNXQL8i7/yiJ75NOsjjvtI4mXx/SX65y4oJIMBQaauW4evNsTYuWnMrqRV5acGm8gLNPlzahti3+z+6Gwtpj36venm4VRy34nZPCjwxIRNDTgUp+FkewE8v4MWw77Y1CjtSfvudYMaJe+mUgJDl7faA6+14AHwQEuOVu71FjoOtVEJWk7mXiAYt/D2+imiFIxD+cMvSTNy0rme+XgyZsBHFaulSTzx7MlbJbzPRFnubrgCHkQgvvCIAdXei0z9gwGGV/67NyOCAvIkQo8hn3Ti77Id692HcmgIAVapGa9n3jgVKsZiiRCyEDy6BR8OLxF3Z/CX0Wb/gfN0qgzAWBkx3M8TQUxTsBtTu3fDxshGbP+LtU1vXXqmhDvCMMygXgs41HTTGEjejcCc3DUvQ==; bm_ss=ab8e18ef4e; bm_so=CA3E6A8BE503CCA8E44FDB535B90DDA3368CDBD976361E945778F6C949ABBC66~YAAQLa0sFxN2adebAQAAHKEh/gaafq1+bF+uOLgBOiQbUTk5zdKUPaMRh/V2iTtJzkQMqVijQSxu0qk2V3F8sc/wKSglwcISD8hk6TUtDrAo4MLOcCJ+BT8CLsOOqjHkUFf3aAWfrU/0KnQmBlIBm9Ye7kyoXA8i0Psq/dNpejU9jBa+jL55s8aAybiLOO7IIO2fQ4Jq074ExGYAnBO+TFbyi4vg9DPgYkf6BjceWttCARkc4IQy8/M9eIT8BXlWwbhzEB4kitVJIIR/Lpq216BL7oKSCPkD+oKNHD4g4byNv33vf48eKB5qZ1f8mouizlSP8Wzw7U6rT0Z4IrbHiVMY6Akdt+tWDYaEhkO2zkJNCTZV80oxPYIcuy4DInEAk1ZJfF82pU+wwWGhlrYOvHwpzPJ9fGDgjamdpVc15flEdk/yVolXmEMtbl5UOm4KMRYMydocfkUvoQejDcz9; bm_sz=54B1C31EC60E1CD843FF594ED0E9104E~YAAQLa0sFwSSadebAQAAkNIh/h5tUj1r85Yf80oj7F1D4wp/n7Cu/ZFQgrFNgmi+tUMDscfUT+4q6gqAXFZi78H8zbKdB12pmw42p3uVRYxBs2MlxdD9zpwtBSoLAacnsJygo7JTVhop2JQP9iBqr09ubBfPLK+ISn25Zj76thTsQ3qhXuip99ZUBEAjj2H9MNsYts4C6QcTJHN1w7tIT7uLflRs37Fkq5YJ0R/we8c3yPkjSVQeIf6LlgfnQLuylvJ+u/xZhir8QIJ4LGdrm5iJedhsJ8t3776Z0Y+I+xz6VYeNqeHlvIIergFKOWl8F74Q/5aAJ2dKlTjiTFDHAe0UGDQVjKPZ5geQepiA5KpLtsoyr84T6MbPU9CpU/tUaI6hvmDoXyh9vQ1VnwbgEBtbxYz7zrxTVWRmT5l1ofOM9PZGCKL2rmFtMIRRrCDA3zwNwUjC2zm7t4jL4GbR8aUmj0CHQInxH7fus59e1rBQPUf6wFC6VYfg3yNKsfSYoZlfqzy8r2Zt8QN4S1xZaij7FVjTtIVMoViKwfFKrkgktXzlpRdmL/BM8M5vPowPLnsKZe8hniwzsR7QRzjq~4404547~3355206; _abck=20D7C86C72D94700947923C0DD6ED8CA~0~YAAQLa0sF3aWadebAQAAM9oh/g/N2qvaTvvJfUDueHdHl6/FZGGjir+tswPSUeyA4ZXp4Vg3OQdqygBtmHIQpVS6BFXJelRXFyR978LFpFn4KvW4wGLKVyhNPxuyr3WS8euDaLQkeyXb+p+WplM7SGf/N17JK0ge2vZzIDN55KXiXT6wJoQANbomfnqzK/adW1kCsUPpEQa18YmXc9Gf6nbl7hQ96t0HL1LKHfV4NHqQAKYRfxwiMpVhMQp2l3Z+abmKIqYjn9lkCxciCTMg3UtxaxcBG6PbSIViVXL4qkOgzwcMGBMDiV7Ctl3PS+tP/uQHXzrErga1V0fjWKDoXRWXPiS/Fy0W8g8QmChazaXJbYVkAy8dwCrLrA5d5bzSco/lx1XoFkyDtm7xlxRBqIAhMgWd934fEXIO74CCHsuDW1IAm+5if9CzRrahmwf8PEqSIpoaf9g6RLTWu3lfWqkJSUcrm7ZHWC4OjNvKBEPojA/EK6UQQqHUW5J4or2Yp66YIfEtk6z708A4SptDxdQ+UQnYjkQm5ax//CUiyMt3XmnntCnOy+7lDCnQCs2qmmtpEVAlnDZlgG3KblfAjlsZMS/MnhPx97EyRtCYcYVF8MS/qeikJfymLd2WQX1Xmneq9ts9soQaq6n+wA+PJoeIGr56N96SecGqr068if8iMQA/D4fZdnzI7SfV9kT3UsiQTHH29Fj/ufWUFyT6N+EkLEuBIcP6MwsD1jn7lL1H~-1~-1~1769495430~AAQAAAAF%2f%2f%2f%2f%2f5PNEJI8h3IC5Kd6wRP9OITzbtL9X0QN3c7tDfMDjn6pCr5M8L2zpphyJ+i4ST7EWHMeDmJELoFrJfrPndN7yebw5gXXmHkmX0dG00Xkjya78wXbDYqXmCdA+fNkdEXC5hNETtPN4Dcq6eY6Bqu%2fPBMFhUMKKFzu6O8HTZHDB9el9p732ti3MhEcgMrzvNPVbCdg6JBduf6ryCPC3rkPJLsEQfvL5Cou78BhTu2w%2fqRAtCU%3d~-1; bm_s=YAAQLa0sF3eWadebAQAAM9oh/gQLXmA3kZ3S2/3eW21ciwoEvwYKxj/2lawoq6p62uF3eTNk1OHZezsaOF88D0VtVjnmjG0X3MtgAI/H91SKILpQnPFC9grc4mUYRst2SgkHOn5Ze/mbNvNBBK7mgLDk5VY2/fJV3Pct6BYBvmiLDakEjYun5j2i2rvFrACeXlmS3k5PNiDv0xP1Qzc79DjzWns1dUO0XbyAdaTxB9pe57NGB7cSfZbOj7Gmcj5crsjDYVL0jNQ/05nSn/DchtE54J0JJ4RrMS+BASwlp1htls8G65iBIRsFlbMrbadkgHmeR0xa3s+JoJbauJJl7oxZr+jiVNMMUvJQVqsFhjSrlmnbdXp00A9qaFCIrXSwW4S5Rrtjgyb97Hdr23RoKCWQaUo5u6FSyjyV/OFORkE/HoVUcxTGxj5wKGDUCo/cC29nIj9Ag3gBr6HrEKO6/sLAN0WGkmYA0mgpwQV6cBkWvEr/iIbYmyazNEYl5DQF1P9PEa7g2R/duihfUip4Mk1rp8eWpu1gkhV4ZbhDpsYZKVqjCvxQowtshx2RIkW0wb16BwaRY1mxV60=; bm_sv=1E7CE6DFEF00FD6ABF4D51AEA28CC676~YAAQLa0sF3iWadebAQAAM9oh/h7YRryix2puoWkCQ0JFsRZ4y1tuFV8uYzReEp6I703f1H1BXsJ2NvvG9eviE/pxtKjPwaaL8bT0yTipmVVdVwvEmzA/MPXOCHObmW65Itq3svBceP8tzt2A1o7hVU1huSYwWtZUb3F4uqRM+UR/kilx86+yd1kM8BG2ql8G+LIOXgQGYJBOoNI/BaKCfVnOPi/+8HWX61Mb2HfIYCF7FvOEC36J6Y3wr+VbCqhAJks=~1"
# 정규식 패턴 (속도 향상을 위해 컴파일)
IMG_PATTERN = re.compile(r'\\"image\\":\s*\[\s*\\"([^\\"]+)\\"')


def get_product_image(url):
    """
    제공된 URL에서 상품 이미지 주소를 추출합니다.
    """
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.coupang.com/",
                "Cookie": COOKIE,
            },
            impersonate="chrome124",
            timeout=10,  # 타임아웃 설정
        )

        if response.status_code != 200:
            return None

        matches = IMG_PATTERN.findall(response.text)
        if matches:
            path = matches[0]
            return f"https:{path}" if path.startswith("//") else path
        return None
    except Exception as e:
        # print(f"Error fetching {url}: {e}")
        return None


def process_files():
    # 폴더 구조 생성을 위한 Path 객체
    input_base = Path(INPUT_DIR)
    output_base = Path(OUTPUT_DIR)

    # 모든 json 파일 찾기
    json_files = list(input_base.rglob("*.json"))
    print(f"총 {len(json_files)}개의 파일을 발견했습니다.")

    total_success = 0
    total_fail = 0
    total_skipped = 0

    for json_file in json_files:
        print(f"\n파일 처리 중: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ JSON 로드 실패: {json_file}")
                continue

        file_success = 0
        file_fail = 0
        file_skipped = 0

        # 데이터 리스트 순회
        products_data = content.get("data", [])

        # tqdm으로 파일 내 상품 처리 진행도 표시
        for item in tqdm(products_data, desc=f"{json_file.name} 처리 중"):
            info = item.get("product_info", {})
            product_url = info.get("product_url")

            # 1. 이미 img_url이 있는지 확인
            if "img_url" in info:
                file_skipped += 1
                continue

            if not product_url:
                file_fail += 1
                continue

            # 2. 이미지 URL 요청
            img_url = get_product_image(product_url)

            if img_url:
                info["img_url"] = img_url
                file_success += 1
                # 과도한 요청 방지를 위한 미세한 대기 (선택 사항)
                # 랜덤
                time.sleep(random.uniform(1, 1.2))
            else:
                file_fail += 1

        # 결과 파일 경로 설정 및 폴더 생성
        relative_path = json_file.relative_to(input_base)
        output_file_path = output_base / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 파일 저장
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

        print(f"✅ 결과 저장 완료: {output_file_path}")
        print(f"   (성공: {file_success}, 실패: {file_fail}, 건너뜀: {file_skipped})")

        total_success += file_success
        total_fail += file_fail
        total_skipped += file_skipped

    print("\n" + "=" * 50)
    print("🎉 모든 작업이 완료되었습니다!")
    print(f"총 성공 상품: {total_success}")
    print(f"총 실패 상품: {total_fail}")
    print(f"총 건너뛴 상품: {total_skipped}")
    print("=" * 50)


if __name__ == "__main__":
    process_files()
