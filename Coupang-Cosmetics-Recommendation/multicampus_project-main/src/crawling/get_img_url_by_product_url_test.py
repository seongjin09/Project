import re
from curl_cffi import requests


def get_single_product_image():
    return next(
        iter(
            [
                f"https:{path}" if path.startswith("//") else path
                for path in re.findall(
                    r'\\"image\\":\s*\[\s*\\"([^\\"]+)\\"',
                    requests.get(
                        "https://www.coupang.com/vp/products/9034791497?itemId=26504119443&vendorItemId=93478816329&sourceType=CATEGORY&categoryId=176483",
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
                            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                            "Referer": "https://www.coupang.com/",
                            "Cookie": "x-coupang-target-market=KR; x-coupang-accept-language=ko-KR; PCID=17664634680114475919418; MARKETID=17664634680114475919418; sid=d0170f68140f4305b6d630c289b26f3a3b23b0a0; baby-isWide=wide; ak_bmsc=1F6440FC19D82C4E415B375C4F9F12BB~000000000000000000000000000000~YAAQheQ1F5Jp9dibAQAAevbj/R7K5YJdHMxFpVXsj8Baf1nv31Mhb4XYirO1zcYST5qmNXQL8i7/yiJ75NOsjjvtI4mXx/SX65y4oJIMBQaauW4evNsTYuWnMrqRV5acGm8gLNPlzahti3+z+6Gwtpj36venm4VRy34nZPCjwxIRNDTgUp+FkewE8v4MWw77Y1CjtSfvudYMaJe+mUgJDl7faA6+14AHwQEuOVu71FjoOtVEJWk7mXiAYt/D2+imiFIxD+cMvSTNy0rme+XgyZsBHFaulSTzx7MlbJbzPRFnubrgCHkQgvvCIAdXei0z9gwGGV/67NyOCAvIkQo8hn3Ti77Id692HcmgIAVapGa9n3jgVKsZiiRCyEDy6BR8OLxF3Z/CX0Wb/gfN0qgzAWBkx3M8TQUxTsBtTu3fDxshGbP+LtU1vXXqmhDvCMMygXgs41HTTGEjejcCc3DUvQ==; bm_ss=ab8e18ef4e; _abck=20D7C86C72D94700947923C0DD6ED8CA~0~YAAQVXXTFwuCRPmbAQAAwycp/g+u0bF1yxDPkv/9TDng+L5vgPpHzqJ5YbYqai3G87YsFegqsM+6Pe3cXUMA2gkiQYT+o2PvhZIhpFCUqyjVBP52nNJeTLpw5AXsbhQlQlHGfYqbu4yQ9mK4MNya3gAXue2fr3jRp+SdHjH8pcE/DcOI022zNZcRF5Q3onoIjRJLp9sQb4h4irA3QAJfXIwQv3C6u8vq1Uk5uLoDcDuqUgH0FZ6qEYnUyeQrlAqX7RekgQ05rhnYIwwMauBLBVFv7UdgR7x+vT7TAAQZFzw9tyOyvm6D3uq1Dhp8cqOA91cV7vzBqyo+FarQSBzD8Ne9aeFWlERPVCwh2L2/Yi/vEKGwa2wLHKot2Xvz+7roYiGBj0PjHYh9k5+c1WVSDv+SrV+8bJ0Vy65aAuh/SsDEISFd6aTk0tkDbGVIn18qEmxWoeIhF0PGEWI8U57XB58qiImFMODLtLnjON/SyZBLtogyIN4rLvcKkomnGO/BdHKqzMOn6AfzkNyR8A5SRfIuc8N4/Mnr5a7YzZPkTzIc696PE91WoPSo/V7eSUTe+sxu+8uZOno4rJEL1C8ccg0coGMQzRbOFWrYubUj46iMPlx2hTcrs/OGyzvgwXYv15+QlfCEyLzySTeTZk1k/3FZbr3U3spNmYSBP6wo+Gvz4JHTp4LdnVdpC8MsfHL8aKkW+5ArNNTbIWApPp2Z9T9nNa1RPFUEL9S1GoYE9IRq~-1~-1~1769499031~AAQAAAAF%2f%2f%2f%2f%2f26Uk3bFUWnO%2f1eILyZh+Gwfe8mJopx+9HiStjEtcnapLiRqpK3CRqiAPKKu10vKt0x+qsKjeKamUXxLCbT2DXXWya%2f9szAOlegq3S0yelnbThF5vuv4JVMfIQgjCxdC9VehKuRbuj0zw%2f2sq8TjQ8KMw2N2lzZmUGQlmU5lADhCY1LvZLvPykfOgNno3CioZwRo75Tz5k951mkJgil5Uie8idSV9LvW06KAjGTWdRNOFG8%3d~-1; bm_s=YAAQVXXTFwyCRPmbAQAAwycp/gSoxdmNnY+VUfsf/x4jyI59zLX7nxpftEqS2/SkI3VGY+OUWmOEH21hY0kMuPhAdXVYejXyS4L2RqtncZiyWoATMPscXt4F4JRmDPWhV8mBYJgjP96bUT/V8iHL0Knb7YQZ2TAkUDJ4g1NixMSHCbla6XkzIiDOyeF8pLwPXev5gD7F7i/9u3Em+08Hjp5Swy6EGXViMG6MAYcOWDQie7aBSqYD/W4iB2Ym63s/jvcNu8WWcIFRqa6T2gGiBM3aAceUvyXne5aRl3tFcC2rvbgtr42iNL2jBA90OYxwBEaFBx7Nd/I27d+vcXSJu9+y36CzGjMezVyil/4DOGB1Fdo1ex3ic0V4J0R85ds6tpExWUdOn4HK9wOzq7if2mpdLAR1aGVtdozL87kFaPtJZz7srE67yxYy7sB+8DHgQuD3QNQmwexOg0pqmfNzYfBdLCajzgBZczqOBAOLDP4QfGFfHXVqEuwnrfA2U9rNU//LOYctX+WIjr+dd6XicK6axxibFyJLUgARz9kNOeKe9W+b4camGeMrh27IgP4yAhf0EJHsrAjEuew=; bm_so=3FB290B652EA9190B0FD0D47ABE620BE3799DF0498B00D115D265BDBBE2540D9~YAAQVXXTFw2CRPmbAQAAwycp/ga5bgRXRaLp6/P/z1dRvcifzl5AyDRDzbwjxvVeN33LE89ENOirm3UItjxo9i27L1HB7urITZ6+tHJ2/CR9IbrlQ2mQUmLmSVfDQbK2axdUmLpXlDAQuzKLRiD2vwkkLv+rtAW4doCrsynP5Zf+SOKErOIpzkO5BP8DElaesIcLSLHYK6eBT8I69YK3h3hbzki2Wh/g+c8vuGED1vu+nLSKts2KlIZkaSuMcvcuWlVTrU5dYvCEOk6HaaTBy/WJDt2WM2EyigvC+8mDRo+ZcPEOSeMzjiA0TgrsKHIGeNvQIYwSy68nIBm7b1tbi0I7dn8m+u8M66QzNVAxNKEx1sewZWoaZVzchRIL87W9MS9+6XWlUCf+2nM840MgDlSpyY2IjfJFP1DlXkjt2MBUxb51vqpgZ/NiZ4FhTomSeDM/2kNGB2QVBJunwvQD; bm_sv=1E7CE6DFEF00FD6ABF4D51AEA28CC676~YAAQVXXTFw6CRPmbAQAAwycp/h4aet5wWuGw2ehCBZbI3AZ842jbVZ454t/TdAuPUspBiMz7fkpGCq0nS2F7+jS2ewF1B73RDcaqsr0sIKQHahbpHrmJ50bJYDkwmsfVBV5UbjaCk72giLtRFojk9rnQdlmeLFda2V5PoTw7nu6OhXLG6/cYqKM5BwjwiQ17y7o9Jb5di9ct2pUIygm7TRs/TvM8DhimHjNiScbg/P19l1wUGu6FVJgbCq/M5t12skA=~1; bm_sz=54B1C31EC60E1CD843FF594ED0E9104E~YAAQVXXTFw+CRPmbAQAAwycp/h4OWCX7PPd/H+iTeY3J96/P+7CtZ6kF1c5s2u/cjtQOd8YpyOzqOq8o1X8WrsegDr4CaMa89fgYeZlxNDdzm0ZPxx3shMULZTWS0ekXcM5fRc2Jc1jmEmXNCKUsRryP78KfV7qYkzQjwOlGYEgYh2WNdvrjaRExJVU7cSAzIA30f+7HS+5BvkX0MQ9sPOKbnWIxYHOvI35WSc3uHBMBxgrP8N1dRfYUp0kTPDpS4BIO/7YEHJgtLppr3tbU+q1nOimf+BiVnSS/lk98JO2lfRyPy5HBViKq3Z697d0ZvDM4chbtLhl0IAVFU80prebcsCSTOr9SSo/SFv7VwxKatDAACQmJXRd0aMIw0cmtPhxCUKMjsk3QmUxoM/g1vX8HKS7QJl7j0bDzkuVkNGTyLC7zqhpnqurJoS/ugc5nCWdcZCwXAR8lEJ1oNrQUQ6gzpjRPqdDnRTb9LlTX11Tomn93VNq93FnRggpOWt5dQUHFuBBZicdTOVgVw2vGAGLUaDpkqCbFvM5SGeOpUDO0Vh4QyUvdqH9TGDGr2jPjU1ACpZMtfa4LzQkP~4404547~3355206",
                        },
                        impersonate="chrome124",
                    ).text,
                )
            ]
        ),
        "이미지를 찾을 수 없습니다.",
    )


print(get_single_product_image())
