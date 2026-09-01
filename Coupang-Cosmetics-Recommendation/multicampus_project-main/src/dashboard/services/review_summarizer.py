"""
리뷰 요약 서비스 (Hugging Face 무료 Inference API 사용)
- 상품의 긍정/부정 대표 리뷰를 LLM으로 요약
- 캐싱을 통한 중복 호출 방지
"""

import os
import streamlit as st
from typing import Optional
from huggingface_hub import InferenceClient


class ReviewSummarizer:
    """
    Hugging Face 무료 Inference API를 사용한 리뷰 요약 클래스
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.client = InferenceClient(token=self.api_token)

    def _call_api(self, prompt: str, max_tokens: int = 300) -> str:
        """HF Inference API 호출"""
        try:
            response = self.client.chat_completion(
                model=self.DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 화장품 리뷰 분석 전문가입니다. "
                            "한국어로 간결하고 핵심적인 요약을 제공합니다. "
                            "반드시 한국어로만 답변하세요."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.5,
            )
            result = response.choices[0].message.content.strip()

            # 후처리: 장점/단점/추천 소비자 키워드 앞에 줄바꿈 추가 (HTML <br> 사용)
            import re

            result = re.sub(r"\s*단점:", "<br>단점:", result)
            result = re.sub(r"\s*추천 소비자:", "<br>추천 소비자:", result)

            return result
        except Exception as e:
            error_msg = str(e)
            print(f"[ReviewSummarizer] 요약 생성 실패: {error_msg}")
            if "503" in error_msg:
                return "AI 모델이 로딩 중입니다. 잠시 후 다시 시도해주세요."
            if "429" in error_msg:
                return "요청이 너무 많습니다. 잠시 후 다시 시도해주세요."
            return "요약 생성 중 오류가 발생했습니다."

    def summarize_reviews(
        self,
        product_name: str,
        keywords: list,
        positive_reviews: list[str],
        negative_reviews: list[str],
    ) -> str:
        """
        긍정/부정 리뷰를 종합 요약

        Args:
            product_name: 상품명
            keywords: 대표 키워드 리스트
            positive_reviews: 긍정 대표 리뷰 텍스트 리스트
            negative_reviews: 부정 대표 리뷰 텍스트 리스트

        Returns:
            str: 요약 문장
        """
        # 리뷰 텍스트 정리 (최대 5개씩, 각 200자 제한)
        pos_texts = [r[:200] for r in positive_reviews[:5]] if positive_reviews else []
        neg_texts = [r[:200] for r in negative_reviews[:5]] if negative_reviews else []

        if not pos_texts and not neg_texts:
            return "리뷰 데이터가 부족하여 요약을 생성할 수 없습니다."

        # 키워드 정리
        kw_str = ", ".join(keywords[:5]) if keywords else "없음"

        # 프롬프트 구성
        prompt = f"""'{product_name}' 화장품에 대한 리뷰를 분석하여 한국어로 요약해주세요.

[대표 키워드] {kw_str}

[긍정 리뷰]
{chr(10).join(f'- {r}' for r in pos_texts) if pos_texts else '- 없음'}

[부정 리뷰]
{chr(10).join(f'- {r}' for r in neg_texts) if neg_texts else '- 없음'}

위 리뷰를 바탕으로 반드시 아래 형식에 맞춰 한국어로 요약해주세요:

장점: [이 제품의 가장 큰 장점을 1~2문장으로]
단점: [사용자들이 언급한 단점이나 주의사항을 1~2문장으로, 없으면 '특별히 언급된 단점 없음']
추천 소비자: [어떤 사용자에게 추천하는지 1문장으로]

반드시 위 형식(장점:/단점:/추천 소비자:)을 지켜서 작성."""

        return self._call_api(prompt)


@st.cache_data(ttl=600, show_spinner=False)
def get_cached_summary(
    product_name: str,
    keywords_str: str,
    positive_reviews_str: str,
    negative_reviews_str: str,
) -> str:
    """
    캐싱된 리뷰 요약 조회/생성

    Args:
        product_name: 상품명
        keywords_str: 키워드 (쉼표 구분 문자열, 캐시 키 용도)
        positive_reviews_str: 긍정 리뷰 (줄바꿈 구분 문자열)
        negative_reviews_str: 부정 리뷰 (줄바꿈 구분 문자열)
    """
    summarizer = ReviewSummarizer()

    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
    pos_reviews = [r for r in positive_reviews_str.split("\n") if r.strip()]
    neg_reviews = [r for r in negative_reviews_str.split("\n") if r.strip()]

    return summarizer.summarize_reviews(
        product_name, keywords, pos_reviews, neg_reviews
    )
