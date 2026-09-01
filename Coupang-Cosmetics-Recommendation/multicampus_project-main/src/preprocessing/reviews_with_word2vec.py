# =====================================
# 통합 파일: 전처리 + 감성 라벨링 + Word2Vec
# =====================================
import json
import os
import re
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from konlpy.tag import Okt

# =====================================
# 0️⃣ 전처리 관련 함수
# =====================================
okt = Okt()


def load_stopwords(filename="stopwords-ko.txt"):
    """불용어 로드"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stopword_path = os.path.join(base_dir, filename)

    with open(stopword_path, "r", encoding="utf-8") as f:
        stopwords = [line.strip() for line in f if line.strip()]

    return set(stopwords)


def clean_text(text: str) -> str:
    """HTML 태그 제거, 특수문자 제거"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_emotion(text: str) -> str:
    """ㅋㅋ, ㅎㅎ, ㅠㅠ 등 반복 문자 정규화"""
    text = re.sub(r"ㅋ{2,}", "ㅋ", text)
    text = re.sub(r"ㅎ{2,}", "ㅎ", text)
    text = re.sub(r"ㅠ{2,}", "ㅠ", text)
    return text


def tokenize(text: str, stopwords: set, allowed_pos=("Noun", "Verb", "Adjective")):
    """형태소 분석 + 품사 필터 + 불용어 제거"""
    tokens = []
    for word, pos in okt.pos(text, stem=True):
        if pos in allowed_pos and word not in stopwords:
            tokens.append(word)
    return tokens


def preprocess_pipeline(text: str, stopwords: set):
    """전체 전처리 파이프라인"""
    text = clean_text(text)
    text = normalize_emotion(text)
    tokens = tokenize(text, stopwords)
    return tokens


# =====================================
# 1️⃣ 감성 라벨링
# =====================================
def sentiment_label(score):
    """1이 긍정, 0이 부정, 3점 중립 제거"""
    if score >= 4:
        return 1
    elif score <= 2:
        return 0
    else:
        return None


# =====================================
# 2️⃣ Word2Vec 학습
# =====================================
def train_word2vec(tokenized_texts, vector_size=100, window=5, min_count=3):
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,  # 1은 Skip-gram, 0은 CBOW. 기본은 0인데 skip-gram이 성능이 더 좋음
    )
    return model


# =====================================
# 3️⃣ 전체 파이프라인 함수
# =====================================
def reviews_with_word2vec(data: dict, stopwords_file="stopwords-ko.txt") -> dict:
    """
    리뷰 데이터를 전처리하고 감성 라벨링 및 Word2Vec 벡터를 생성합니다.

    Args:
        data: 전처리할 JSON 데이터 (with_text)
        stopwords_file: 불용어 파일명

    Returns:
        처리된 데이터 (tokens, label, Word2Vec 추가)
    """
    stopwords = load_stopwords(stopwords_file)

    all_tokens = []
    review_indices = []  # Word2Vec 연결용

    # 4️⃣ 리뷰 전처리 & 감성 라벨링
    for p_idx, product in enumerate(tqdm(data["data"], desc="제품 처리 중")):
        for r_idx, review in enumerate(product["reviews"]["data"]):
            tokens = preprocess_pipeline(review.get("full_text", ""), stopwords)
            review["tokens"] = tokens
            review["label"] = sentiment_label(review.get("score", 3))

            if tokens and review["label"] is not None:
                all_tokens.append(tokens)
                review_indices.append((p_idx, r_idx))

    # 5️⃣ Word2Vec 학습
    if all_tokens:
        w2v_model = train_word2vec(all_tokens)

        # 리뷰 단위 Word2Vec 벡터 생성 (토큰 평균)
        for idx, (p_idx, r_idx) in enumerate(review_indices):
            tokens = data["data"][p_idx]["reviews"]["data"][r_idx]["tokens"]
            vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
            if vectors:
                avg_vector = np.mean(vectors, axis=0).tolist()
            else:
                avg_vector = []
            data["data"][p_idx]["reviews"]["data"][r_idx]["word2vec"] = avg_vector

    return data


# =====================================
# 단독 실행용 코드
# =====================================
if __name__ == "__main__":
    json_path = (
        r"/Users/choimanseon/Documents/dev/project/multicampus_project/result_오일.json"
    )
    output_path = r"/Users/choimanseon/Documents/dev/project/multicampus_project/processed_reviews_word2vec.json"

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    processed_data = reviews_with_word2vec(raw)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"완료! 파일 저장: {output_path}")
