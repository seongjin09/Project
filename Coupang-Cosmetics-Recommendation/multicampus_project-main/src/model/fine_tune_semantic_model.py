"""
의미 기반 벡터 학습 - SimCSE 방식 미세조정
감성 분류 모델의 한계를 극복하기 위해 문맥 유사도를 직접 학습
환경 자동 감지 (Colab A100, 일반 GPU, Mac MPS, CPU) 및 메모리 최적화 적용
"""

import os
import sys
import pandas as pd
import torch
import glob
import shutil
import time
from tqdm import tqdm
from typing import List
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

# utils 모듈 import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.environment import get_execution_mode

# WANDB 비활성화 (선택창 방지)
os.environ["WANDB_MODE"] = "disabled"

# ========== 설정 ==========
# 미세조정할 베이스 모델 (원본 사전학습 모델 사용)
BASE_MODEL_PATH = "klue/roberta-base"

# 학습 설정
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1  # SimCSE는 보통 1에폭으로도 충분함
MAX_LENGTH = 512
RANDOM_SEED = 42

# 샘플링 설정
MAX_SAMPLES = 100000  # None이면 전체 데이터 사용, 숫자 입력 시 해당 개수만큼 샘플링
MIN_TEXT_LENGTH = 20  # 최소 글자수 (너무 짧은 리뷰 제외)

# 환경별 경로 설정
exec_mode = get_execution_mode("auto")

if exec_mode == "colab":
    DATA_DIR = "/content/data/processed_data"
    OUTPUT_BASE_DIR = "/content/models/fine_tuned"
    print("[알림] Colab 환경: /content 로컬 스토리지 사용")
else:
    DATA_DIR = "./data/processed_data"
    OUTPUT_BASE_DIR = "./models/fine_tuned"

REVIEWS_DIR = os.path.join(DATA_DIR, "partitioned_reviews")


def get_env_config():
    """실행 환경을 감지하여 최적의 배치 사이즈 반환 (감성 모델 코드와 동일 구조)"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU 감지: {gpu_name}")
        if "A100" in gpu_name:
            return {
                "device": "cuda",
                "batch_size": 64,
                "fp16": True,
            }  # A100은 대용량 배치 가능
        else:
            return {"device": "cuda", "batch_size": 32, "fp16": True}

    if torch.backends.mps.is_available():
        print("✓ Mac MPS 감지")
        return {
            "device": "mps",
            "batch_size": 4,
            "fp16": False,
        }  # MPS는 메모리 고려하여 작게

    return {"device": "cpu", "batch_size": 16, "fp16": False}


def load_semantic_data():
    """Parquet 파일에서 텍스트 추출 (ZeroDivision 방어 로직 포함)"""
    print("\n" + "=" * 60)
    print("리뷰 텍스트 로딩 중...")
    print("=" * 60)

    parquet_files = glob.glob(
        os.path.join(REVIEWS_DIR, "**", "*.parquet"), recursive=True
    )
    if not parquet_files:
        raise FileNotFoundError(f"{REVIEWS_DIR}에서 데이터를 찾을 수 없습니다.")

    all_texts = []
    # 여러 컬럼명 대응
    target_cols = ["full_text", "cleaned_review", "review_text"]

    for file in tqdm(parquet_files, desc="데이터 읽기"):
        df = pd.read_parquet(file)
        found_col = next((col for col in target_cols if col in df.columns), None)

        if found_col:
            texts = df[found_col].dropna().astype(str).str.strip().tolist()
            all_texts.extend([t for t in texts if len(t) >= MIN_TEXT_LENGTH])

    if not all_texts:
        raise ValueError("학습할 텍스트가 없습니다. 컬럼명을 확인하세요.")

    print(f"✓ 최소 글자수({MIN_TEXT_LENGTH}자) 이상 텍스트: {len(all_texts):,}개")

    # 샘플링 (MAX_SAMPLES 설정된 경우)
    if MAX_SAMPLES is not None and len(all_texts) > MAX_SAMPLES:
        import random

        random.seed(RANDOM_SEED)
        all_texts = random.sample(all_texts, MAX_SAMPLES)
        print(f"✓ 샘플링 완료: {MAX_SAMPLES:,}개")

    return all_texts


def train_semantic_model(texts, config):
    """SimCSE 미세조정 실행"""
    print("\n" + "=" * 60)
    print("SimCSE 의미 기반 학습 시작")
    print(f"Device: {config['device']}, Batch: {config['batch_size']}")
    print("=" * 60)

    # 1. 데이터셋 준비 (InputExample 생성)
    train_examples = [InputExample(texts=[t, t]) for t in texts]
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=config["batch_size"]
    )

    # 2. 모델 구성 (원본 klue/roberta-base 사용)
    print(f"✓ 베이스 모델 로드: {BASE_MODEL_PATH}")
    word_embedding_model = models.Transformer(
        BASE_MODEL_PATH, max_seq_length=MAX_LENGTH
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model], device=config["device"]
    )

    # 3. Loss 설정
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 4. 학습 (fit)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1),
        optimizer_params={"lr": LEARNING_RATE},
        show_progress_bar=True,
        use_amp=False,  # 에러 방지를 위해 False 권장 (A100은 빨라서 무관)
    )

    # 5. 저장
    final_path = os.path.join(OUTPUT_BASE_DIR, "roberta_semantic_final")
    model.save(final_path)
    print(f"\n✓ 모델 저장 완료: {final_path}")
    return final_path


def main():
    print("\n" + "=" * 60)
    print(f"{'의미 기반 벡터 학습 (SimCSE) 시작':^60}")
    print("=" * 60)

    # 1. 환경 설정
    config = get_env_config()

    # 2. 데이터 로드
    try:
        texts = load_semantic_data()
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return

    # 3. 학습 실행
    try:
        final_model_path = train_semantic_model(texts, config)
    except Exception as e:
        print(f"학습 실패: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Colab 백업
    if exec_mode == "colab":
        print("\n" + "=" * 60)
        print("Google Drive 백업 시작")
        print("=" * 60)
        try:
            from google.colab import drive

            if not os.path.exists("/content/drive"):
                drive.mount("/content/drive")

            drive_path = (
                "/content/drive/MyDrive/multicampus_project_backup/models/fine_tuned"
            )
            if os.path.exists(drive_path):
                shutil.rmtree(drive_path)

            shutil.copytree(OUTPUT_BASE_DIR, drive_path)
            print(f"✓ Drive 백업 완료: {drive_path}")
        except Exception as e:
            print(f"⚠ 백업 실패: {e}")

    print("\n모든 과정이 완료되었습니다.")


if __name__ == "__main__":
    main()
