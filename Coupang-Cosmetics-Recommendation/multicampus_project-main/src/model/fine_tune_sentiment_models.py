"""
트랜스포머 모델 미세조정 (Fine-tuning)
화장품 리뷰 감성 분류를 위한 BERT, RoBERTa, KoELECTRA 미세조정
환경 자동 감지 (Colab A100, 일반 GPU, Mac MPS, CPU) 및 메모리 최적화 적용
"""

import os
import sys
import pandas as pd
import torch
import glob
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# utils 모듈 import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.environment import get_execution_mode

# ========== 설정 ==========
MODEL_CONFIGS = {
    "bert": "klue/bert-base",
    "roberta": "klue/roberta-base",
    "koelectra": "monologg/koelectra-base-v3-discriminator",
}

# 미세조정할 모델 선택
MODELS_TO_FINETUNE = ["roberta"]
# MODELS_TO_FINETUNE = ["bert", "roberta", "koelectra"]

# 기본 학습 설정 (환경에 따라 자동 조정됨)
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512
TEST_SIZE = 0.2
RANDOM_SEED = 42

# 샘플링 설정
MAX_SAMPLES = 100000  # None이면 전체 데이터 사용, 숫자 입력 시 해당 개수만큼 샘플링
MIN_TEXT_LENGTH = 20  # 최소 글자수 (너무 짧은 리뷰 제외)
BALANCE_LABELS = True  # 긍정/부정 비율 균형 맞추기

# 환경별 경로 설정
exec_mode = get_execution_mode("auto")

if exec_mode == "colab":
    BASE_DIR = "/content"
    DATA_DIR = "/content/data/processed_data"
    OUTPUT_BASE_DIR = "/content/models/fine_tuned"
    LOGS_DIR = "/content/logs/fine_tuning_sentiment"
    print("[알림] Colab 환경: /content 로컬 스토리지 사용 (빠른 I/O)")
else:
    BASE_DIR = "./data"
    DATA_DIR = "./data/processed_data"
    OUTPUT_BASE_DIR = "./models/fine_tuned"
    LOGS_DIR = "./logs/fine_tuning_sentiment"

REVIEWS_DIR = os.path.join(DATA_DIR, "partitioned_reviews")


def get_env_config():
    """
    실행 환경(GPU/MPS/CPU)을 감지하여 최적의 배치 사이즈와 설정을 반환

    - Colab A100: 배치 64, FP16 (최고 성능)
    - 일반 GPU (T4/V100): 배치 16, FP16
    - Mac MPS: 배치 1, FP32 (메모리 안정성 우선)
    - CPU: 배치 2, FP32
    """
    # 1. CUDA (NVIDIA GPU) - A100 감지
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU 감지: {gpu_name}")

        # A100 GPU인 경우 (Colab Pro+ 등)
        if "A100" in gpu_name:
            print("  → A100 최적화 모드: Batch=64, FP16=ON")
            return {
                "device": "cuda",
                "batch_size": 64,  # A100 최적값
                "acc_steps": 1,  # 누적 불필요
                "fp16": True,
                "gradient_checkpointing": False,
            }
        # 일반 GPU (T4, V100 등)
        else:
            print("  → 일반 GPU 모드: Batch=16, FP16=ON")
            return {
                "device": "cuda",
                "batch_size": 16,  # 일반 GPU용
                "acc_steps": 4,  # Total Batch = 64
                "fp16": True,
                "gradient_checkpointing": False,
            }

    # 2. MPS (Mac Apple Silicon) - 메모리 최적화 필수
    if torch.backends.mps.is_available():
        print("✓ Mac Apple Silicon (MPS) 감지")
        print("  → 메모리 절약 모드: Batch=1, FP32 (FP16 OFF)")
        return {
            "device": "mps",
            "batch_size": 1,  # 메모리 오류 방지 (극소 배치)
            "acc_steps": 64,  # Total Batch 64 효과 유지
            "fp16": False,  # FP32 사용 (안정성 우선)
            "gradient_checkpointing": True,  # 메모리 절약
        }

    # 3. CPU (기타 환경)
    print("✓ CPU 모드 감지")
    print("  → CPU 모드: Batch=2, FP32")
    return {
        "device": "cpu",
        "batch_size": 2,
        "acc_steps": 32,
        "fp16": False,
        "gradient_checkpointing": False,
    }


def load_reviews_data():
    """Parquet 파일 로드 및 전처리"""
    print("\n" + "=" * 60)
    print("리뷰 데이터 로딩 중...")
    print("=" * 60)

    parquet_files = glob.glob(
        os.path.join(REVIEWS_DIR, "**", "*.parquet"), recursive=True
    )
    if not parquet_files:
        raise FileNotFoundError(f"{REVIEWS_DIR}에서 데이터를 찾을 수 없습니다.")

    # 파일 병합
    dfs = []
    for file in parquet_files:
        dfs.append(pd.read_parquet(file))

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"전체 리뷰: {len(df_all):,}개")

    # 유효한 데이터 필터링 (라벨 있고, 텍스트 있는 경우)
    # product_id, id 컬럼 포함하여 로드
    df_filtered = df_all[
        df_all["label"].notna() & (df_all["full_text"].str.strip() != "")
    ][["product_id", "id", "full_text", "label"]].copy()

    df_filtered["label"] = df_filtered["label"].astype(int)
    print(f"유효한 리뷰: {len(df_filtered):,}개")

    # 최소 글자수 필터링
    df_filtered = df_filtered[
        df_filtered["full_text"].str.len() >= MIN_TEXT_LENGTH
    ].copy()
    print(f"최소 글자수({MIN_TEXT_LENGTH}자) 이상: {len(df_filtered):,}개")

    # 샘플링 (MAX_SAMPLES 설정된 경우)
    if MAX_SAMPLES is not None and len(df_filtered) > MAX_SAMPLES:
        print(f"\n샘플링: {len(df_filtered):,}개 → {MAX_SAMPLES:,}개")

        if BALANCE_LABELS:
            # 긍정/부정 비율 균형 맞춤 샘플링
            label_counts = df_filtered["label"].value_counts()
            print(f"  원본 레이블 분포: {dict(label_counts)}")

            # 각 레이블별 샘플링 개수
            samples_per_label = MAX_SAMPLES // len(label_counts)
            sampled_dfs = []

            for label in df_filtered["label"].unique():
                label_df = df_filtered[df_filtered["label"] == label]
                n_samples = min(len(label_df), samples_per_label)
                sampled_dfs.append(
                    label_df.sample(n=n_samples, random_state=RANDOM_SEED)
                )

            df_filtered = pd.concat(sampled_dfs, ignore_index=True).sample(
                frac=1, random_state=RANDOM_SEED
            )
            print(
                f"  샘플링 후 레이블 분포: {dict(df_filtered['label'].value_counts())}"
            )
        else:
            # 단순 랜덤 샘플링
            df_filtered = df_filtered.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED)

    print(f"최종 학습 데이터: {len(df_filtered):,}개")

    # 파인튜닝에 사용된 ID 저장 (ML 모델 학습 시 제외용)
    used_ids_path = os.path.join(BASE_DIR, "finetune_used_ids.csv")
    df_filtered[["product_id", "id"]].to_csv(used_ids_path, index=False)
    print(f"\n✓ 파인튜닝 사용 ID 저장: {used_ids_path}")
    print(f"  - 저장된 ID 개수: {len(df_filtered):,}개")

    return df_filtered


def prepare_datasets(df):
    """학습/검증 데이터셋 분리"""
    print("\n데이터셋 분리 중...")
    train_df, eval_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["label"]
    )

    print(f"  - 학습: {len(train_df):,}개")
    print(f"  - 검증: {len(eval_df):,}개")

    # Dataset 객체 생성 (full_text와 label만 사용)
    train_ds = Dataset.from_pandas(
        train_df[["full_text", "label"]].copy(), preserve_index=False
    )
    eval_ds = Dataset.from_pandas(
        eval_df[["full_text", "label"]].copy(), preserve_index=False
    )

    return train_ds, eval_ds


def tokenize_core(examples, tokenizer):
    """토큰화 수행 함수 (lambda 대신 사용)"""
    return tokenizer(
        examples["full_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(pred):
    """평가 지표 계산"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fine_tune_model(model_name, model_path, train_dataset, eval_dataset, config):
    """모델별 미세조정 실행"""
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} 학습 시작")
    print(
        f"Device: {config['device']}, Batch: {config['batch_size']}, FP16: {config['fp16']}"
    )
    print("=" * 60)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 데이터셋 토큰화 (fn_kwargs로 tokenizer 전달)
    print("토큰화 진행 중...")
    tokenized_train = train_dataset.map(
        tokenize_core,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        desc="Train Tokenization",
    )
    tokenized_eval = eval_dataset.map(
        tokenize_core,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        desc="Eval Tokenization",
    )

    # 모델 로드
    print(f"모델 로딩: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # Gradient Checkpointing (메모리 절약, MPS에서 활성화)
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        print("✓ Gradient Checkpointing 활성화 (메모리 절약)")

    model.to(config["device"])

    output_dir = os.path.join(OUTPUT_BASE_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        # 메모리 최적화 설정 적용
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["acc_steps"],
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=config["fp16"],  # A100/일반GPU: True, MPS/CPU: False
        eval_strategy="epoch",  # 최신 버전 호환 (evaluation_strategy -> eval_strategy)
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(LOGS_DIR, model_name),
        logging_steps=50,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 학습 실행
    print(
        f"\n학습 시작 (Total Effective Batch = {config['batch_size'] * config['acc_steps']})"
    )
    trainer.train()

    # 최종 평가 및 저장
    print("\n최종 평가 중...")
    eval_results = trainer.evaluate()

    print(f"\n{model_name.upper()} 최종 결과:")
    print(f"  - Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  - F1 Score: {eval_results['eval_f1']:.4f}")

    final_model_dir = os.path.join(OUTPUT_BASE_DIR, f"{model_name}_sentiment_final")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✓ 모델 저장 완료: {final_model_dir}")

    # 메모리 정리 (Mac MPS, CUDA)
    del model
    del trainer

    if config["device"] == "mps":
        torch.mps.empty_cache()
        print("✓ MPS 메모리 캐시 정리 완료")
    elif config["device"] == "cuda":
        torch.cuda.empty_cache()
        print("✓ CUDA 메모리 캐시 정리 완료")

    return eval_results


def main():
    print("\n" + "=" * 60)
    print(f"{'트랜스포머 모델 미세조정 (환경 자동 감지)':^60}")
    print("=" * 60)

    # 환경 정보 출력
    print(f"\n실행 환경: {exec_mode.upper()}")
    print(f"데이터 경로: {DATA_DIR}")
    print(f"모델 저장: {OUTPUT_BASE_DIR}")

    # 1. GPU/MPS/CPU 환경 감지
    config = get_env_config()
    print(f"\n{'설정 요약':-^60}")
    print(f"  Device: {config['device'].upper()}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Gradient Accumulation: {config['acc_steps']}")
    print(f"  Effective Total Batch: {config['batch_size'] * config['acc_steps']}")
    print(f"  FP16 (Mixed Precision): {config['fp16']}")
    print(f"  Gradient Checkpointing: {config.get('gradient_checkpointing', False)}")
    print("-" * 60)

    # 2. 데이터 준비
    try:
        df = load_reviews_data()
        train_dataset, eval_dataset = prepare_datasets(df)
    except Exception as e:
        print(f"\n[오류] 데이터 로드 실패: {e}")
        import traceback

        traceback.print_exc()
        return

    # 3. 모델 학습 루프
    results = {}
    for model_name in MODELS_TO_FINETUNE:
        if model_name not in MODEL_CONFIGS:
            print(f"\n[경고] {model_name}는 지원하지 않는 모델입니다. 건너뜁니다.")
            continue

        try:
            results[model_name] = fine_tune_model(
                model_name,
                MODEL_CONFIGS[model_name],
                train_dataset,
                eval_dataset,
                config,
            )
        except Exception as e:
            print(f"\n[오류] {model_name} 학습 실패: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 4. 결과 출력
    print("\n" + "=" * 60)
    print(f"{'최종 결과 요약':^60}")
    print("=" * 60)
    print(f"{'모델':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)

    for name, res in results.items():
        print(
            f"{name.upper():<15} "
            f"{res['eval_accuracy']:<12.4f} "
            f"{res['eval_precision']:<12.4f} "
            f"{res['eval_recall']:<12.4f} "
            f"{res['eval_f1']:<12.4f}"
        )

    # 최고 성능 모델
    if results:
        best_model = max(results.items(), key=lambda x: x[1]["eval_f1"])
        print("\n" + "=" * 60)
        print(
            f"✓ 최고 성능 모델: {best_model[0].upper()} (F1: {best_model[1]['eval_f1']:.4f})"
        )
        print("=" * 60)

    print("\n✓ 모든 미세조정 완료!")

    # ========== Colab: Google Drive 백업 ==========
    if exec_mode == "colab":
        print("\n" + "=" * 60)
        print("Google Drive 백업 시작")
        print("=" * 60)

        try:
            from google.colab import drive
            import shutil

            # Drive 마운트 (이미 마운트되어 있으면 스킵)
            if not os.path.exists("/content/drive"):
                print("\nDrive 마운트 중...")
                drive.mount("/content/drive")

            # 백업 경로 설정
            drive_backup_base = "/content/drive/MyDrive/multicampus_project_backup"
            drive_models = os.path.join(drive_backup_base, "models/fine_tuned")
            drive_logs = os.path.join(drive_backup_base, "logs/fine_tuning_sentiment")

            # 기존 백업 삭제 (덮어쓰기 위해)
            if os.path.exists(drive_models):
                print(f"\n기존 모델 백업 삭제 중: {drive_models}")
                shutil.rmtree(drive_models)
            if os.path.exists(drive_logs):
                print(f"기존 로그 백업 삭제 중: {drive_logs}")
                shutil.rmtree(drive_logs)

            # 모델 백업
            if os.path.exists(OUTPUT_BASE_DIR):
                print(f"\n미세조정 모델을 Drive로 백업 중...")
                shutil.copytree(OUTPUT_BASE_DIR, drive_models)
                backup_size = (
                    sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(drive_models)
                        for filename in filenames
                    )
                    / 1024
                    / 1024
                )
                print(f"✓ 모델 백업 완료: {drive_models}")
                print(f"  - 크기: {backup_size:.1f} MB")

            # 로그 백업
            if os.path.exists(LOGS_DIR):
                print(f"\n학습 로그를 Drive로 백업 중...")
                shutil.copytree(LOGS_DIR, drive_logs)
                print(f"✓ 로그 백업 완료: {drive_logs}")

            print("\n" + "=" * 60)
            print("Drive 백업 완료!")
            print(f"백업 위치: {drive_backup_base}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"\n[경고] Drive 백업 실패: {e}")
            print("세션 종료 시 /content 데이터가 삭제될 수 있습니다.")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
