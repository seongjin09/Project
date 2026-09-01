"""
감성 분석 모델 학습 스크립트

수정 사항:
  - ZeroDivisionError 방지 로직 추가 (데이터 0개일 때 중단)
  - 모델 저장 파일명 변경: logistic_regression_sentiment.joblib
  - NaN 값 처리를 위한 pd.notna() 도입
"""

import os
import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
from matplotlib import font_manager, rc
import platform
import sys
import warnings

# 환경 감지 유틸리티 import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.environment import is_colab

# matplotlib 폰트 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
    KOREAN_FONT = "Malgun Gothic"
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
    KOREAN_FONT = "AppleGothic"
else:  # 리눅스 (예: Google Colab, Ubuntu)
    # 사용 가능한 한글 폰트 찾기
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    korean_fonts = ["NanumGothic", "NanumBarunGothic", "DejaVu Sans", "Liberation Sans"]

    selected_font = None
    for font in korean_fonts:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font:
        plt.rc("font", family=selected_font)
        KOREAN_FONT = selected_font
    else:
        # 폰트를 찾을 수 없으면 기본 폰트 사용 (경고 없이)
        KOREAN_FONT = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

# ========== 학습할 조합 선택 ==========
# 1) 벡터 타입 선택 (None이면 전부 사용)
#    사용 가능: "word2vec_sentiment", "bert_sentiment", "roberta_sentiment", "koelectra_sentiment"
VECTOR_TYPES_TO_USE = [
    # "bert_sentiment",
    "roberta_sentiment",
    # "koelectra_sentiment",
]  # roberta_sentiment만 사용
# VECTOR_TYPES_TO_USE = None  # 전부 사용하려면 None

# 2) ML 모델 선택
#    사용 가능: "Logistic", "DecisionTree","LightGBM", "XGBoost","RandomForest", "Voting", "Stacking","SVM"
ML_MODELS_TO_USE = [
    # "Logistic",
    # "LightGBM",
    "XGBoost",
    # "DecisionTree",
    # "RandomForest",
    # "Voting",
    # "Stacking",
    # "SVM",
]


def load_review_data(partitioned_reviews_dir, finetune_ids_path=None):
    print("======================================================================")
    print(f"전처리된 리뷰 데이터 로드 중: {partitioned_reviews_dir}")

    # Hive 파티셔닝: category=*/data.parquet 패턴
    parquet_files = glob.glob(
        os.path.join(partitioned_reviews_dir, "category=*", "data.parquet")
    )

    if not parquet_files:
        print("[오류] Parquet 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    # 파인튜닝에 사용된 ID 로드 (있는 경우)
    finetune_id_strings = set()
    if finetune_ids_path and os.path.exists(finetune_ids_path):
        print(f"\n파인튜닝 사용 ID 로드 중: {finetune_ids_path}")
        finetune_df = pd.read_csv(finetune_ids_path)
        print(f"  - CSV 컬럼: {list(finetune_df.columns)}")
        print(f"  - CSV 행 수: {len(finetune_df):,}개")
        if len(finetune_df) > 0:
            print(
                f"  - 샘플 CSV product_id: '{finetune_df['product_id'].iloc[0]}' (타입: {type(finetune_df['product_id'].iloc[0])})"
            )
            print(
                f"  - 샘플 CSV id: '{finetune_df['id'].iloc[0]}' (타입: {type(finetune_df['id'].iloc[0])})"
            )
        # 벡터화를 위해 "product_id\x00id" 형식의 문자열 세트로 저장
        finetune_id_strings = {
            f"{pid}\x00{rid}"
            for pid, rid in zip(
                finetune_df["product_id"],
                finetune_df["id"],
            )
        }
        print(f"✓ 제외할 ID: {len(finetune_id_strings):,}개")
        if len(finetune_id_strings) > 0:
            sample_id = list(finetune_id_strings)[0]
            print(f"  - 샘플 결합 ID: {repr(sample_id)}")  # repr()로 변경
    else:
        print(f"\n⚠️  파인튜닝 ID 파일 없음: {finetune_ids_path}")

    all_dfs = []  # 딕셔너리 리스트 대신 DataFrame 리스트 사용
    total_loaded = 0
    total_excluded = 0

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            category = os.path.basename(os.path.dirname(file_path)).replace(
                "category=", ""
            )
            total_loaded += len(df)

            # 파인튜닝 사용 ID 제외 (벡터화 방식으로 최적화)
            if finetune_id_strings:
                before_count = len(df)
                # product_id, id를 결합한 문자열 생성 (apply로 명시적 결합)
                current_ids = df.apply(
                    lambda row: f"{row['product_id']}\x00{row['id']}", axis=1
                )

                # 첫 번째 파일에서만 디버깅 정보 출력
                if total_loaded == before_count and len(df) > 0:
                    print(f"\n  [디버깅 - {category}]")
                    print(
                        f"    데이터 product_id: '{df['product_id'].iloc[0]}' (타입: {type(df['product_id'].iloc[0])})"
                    )
                    print(
                        f"    데이터 id: '{df['id'].iloc[0]}' (타입: {type(df['id'].iloc[0])})"
                    )
                    print(f"    결합 ID: {repr(current_ids.iloc[0])}")
                    print(
                        f"    매칭 여부: {current_ids.iloc[0] in finetune_id_strings}"
                    )
                    # 바이트 레벨 확인으로 null byte 존재 여부 검증
                    data_id_bytes = (
                        current_ids.iloc[0].encode("unicode_escape").decode("ascii")
                    )
                    csv_sample = list(finetune_id_strings)[0]
                    csv_id_bytes = csv_sample.encode("unicode_escape").decode("ascii")
                    print(f"    데이터 ID (바이트): {data_id_bytes}")
                    print(f"    CSV ID (바이트):    {csv_id_bytes}")
                    print(
                        f"    결합 ID 길이: {len(current_ids.iloc[0])} (예상: {len(df['product_id'].iloc[0]) + 1 + len(str(df['id'].iloc[0]))})"
                    )
                    # CSV 샘플 (이중 repr 제거)
                    print(f"    CSV 샘플 5개: {list(finetune_id_strings)[:5]}")
                    print()

                df = df[~current_ids.isin(finetune_id_strings)]
                excluded = before_count - len(df)
                total_excluded += excluded
                print(f"  - {category}: {len(df):,}개 리뷰 (제외: {excluded:,}개)")
            else:
                print(f"  - {category}: {len(df):,}개 리뷰")

            # 필요한 컬럼만 선택 (메모리 절약)
            needed_cols = [
                "product_id",
                "id",
                "label",
                "word2vec_sentiment",
                "bert_sentiment",
                "roberta_sentiment",
                "koelectra_sentiment",
            ]
            # 존재하는 컬럼만 선택
            existing_cols = [col for col in needed_cols if col in df.columns]
            all_dfs.append(df[existing_cols])

        except Exception as e:
            print(f"파일 로드 오류: {file_path} - {e}")

    # DataFrame들을 하나로 합침 (딕셔너리 변환보다 메모리 효율적)
    final_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n✓ 총 로드: {total_loaded:,}개")
    if finetune_id_strings:
        print(f"✓ 파인튜닝 ID 제외: {total_excluded:,}개")
        print(f"✓ ML 학습용 데이터: {len(final_df):,}개")
    else:
        print(f"✓ 총 {len(final_df):,}개 리뷰 로드 완료")
    return final_df


def prepare_training_data(df):
    print("\n학습 데이터 준비 중...")

    if df.empty:
        print("[경고] 빈 DataFrame입니다.")
        return {}

    # 사용 가능한 모델 타입 자동 감지
    available_models = set()
    for key in df.columns:
        # word2vec_sentiment, bert_sentiment, roberta_sentiment, koelectra_sentiment 등 벡터 필드 감지
        if key in [
            "word2vec_sentiment",
            "bert_sentiment",
            "roberta_sentiment",
            "koelectra_sentiment",
        ]:
            available_models.add(key)

    print(f"\n[감지된 모델 타입]")
    print(f"  - 사용 가능: {sorted(available_models)}")

    # 데이터 샘플로 구조 확인
    print(f"\n[데이터 구조 확인]")
    print(f"  - 전체 컬럼: {list(df.columns)}")
    print(f"  - 총 행 수: {len(df):,}개")
    print(f"  - label 존재: {'label' in df.columns}")

    for model_name in available_models:
        non_null_count = df[model_name].notna().sum()
        print(f"  - {model_name} 비어있지 않은 행: {non_null_count:,}개")

    # 각 벡터 타입별로 데이터 수집 (반복문 대신 벡터화 연산)
    results = {}
    for model_name in sorted(available_models):
        # label과 벡터가 모두 있는 행만 선택
        valid_df = df[df["label"].notna() & df[model_name].notna()].copy()

        if len(valid_df) == 0:
            print(f"\n[건너뜀] {model_name.upper()}: 유효한 데이터 없음")
            results[model_name] = (np.array([]), np.array([]))
            continue

        # 벡터를 numpy 배열로 변환 (반복문보다 100배 빠름)
        X = np.stack(valid_df[model_name].values)
        y = valid_df["label"].values.astype(int)

        count = len(y)
        pos = np.sum(y)
        neg = count - pos

        print(f"\n✓ {model_name.upper()} 데이터: {count:,}개")
        print(f"  - 긍정: {pos:,}개 ({pos/count*100:.1f}%)")
        print(f"  - 부정: {neg:,}개 ({neg/count*100:.1f}%)")
        print(f"  - 벡터 차원: {X.shape[1]}")
        results[model_name] = (X, y)

    return results


def get_model_dictionary():
    """비교할 ML 모델들을 정의합니다."""
    from sklearn.linear_model import SGDClassifier
    import torch

    # GPU 사용 가능 여부 확인 (Colab 환경에서만)
    use_gpu = is_colab() and torch.cuda.is_available()

    if use_gpu:
        print("\n🚀 GPU 모드 활성화 (XGBoost, LightGBM에서 GPU 사용)")
    else:
        print("\n💻 CPU 모드 (로컬 환경 또는 GPU 없음)")

    # 1. 조기 종료 가능한 Logistic Regression (SGD 방식)
    lr = SGDClassifier(
        loss="log_loss",  # 로지스틱 회귀와 동일한 손실 함수
        early_stopping=True,  # 조기 종료 활성화
        validation_fraction=0.1,  # 10%를 검증용으로 사용
        n_iter_no_change=5,  # 5번 동안 안 변하면 정지
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )

    # 2. XGBoost & LightGBM (GPU/CPU 자동 선택)
    if use_gpu:
        xgb = XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            device="cuda",  # GPU 사용 (XGBoost 3.1+에서 gpu_id 대신 device 사용)
            n_jobs=-1,
        )
        lgbm = LGBMClassifier(
            class_weight="balanced",
            random_state=42,
            device="gpu",  # GPU 사용
            verbose=-1,
        )
    else:
        xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
        lgbm = LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )

    # 3. 나머지는 조기 종료 미지원 (기존 그대로 유지)
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    dt = DecisionTreeClassifier(class_weight="balanced", random_state=42, max_depth=10)
    svc = SVC(probability=True, class_weight="balanced", random_state=42)

    # 4. 앙상블 모델 (Voting, Stacking)
    # Voting: 여러 모델의 예측을 투표로 결합
    voting = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("dt", dt),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=30, class_weight="balanced", random_state=42, n_jobs=-1
                ),
            ),
        ],
        voting="soft",  # 확률 기반 투표
        n_jobs=-1,
    )

    # Stacking: 베이스 모델 위에 메타 학습자 추가
    stacking = StackingClassifier(
        estimators=[
            ("lr", lr),
            ("dt", dt),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=30, class_weight="balanced", random_state=42, n_jobs=-1
                ),
            ),
        ],
        final_estimator=LogisticRegression(class_weight="balanced", random_state=42),
        cv=3,  # 3-fold 교차검증
        n_jobs=-1,
    )

    return {
        "Logistic": lr,
        "RandomForest": rf,
        "DecisionTree": dt,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "SVM": svc,
        "Voting": voting,
        "Stacking": stacking,
    }


def train_model(X_train, y_train, ml_model=None):
    """모델 학습

    Args:
        X_train: 학습 데이터
        y_train: 학습 레이블
        ml_model: 사용할 ML 모델 (None이면 기본 LogisticRegression)
    """
    import time
    from sklearn.preprocessing import StandardScaler

    start_time = time.time()

    if ml_model is None:
        # 기본값: Logistic Regression
        model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )
        print("\n[Logistic Regression] 모델 학습 중...")
        model.fit(X_train, y_train)
    else:
        model = ml_model
        model_name = type(model).__name__
        print(f"\n[{model_name}] 모델 학습 중...")

        # SGDClassifier는 스케일에 민감하므로 StandardScaler 적용
        if model_name == "SGDClassifier":
            print("  → StandardScaler 적용 중...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.scaler = scaler  # 추론 시 사용하기 위해 모델에 scaler 저장
            model.fit(X_train_scaled, y_train)
        # 조기 종료를 위한 내부 검증 데이터 분리 (10%)
        elif model_name == "XGBClassifier":
            # 조기 종료는 버전별 API가 다른 문제가 있으므로 제외
            model.fit(X_train, y_train)
        elif model_name == "LGBMClassifier":
            # 조기 종료는 버전별 API가 다른 문제가 있으므로 제외
            model.fit(X_train, y_train)
        elif model_name in ["VotingClassifier", "StackingClassifier"]:
            # 앙상블 모델은 내부적으로 여러 모델 학습
            print(f"  → 베이스 모델들을 학습 중...")
            model.fit(X_train, y_train)
        else:
            # RF/DT/SVM 등은 전체 데이터로 학습
            model.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"✓ 모델 학습 완료 ({train_time:.1f}초)")
    return model, train_time


def evaluate_model(model, X_test, y_test, output_dir, model_name="model"):
    print("\n모델 평가 중...")

    # SGDClassifier의 경우 저장된 scaler로 변환
    model_type = type(model).__name__
    if model_type == "SGDClassifier" and hasattr(model, "scaler"):
        X_test_scaled = model.scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # 긍정 클래스 확률

    # average=None으로 각 클래스(0:부정, 1:긍정)별 점수를 얻음
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])
    )

    # ============ 기본 메트릭 ============
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")  # 클래스 평균
    f1_weighted = f1_score(y_test, y_pred, average="weighted")  # 가중 평균
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\n" + "=" * 70)
    print("기본 성능 메트릭")
    print("=" * 70)
    print(f"정확도 (Accuracy):         {accuracy:.4f}")
    print(f"F1 Score (Macro Avg):      {f1_macro:.4f}")
    print(f"F1 Score (Weighted Avg):   {f1_weighted:.4f}")
    print(f"Matthews Corr Coef:        {mcc:.4f}")

    # ============ 클래스별 상세 성능 (핵심!) ============
    print("\n" + "=" * 70)
    print("클래스별 상세 성능 (Class-wise Performance)")
    print("=" * 70)
    print(
        f"{'클래스':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}"
    )
    print("-" * 70)
    print(
        f"{'부정(0)':<10} {precision_per_class[0]:<12.4f} {recall_per_class[0]:<12.4f} {f1_per_class[0]:<12.4f} {support_per_class[0]:<10,}"
    )
    print(
        f"{'긍정(1)':<10} {precision_per_class[1]:<12.4f} {recall_per_class[1]:<12.4f} {f1_per_class[1]:<12.4f} {support_per_class[1]:<10,}"
    )
    print("\n[해석]")
    print(
        f"  • 부정 리뷰 F1: {f1_per_class[0]:.4f} - 부정 리뷰를 얼마나 정확하게 분류하는가"
    )
    print(
        f"  • 긍정 리뷰 F1: {f1_per_class[1]:.4f} - 긍정 리뷰를 얼마나 정확하게 분류하는가"
    )
    print(
        f"  • 부정 Recall: {recall_per_class[0]:.4f} - 실제 부정 리뷰 중 몇 %를 찾아냈는가"
    )
    print(
        f"  • 긍정 Recall: {recall_per_class[1]:.4f} - 실제 긍정 리뷰 중 몇 %를 찾아냈는가"
    )

    # ============ 분류 리포트 ============
    print("\n" + "=" * 70)
    print("분류 리포트 (Classification Report)")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["부정(0)", "긍정(1)"]))

    # ============ 혼동 행렬 상세 분석 ============
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 70)
    print("혼동 행렬 상세 분석")
    print("=" * 70)
    print(f"True Negative (TN):    {tn:,}개 - 부정을 부정으로 맞춤")
    print(f"False Positive (FP):   {fp:,}개 - 부정을 긍정으로 잘못 예측")
    print(f"False Negative (FN):   {fn:,}개 - 긍정을 부정으로 잘못 예측")
    print(f"True Positive (TP):    {tp:,}개 - 긍정을 긍정으로 맞춤")
    print(f"\nSpecificity (특이도):  {tn/(tn+fp):.4f} - 부정 클래스 탐지 성능")
    print(f"Sensitivity (민감도):  {tp/(tp+fn):.4f} - 긍정 클래스 탐지 성능")

    # ============ ROC Curve & AUC ============
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print("\n" + "=" * 70)
    print("ROC & AUC")
    print("=" * 70)
    print(f"AUC (Area Under ROC): {roc_auc:.4f}")

    # ============ Precision-Recall Curve ============
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    print(f"Average Precision:    {avg_precision:.4f}")

    # ============ 시각화 ============
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["부정", "긍정"],
        yticklabels=["부정", "긍정"],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylabel("실제", fontsize=12)
    axes[0, 0].set_xlabel("예측", fontsize=12)

    # 2. ROC Curve
    axes[0, 1].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})"
    )
    axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel("False Positive Rate", fontsize=12)
    axes[0, 1].set_ylabel("True Positive Rate", fontsize=12)
    axes[0, 1].set_title("ROC Curve", fontsize=14, fontweight="bold")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)

    # 3. Precision-Recall Curve
    axes[1, 0].plot(
        recall, precision, color="blue", lw=2, label=f"PR (AP = {avg_precision:.3f})"
    )
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel("Recall", fontsize=12)
    axes[1, 0].set_ylabel("Precision", fontsize=12)
    axes[1, 0].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(alpha=0.3)

    # 4. 확률 분포 히스토그램
    axes[1, 1].hist(
        y_proba[y_test == 0], bins=50, alpha=0.5, label="부정(실제)", color="red"
    )
    axes[1, 1].hist(
        y_proba[y_test == 1], bins=50, alpha=0.5, label="긍정(실제)", color="green"
    )
    axes[1, 1].set_xlabel("예측 확률 (긍정 클래스)", fontsize=12)
    axes[1, 1].set_ylabel("빈도", fontsize=12)
    axes[1, 1].set_title("예측 확률 분포", fontsize=14, fontweight="bold")
    axes[1, 1].legend(loc="upper center")
    axes[1, 1].grid(alpha=0.3)

    # 5. 빈 공간 활용
    axes[0, 2].axis("off")

    # 6. 빈 공간에 메트릭 요약 표시
    metrics_text = f"""성능 요약
    
정확도: {accuracy:.4f}
F1 (Macro): {f1_macro:.4f}
F1 (Weighted): {f1_weighted:.4f}
MCC: {mcc:.4f}
AUC: {roc_auc:.4f}
Avg Precision: {avg_precision:.4f}

부정 F1: {f1_per_class[0]:.4f}
긍정 F1: {f1_per_class[1]:.4f}

TN: {tn:,}  FP: {fp:,}
FN: {fn:,}  TP: {tp:,}

Specificity: {tn/(tn+fp):.4f}
Sensitivity: {tp/(tp+fn):.4f}"""

    axes[1, 2].text(
        0.1,
        0.5,
        metrics_text,
        fontsize=11,
        verticalalignment="center",
        family=KOREAN_FONT,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    axes[1, 2].axis("off")

    plt.tight_layout()

    # 파일명에 모델 이름 포함
    eval_path = os.path.join(output_dir, f"model_evaluation_{model_name}.png")
    plt.savefig(eval_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ 평가 결과 시각화 저장: {eval_path}")

    # ============ 임계값 분석 ============
    print("\n" + "=" * 70)
    print("임계값별 성능 (상위 5개)")
    print("=" * 70)
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 70)

    # 다양한 임계값에서 성능 계산
    threshold_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in threshold_candidates:
        y_pred_custom = (y_proba >= threshold).astype(int)
        prec = precision_score_custom(y_test, y_pred_custom)
        rec = recall_score_custom(y_test, y_pred_custom)
        f1_custom = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{threshold:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1_custom:>10.4f}")

    # 성능 메트릭 반환
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_neg": f1_per_class[0],
        "f1_pos": f1_per_class[1],
        "mcc": mcc,
        "auc": roc_auc,
        "avg_precision": avg_precision,
        "sensitivity": sensitivity,
    }


def precision_score_custom(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall_score_custom(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def main():
    print("=" * 70)
    print(f"감성 분석 모델 학습 (사용 모델: {', '.join(ML_MODELS_TO_USE)})")
    print("=" * 70)

    # 경로 설정 (Colab 환경 고려)
    if is_colab():
        BASE_DIR = "/content"
        PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/new_processed_data")
        MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")
        FINETUNE_IDS_PATH = os.path.join(BASE_DIR, "data/finetune_used_ids.csv")
    else:
        BASE_DIR = "./data"
        PROCESSED_DATA_DIR = "./data/new_processed_data"
        MODEL_OUTPUT_DIR = "./models"
        FINETUNE_IDS_PATH = os.path.join(BASE_DIR, "finetune_used_ids.csv")

    PARTITIONED_REVIEWS_DIR = os.path.join(PROCESSED_DATA_DIR, "partitioned_reviews")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # 1. 데이터 로드 (파인튜닝 사용 ID 제외)
    df = load_review_data(PARTITIONED_REVIEWS_DIR, FINETUNE_IDS_PATH)
    if df.empty:
        print("[중단] 로드된 리뷰 데이터가 없습니다.")
        return

    # 2. 학습 데이터 준비 (모든 모델 자동 감지)
    model_data = prepare_training_data(df)
    if not model_data:
        print("\n[중단] 학습 가능한 데이터가 없습니다.")
        return

    # 성능 비교를 위한 결과 저장
    performance_results = []

    # ML 모델 딕셔너리 가져오기
    ml_models_dict = get_model_dictionary()

    # 선택된 ML 모델만 필터링 (ML_MODELS_TO_USE 순서 유지)
    selected_ml_models = {}
    for name in ML_MODELS_TO_USE:
        if name in ml_models_dict:
            selected_ml_models[name] = ml_models_dict[name]
        else:
            print(f"⚠️  경고: '{name}' 모델을 찾을 수 없습니다. 건너뜁니다.")

    if not selected_ml_models:
        print("\n[경고] ML_MODELS_TO_USE가 비어있거나 유효하지 않은 모델명입니다.")
        print(f"사용 가능한 모델: {list(ml_models_dict.keys())}")
        return

    print(f"\n선택된 ML 모델: {list(selected_ml_models.keys())}")

    # 벡터 타입 필터링
    if VECTOR_TYPES_TO_USE is not None:
        available_vectors = {
            vname: vdata
            for vname, vdata in model_data.items()
            if vname in VECTOR_TYPES_TO_USE
        }
        if not available_vectors:
            print(f"\n[경고] VECTOR_TYPES_TO_USE에 지정된 벡터가 데이터에 없습니다.")
            print(f"  - 요청: {VECTOR_TYPES_TO_USE}")
            print(f"  - 사용 가능: {list(model_data.keys())}")
            return
        print(f"선택된 벡터 타입: {list(available_vectors.keys())}")
    else:
        available_vectors = model_data
        print(f"전체 벡터 타입 사용: {list(available_vectors.keys())}")

    # 학습할 조합 개수 미리 계산
    total_combinations = len(available_vectors) * len(selected_ml_models)
    print(f"\n💡 총 {total_combinations}개 조합 학습 예정")
    print(f"   ({len(available_vectors)}개 벡터 × {len(selected_ml_models)}개 ML 모델)")

    # 3. 각 벡터 타입별로 학습 및 평가
    for vector_name in sorted(available_vectors.keys()):
        X, y = available_vectors[vector_name]

        if X.size == 0:
            print(f"\n[건너뜀] {vector_name.upper()}: 데이터 없음")
            continue

        print("\n" + "=" * 100)
        print(f"{vector_name.upper()} 벡터 기반 모델 학습")
        print("=" * 100)

        print("\n데이터 분할 중...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✓ 훈련: {len(X_train):,}개 / 테스트: {len(X_test):,}개")

        # 각 ML 모델별로 학습
        for ml_model_name, ml_model in selected_ml_models.items():
            print("\n" + "-" * 80)
            print(f"[{vector_name.upper()}] × [{ml_model_name}] 조합")
            print("-" * 80)

            # 모델 학습
            model, train_time = train_model(X_train, y_train, ml_model)

            # 모델 평가
            combined_name = f"{vector_name}_{ml_model_name}"
            performance = evaluate_model(
                model,
                X_test,
                y_test,
                MODEL_OUTPUT_DIR,
                model_name=combined_name,
            )

            # 성능 결과 저장
            performance["vector_name"] = vector_name
            performance["ml_model_name"] = ml_model_name
            performance["combined_name"] = combined_name
            performance["train_time"] = train_time
            performance_results.append(performance)

            # 모델 저장
            model_path = os.path.join(MODEL_OUTPUT_DIR, f"{combined_name}.joblib")
            joblib.dump(model, model_path)
            print(f"\n✓ 모델 저장 완료: {model_path}")

            # ========== 누적 성능 비교 (모델 하나 끝날 때마다) ==========
            print("\n" + "=" * 140)
            print(f"누적 성능 비교 ({len(performance_results)}개 모델 완료)")
            print("=" * 140)

            # 헤더
            header = f"{'Vector':<12} {'ML Model':<15} {'Accuracy':>9} {'F1 Macro':>9} {'F1 Neg':>9} {'F1 Pos':>9} {'AUC':>9} {'MCC':>9} {'Sens':>9} {'Train Time':>12}"
            print(header)
            print("-" * 140)

            # MCC 기준으로 정렬
            sorted_current = sorted(
                performance_results, key=lambda x: x["mcc"], reverse=True
            )

            # 각 모델 결과
            for result in sorted_current:
                row = (
                    f"{result['vector_name']:<12} "
                    f"{result['ml_model_name']:<15} "
                    f"{result['accuracy']:>9.4f} "
                    f"{result['f1_macro']:>9.4f} "
                    f"{result['f1_neg']:>9.4f} "
                    f"{result['f1_pos']:>9.4f} "
                    f"{result['auc']:>9.4f} "
                    f"{result['mcc']:>9.4f} "
                    f"{result['sensitivity']:>9.4f} "
                    f"{result['train_time']:>11.1f}s"
                )
                print(row)

            # 현재까지 최고 성능
            if len(performance_results) > 0:
                best_current = max(performance_results, key=lambda x: x["mcc"])
                print("\n" + "-" * 140)
                print(
                    f"현재까지 최고 MCC: {best_current['combined_name']} (MCC={best_current['mcc']:.4f})"
                )
                print("=" * 140)

    # 4. 최종 성능 비교 표 출력
    if performance_results:
        print("\n" + "=" * 140)
        print("🏆 최종 모델 성능 비교 (전체 요약)")
        print("=" * 140)

        # 헤더
        header = f"{'Vector':<12} {'ML Model':<15} {'Accuracy':>9} {'F1 Macro':>9} {'F1 Neg':>9} {'F1 Pos':>9} {'AUC':>9} {'MCC':>9} {'Sens':>9} {'Train Time':>12}"
        print(header)
        print("-" * 140)

        # MCC 기준으로 정렬
        sorted_results = sorted(
            performance_results, key=lambda x: x["mcc"], reverse=True
        )

        # 각 모델 결과
        for result in sorted_results:
            row = (
                f"{result['vector_name']:<12} "
                f"{result['ml_model_name']:<15} "
                f"{result['accuracy']:>9.4f} "
                f"{result['f1_macro']:>9.4f} "
                f"{result['f1_neg']:>9.4f} "
                f"{result['f1_pos']:>9.4f} "
                f"{result['auc']:>9.4f} "
                f"{result['mcc']:>9.4f} "
                f"{result['sensitivity']:>9.4f} "
                f"{result['train_time']:>11.1f}s"
            )
            print(row)

        # 최고 성능 모델 표시
        best_acc = max(performance_results, key=lambda x: x["accuracy"])
        best_f1_macro = max(performance_results, key=lambda x: x["f1_macro"])
        best_f1_neg = max(performance_results, key=lambda x: x["f1_neg"])
        best_auc = max(performance_results, key=lambda x: x["auc"])
        best_mcc = max(performance_results, key=lambda x: x["mcc"])

        print("\n" + "-" * 140)
        print("최고 성능:")
        print(
            f"  - Accuracy:  {best_acc['combined_name']} ({best_acc['accuracy']:.4f})"
        )
        print(
            f"  - F1 Macro:  {best_f1_macro['combined_name']} ({best_f1_macro['f1_macro']:.4f})"
        )
        print(
            f"  - F1 부정:   {best_f1_neg['combined_name']} ({best_f1_neg['f1_neg']:.4f})"
        )
        print(f"  - AUC:       {best_auc['combined_name']} ({best_auc['auc']:.4f})")
        print(
            f"  - MCC:       {best_mcc['combined_name']} ({best_mcc['mcc']:.4f}) ⭐ 추천"
        )
        print("=" * 140)

    print("\n" + "=" * 70)
    print("학습 완료!")
    print(f"총 {len(performance_results)}개 모델 저장됨:")
    for result in sorted_results[:5]:  # 상위 5개만 표시
        print(
            f"  - {result['combined_name']}: sentiment_{result['combined_name']}.joblib"
        )
    if len(sorted_results) > 5:
        print(f"  ... 외 {len(sorted_results) - 5}개")
    print("=" * 70)


if __name__ == "__main__":
    main()
