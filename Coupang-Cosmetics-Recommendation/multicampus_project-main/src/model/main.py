"""
모델 학습 및 벡터화 파이프라인
1. 감성 분류 모델 미세조정 (fine_tune_sentiment_models)
2. 의미 기반 모델 미세조정 (fine_tune_semantic_model)
3. 감성 벡터화 (sentiment_vectorize)
4. 의미 벡터화 (semantic_vectorize)
"""

import os
import sys
import time
from datetime import datetime

# 현재 파일의 상위 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_pipeline():
    """전체 파이프라인 실행"""

    start_time = time.time()
    print("\n" + "=" * 80)
    print(f"{'모델 학습 및 벡터화 파이프라인 시작':^80}")
    print(f"{'시작 시간: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")
    print("=" * 80 + "\n")

    # ========== Step 1: 감성 분류 모델 미세조정 ==========
    print("\n" + "🔹" * 40)
    print("Step 1/4: 감성 분류 모델 미세조정 (fine_tune_sentiment_models)")
    print("🔹" * 40)

    try:
        from fine_tune_sentiment_models import main as sentiment_finetune_main

        step1_start = time.time()
        sentiment_finetune_main()
        step1_time = time.time() - step1_start
        print(f"\nStep 1 완료 (소요 시간: {step1_time/60:.1f}분)")
    except Exception as e:
        print(f"\nStep 1 실패: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ========== Step 2: 의미 기반 모델 미세조정 ==========
    print("\n" + "🔹" * 40)
    print("Step 2/4: 의미 기반 모델 미세조정 (fine_tune_semantic_model)")
    print("🔹" * 40)

    try:
        from fine_tune_semantic_model import main as semantic_finetune_main

        step2_start = time.time()
        semantic_finetune_main()
        step2_time = time.time() - step2_start
        print(f"\nStep 2 완료 (소요 시간: {step2_time/60:.1f}분)")
    except Exception as e:
        print(f"\nStep 2 실패: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ========== Step 3: 감성 벡터화 ==========
    print("\n" + "🔹" * 40)
    print("Step 3/4: 감성 벡터화 (sentiment_vectorize)")
    print("🔹" * 40)

    try:
        from sentiment_vectorize import main as sentiment_vectorize_main

        step3_start = time.time()
        sentiment_vectorize_main()
        step3_time = time.time() - step3_start
        print(f"\nStep 3 완료 (소요 시간: {step3_time/60:.1f}분)")
    except Exception as e:
        print(f"\nStep 3 실패: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ========== Step 4: 의미 벡터화 ==========
    print("\n" + "🔹" * 40)
    print("Step 4/4: 의미 벡터화 (semantic_vectorize)")
    print("🔹" * 40)

    try:
        from semantic_vectorize import main as semantic_vectorize_main

        step4_start = time.time()
        semantic_vectorize_main()
        step4_time = time.time() - step4_start
        print(f"\nStep 4 완료 (소요 시간: {step4_time/60:.1f}분)")
    except Exception as e:
        print(f"\nStep 4 실패: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ========== 완료 ==========
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"{'전체 파이프라인 완료!':^80}")
    print(f"{'종료 시간: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")
    print(
        f"{'총 소요 시간: ' + f'{total_time/60:.1f}분 ({total_time/3600:.2f}시간)':^80}"
    )
    print("=" * 80 + "\n")

    return True


def main():
    """메인 함수"""
    success = run_pipeline()
    if not success:
        print("\n 파이프라인 실행 중 오류가 발생했습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
