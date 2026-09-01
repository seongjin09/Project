"""
실행 환경 감지 유틸리티
Colab, 로컬 환경을 자동으로 감지하고 최적의 실행 모드를 반환합니다.
"""

import os


def detect_environment():
    """
    실행 환경 자동 감지

    Returns:
        str: "colab" 또는 "local"
    """
    try:
        import sys

        # 방법 1: google.colab 모듈이 이미 import되어 있는지 확인
        if "google.colab" in sys.modules:
            return "colab"

        # 방법 2: 직접 import 시도
        try:
            import google.colab

            return "colab"
        except ImportError:
            pass

        # 방법 3: 환경 변수 확인
        if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
            return "colab"

        return "local"
    except Exception:
        return "local"


def get_execution_mode(mode="auto"):
    """
    실행 모드 결정

    Args:
        mode (str): "auto" (자동 감지), "colab" (강제 Colab), "local" (강제 로컬)

    Returns:
        str: "colab" 또는 "local"
    """
    if mode == "auto":
        return detect_environment()
    return mode


def is_colab():
    """
    Colab 환경 여부 확인

    Returns:
        bool: Colab 환경이면 True, 아니면 False
    """
    return detect_environment() == "colab"


def is_local():
    """
    로컬 환경 여부 확인

    Returns:
        bool: 로컬 환경이면 True, 아니면 False
    """
    return detect_environment() == "local"
