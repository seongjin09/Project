"""
모델 자동 다운로드 유틸리티
"""

import os
from pathlib import Path


def download_model_from_huggingface(
    model_id: str, local_dir: str, use_auth_token: str = None
):
    """
    Hugging Face Hub에서 모델 다운로드

    Args:
        model_id: Hugging Face 모델 ID (예: "username/roberta-semantic-final")
        local_dir: 로컬 저장 경로
        use_auth_token: Hugging Face 토큰 (private 모델일 경우)
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"🔽 모델 다운로드 중: {model_id}")
        print(f"📁 저장 위치: {local_dir}")

        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=use_auth_token,
        )

        print(f"✅ 모델 다운로드 완료!")
        return True

    except Exception as e:
        print(f"❌ 모델 다운로드 실패: {e}")
        return False


def ensure_model_exists(model_name: str, model_id: str = None):
    """
    모델이 로컬에 없으면 다운로드

    Args:
        model_name: 로컬 모델 경로 (예: "./models/fine_tuned/roberta_semantic_final")
        model_id: Hugging Face 모델 ID (로컬에 없을 경우 다운로드)

    Returns:
        str: 모델 경로
    """
    model_path = Path(model_name)

    # 이미 존재하면 그대로 반환
    if model_path.exists() and (model_path / "config.json").exists():
        return str(model_path)

    # 모델이 없고 model_id가 제공되었으면 다운로드
    if model_id:
        print(f"⚠️ 로컬에 모델이 없습니다. Hugging Face에서 다운로드합니다...")

        # 부모 디렉토리 생성
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # 다운로드
        if download_model_from_huggingface(model_id, str(model_path)):
            return str(model_path)
        else:
            raise FileNotFoundError(
                f"모델 다운로드 실패: {model_id}\n"
                f"1. Hugging Face에 모델이 업로드되어 있는지 확인하세요.\n"
                f"2. Private 모델인 경우 HF_TOKEN을 설정하세요.\n"
                f"   export HF_TOKEN=your_token_here"
            )

    # model_id도 없으면 에러
    raise FileNotFoundError(
        f"모델을 찾을 수 없습니다: {model_name}\n"
        f"다음 중 하나를 수행하세요:\n"
        f"1. 로컬에 모델 파일을 배치하세요.\n"
        f"2. model_id 파라미터를 제공하여 Hugging Face에서 다운로드하세요."
    )


# 모델 ID 매핑 (필요시 수정)
MODEL_ID_MAP = {
    "./models/fine_tuned/roberta_semantic_final": "YOUR_USERNAME/roberta-semantic-final",
    "./models/fine_tuned/roberta_sentiment_final": "YOUR_USERNAME/roberta-sentiment-final",
}


def get_model_path(local_path: str) -> str:
    """
    모델 경로 반환 (없으면 자동 다운로드)

    Args:
        local_path: 로컬 모델 경로

    Returns:
        str: 사용 가능한 모델 경로
    """
    model_id = MODEL_ID_MAP.get(local_path)
    return ensure_model_exists(local_path, model_id)
