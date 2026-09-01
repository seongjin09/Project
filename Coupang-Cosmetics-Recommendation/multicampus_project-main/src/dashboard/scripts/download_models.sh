#!/bin/bash

# 모델 다운로드 스크립트
# 사용법: ./scripts/download_models.sh

echo "🤖 모델 다운로드 스크립트"
echo "=========================="
echo ""

# 환경 확인
if ! command -v gdown &> /dev/null; then
    echo "📦 gdown 설치 중..."
    pip install gdown
fi

# 모델 디렉토리 생성
mkdir -p models/fine_tuned

echo ""
echo "📥 모델 다운로드 옵션:"
echo "1. Hugging Face에서 다운로드 (자동)"
echo "2. Google Drive에서 다운로드 (수동 설정 필요)"
echo "3. 로컬 파일 복사"
echo ""
read -p "선택 (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🤗 Hugging Face에서 다운로드..."
        
        # huggingface_hub 설치
        pip install -q huggingface_hub
        
        # 모델 ID 입력 받기
        read -p "모델 ID 입력 (예: username/roberta-semantic-final): " MODEL_ID
        
        if [ -z "$MODEL_ID" ]; then
            echo "❌ 모델 ID가 비어있습니다."
            exit 1
        fi
        
        # 다운로드
        python3 << EOF
from huggingface_hub import snapshot_download
import os

model_id = "${MODEL_ID}"
local_dir = "./models/fine_tuned/roberta_semantic_final"

print(f"📁 다운로드 위치: {local_dir}")
os.makedirs(local_dir, exist_ok=True)

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print("✅ 다운로드 완료!")
except Exception as e:
    print(f"❌ 다운로드 실패: {e}")
    exit(1)
EOF
        ;;
        
    2)
        echo ""
        echo "📂 Google Drive 다운로드..."
        echo "⚠️  먼저 파일 공유 링크에서 FILE_ID를 추출하세요."
        echo "    예: https://drive.google.com/file/d/1ABC123xyz/view"
        echo "        → FILE_ID = 1ABC123xyz"
        echo ""
        read -p "FILE_ID 입력: " FILE_ID
        
        if [ -z "$FILE_ID" ]; then
            echo "❌ FILE_ID가 비어있습니다."
            exit 1
        fi
        
        # 다운로드
        echo "📥 다운로드 중..."
        gdown "https://drive.google.com/uc?id=${FILE_ID}" -O models_temp.zip
        
        # 압축 해제
        echo "📦 압축 해제 중..."
        unzip -q models_temp.zip -d models/fine_tuned/
        rm models_temp.zip
        
        echo "✅ 다운로드 완료!"
        ;;
        
    3)
        echo ""
        echo "📁 로컬 파일 경로를 입력하세요:"
        read -p "경로: " LOCAL_PATH
        
        if [ ! -d "$LOCAL_PATH" ]; then
            echo "❌ 경로를 찾을 수 없습니다: $LOCAL_PATH"
            exit 1
        fi
        
        # 복사
        echo "📋 복사 중..."
        cp -r "$LOCAL_PATH" models/fine_tuned/roberta_semantic_final
        
        echo "✅ 복사 완료!"
        ;;
        
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "🎉 모델 설치가 완료되었습니다!"
echo "   위치: $(pwd)/models/fine_tuned/roberta_semantic_final"
echo ""
echo "다음 명령어로 앱을 실행하세요:"
echo "  streamlit run main.py"
