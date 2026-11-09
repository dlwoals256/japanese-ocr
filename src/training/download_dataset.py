import os
from pathlib import Path
from huggingface_hub import snapshot_download

# プロジェクトルートの解決
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = PROJECT_ROOT / "data" / "japanese_receipts"

print(f"Hugging Faceから IngCrowd/japanese-receipts-detailed をダウンロード中...")
print(f"保存先: {LOCAL_DIR}")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="IngCrowd/japanese-receipts-detailed",
    repo_type="dataset",
    local_dir=str(LOCAL_DIR)
)

print(f"✅ ダウンロード完了: {LOCAL_DIR}")
