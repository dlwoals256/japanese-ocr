import json
import os
import re
import random
from pathlib import Path
from PIL import Image

# プロジェクトルートの解決
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_IMG_DIR = DATA_DIR / "receipts_resized"
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATIONS_PATH = DATA_DIR / "japanese_receipts" / "annotations.json"
BASE_IMG_DIR = DATA_DIR / "japanese_receipts"

def clean_total(raw_str):
    nums = re.findall(r'[0-9,]+', raw_str)
    if not nums:
        return raw_str.strip()
    amt = nums[-1].strip(',')
    return f"{amt}円"

def resize_image(src_path: Path, dst_path: Path, max_dim=1024):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img.save(dst_path, "JPEG", quality=90)

print("アノテーションファイルを読み込み中...")
with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

extracted_samples = []

for idx, item in enumerate(raw_data):
    rel_img_path = item["data"]["image"]  # e.g. "images/receipt_convenience_0004.jpg"
    full_src_path = BASE_IMG_DIR / rel_img_path
    if not full_src_path.exists():
        continue

    results = item["annotations"][0]["result"]
    id_map = {}
    for res in results:
        res_id = res.get("id")
        if not res_id:
            continue
        if res_id not in id_map:
            id_map[res_id] = {}
        val = res.get("value", {})
        if "rectanglelabels" in val:
            id_map[res_id]["labels"] = val["rectanglelabels"]
        if "text" in val:
            id_map[res_id]["text"] = val["text"]

    store_names = []
    datetimes = []
    totals = []

    for res_id, info in id_map.items():
        labels = info.get("labels", [])
        text_list = info.get("text", [])
        t = " ".join([x.strip() for x in text_list if x.strip()])
        if not t:
            continue
        if "store_name" in labels:
            store_names.append(t)
        if "datetime" in labels:
            datetimes.append(t)
        if "total" in labels:
            totals.append(t)

    # 3項目すべて揃っている高品質データのみ採用
    if store_names and datetimes and totals:
        store = store_names[0]
        dt = datetimes[0]
        tot = clean_total(totals[0])

        # 画像のリサイズ保存
        file_name = Path(rel_img_path).name
        dst_img_path = OUTPUT_IMG_DIR / file_name
        if not dst_img_path.exists():
            resize_image(full_src_path, dst_img_path, max_dim=1024)

        # プロジェクトルートからの相対パス
        img_rel_to_root = str(dst_img_path.relative_to(PROJECT_ROOT))

        target_json = json.dumps({
            "store_name": store,
            "datetime": dt,
            "total": tot
        }, ensure_ascii=False, indent=2)

        prompt_text = "このレシート画像から、店舗名(store_name)、日時(datetime)、合計金額(total)を抽出し、以下のキーを持つJSON形式で出力してください: store_name, datetime, total"

        extracted_samples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_rel_to_root},
                        {"type": "text", "text": prompt_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target_json}
                    ]
                }
            ],
            "_meta": {
                "file_name": file_name,
                "store_name": store,
                "datetime": dt,
                "total": tot
            }
        })

print(f"有効なサンプル数: {len(extracted_samples)}件")

# シャッフルしてTrain(72件)とTest(20件)に分割
random.seed(42)
random.shuffle(extracted_samples)

train_samples = extracted_samples[:72]
test_samples = extracted_samples[72:]

print(f"Train件数: {len(train_samples)}件, Test件数: {len(test_samples)}件")

def write_jsonl(filepath: Path, samples, include_meta=False):
    with open(filepath, "w", encoding="utf-8") as f:
        for s in samples:
            item = {"messages": s["messages"]}
            if include_meta:
                item["_meta"] = s["_meta"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 保存
write_jsonl(DATA_DIR / "dataset_train.jsonl", train_samples)
write_jsonl(DATA_DIR / "dataset_test.jsonl", test_samples, include_meta=True)
write_jsonl(DATA_DIR / "dataset.jsonl", train_samples)

print("✅ データセットの作成完了:")
print(f" - {DATA_DIR / 'dataset_train.jsonl'}")
print(f" - {DATA_DIR / 'dataset_test.jsonl'}")
print(f" - {DATA_DIR / 'dataset.jsonl'}")
print(f" - {OUTPUT_IMG_DIR} (92枚)")
