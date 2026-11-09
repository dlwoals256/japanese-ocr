import os
import json
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# プロジェクトルートの解決とカレントディレクトリの統一
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
TEST_DATA_PATH = DATA_DIR / "dataset_test.jsonl"
ADAPTER_PATH = PROJECT_ROOT / "models" / "vlm-lora-final"

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"プロジェクトルート: {PROJECT_ROOT}")
print("1. ベースモデルとプロセッサをロード中...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16
)

if ADAPTER_PATH.exists():
    print(f"2. 学習済みLoRAアダプタ ({ADAPTER_PATH}) をロード中...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model.eval()
    print("✅ LoRAアダプタのロード成功")
else:
    print(f"⚠️ {ADAPTER_PATH} が見つからないため、ベースモデルで推論します。")
    model = base_model
    model.eval()

print(f"\n3. テストデータ（{TEST_DATA_PATH}）を評価します...")
if not TEST_DATA_PATH.exists():
    print(f"エラー: テストデータが見つかりません: {TEST_DATA_PATH}")
    exit(1)

with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"テストサンプル総数: {len(lines)}件")
num_eval = min(3, len(lines)) # 先頭3件をテスト表示

for i in range(num_eval):
    sample = json.loads(lines[i])
    meta = sample.get("_meta", {})
    user_msg = sample["messages"][0]

    img_rel_path = None
    prompt_text = ""
    for c in user_msg["content"]:
        if c.get("type") == "image":
            img_rel_path = c.get("image")
        elif c.get("type") == "text":
            prompt_text = c.get("text")

    img_full_path = PROJECT_ROOT / img_rel_path
    print(f"\n==================== [ テストサンプル {i+1} / {num_eval} ] ====================")
    print(f"画像: {img_rel_path}")
    print("【正解ラベル】:")
    print(f"  店舗名: {meta.get('store_name')}")
    print(f"  日時:   {meta.get('datetime')}")
    print(f"  合計:   {meta.get('total')}")

    image = Image.open(img_full_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    elapsed = time.time() - start_time

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n【AIの出力】:")
    print(output_text.strip())
    print(f"⏱️ 推論時間: {elapsed:.2f}秒")
