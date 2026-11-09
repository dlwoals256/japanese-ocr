import io
import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# パス定義
SERVING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVING_DIR.parents[1]
STATIC_DIR = SERVING_DIR / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"
ADAPTER_PATH = PROJECT_ROOT / "models" / "vlm-lora-final"

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("モデルとプロセッサをロードしています...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=128 * 28 * 28,
        max_pixels=640 * 28 * 28,
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16
    )

    if ADAPTER_PATH.exists():
        print(f"LoRAアダプタ ({ADAPTER_PATH}) をロード中...")
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        print("LoRAアダプタのロードが完了しました。")
    else:
        print(f"⚠️ {ADAPTER_PATH} が存在しないため、ベースモデルを使用します。")
        model = base_model

    model.eval()
    ml_models["processor"] = processor
    ml_models["model"] = model
    print("モデルの初期化が完了しました。")
    yield
    ml_models.clear()

app = FastAPI(title="Receipt Extraction API", lifespan=lifespan)

# 静的ファイルのホスト
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html が見つかりません。")
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()

def resize_image_if_needed(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    w, h = image.size
    if max(w, h) > max_dim:
        if w > h:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        else:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image

@app.post("/api/extract")
async def extract_receipt(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルを指定してください。")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = resize_image_if_needed(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {e}")

    processor = ml_models["processor"]
    model = ml_models["model"]

    prompt_text = "このレシート画像から、店舗名(store_name)、日時(datetime)、合計金額(total)を抽出し、以下のキーを持つJSON形式で出力してください: store_name, datetime, total"

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
    raw_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # JSONパース（マークダウンブロック等の除去）
    parsed_json = None
    try:
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
        else:
            parsed_json = json.loads(raw_output)
    except Exception:
        parsed_json = {
            "raw_text": raw_output
        }

    return {
        "status": "success",
        "latency_sec": round(elapsed, 2),
        "result": parsed_json,
        "raw_output": raw_output
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)