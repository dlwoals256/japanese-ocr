import os
import time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

# プロジェクトルートの解決とカレントディレクトリの統一
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vlm-lora-output"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "vlm-lora-final"
DATASET_PATH = DATA_DIR / "dataset.jsonl"

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"プロジェクトルート: {PROJECT_ROOT}")
print(f"学習データ: {DATASET_PATH}")
print(f"ベースモデルをダウンロード＆ロード中 ({MODEL_ID})...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16
)

# LoRA設定 (LLMのAttention層に適用)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    remove_unused_columns=False,
    dataset_text_field="messages",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=processor
)

print(f"学習を開始します (総サンプル数: {len(dataset)}件, 5エポック)...")
start_time = time.time()

trainer.train()

training_time = time.time() - start_time
print(f"✅ 学習が完了しました。かかった時間: {training_time:.2f}秒")

FINAL_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
trainer.model.save_pretrained(str(FINAL_MODEL_DIR))
print(f"✅ アダプタを保存しました: {FINAL_MODEL_DIR}")
