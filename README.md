# 日本語レシート情報抽出 AI (Qwen2-VL LoRA)

実写の日本語レシート画像から、**店舗名 (store_name)**、**日時 (datetime)**、**合計金額 (total)** を高精度に構造化JSONとして抽出するマルチモーダルAIプロジェクトです。

---

## 📁 ディレクトリ構成

```text
finetune/
├── data/                         # データセット
│   ├── japanese_receipts/        # 実写レシート原画 & アノテーション (100枚)
│   ├── receipts_resized/         # 長辺1024pxに最適化された画像 (92枚)
│   ├── dataset_train.jsonl       # 学習用データ (72件)
│   ├── dataset_test.jsonl        # 評価用データ (20件)
│   └── dataset.jsonl             # 学習用の共通参照
│
├── models/                       # 学習済みモデル
│   └── vlm-lora-final/           # LoRAアダプタ (4.2MB)
│
├── outputs/                      # 学習時の中間ログ
│
├── sample_images/                # 手動テスト用画像 (receipt.png など)
│
└── src/                          # ソースコード
    ├── training/                 # 【学習・評価】
    │   ├── download_dataset.py   # データセットダウンロード
    │   ├── prepare_data.py       # データクレンジング・リサイズ・JSONL生成
    │   ├── train.py              # LoRAファインチューニング実行
    │   └── evaluate.py           # 未知テストデータでの推論評価
    │
    └── serving/                  # 【Webサービング】
        ├── app.py                # FastAPI バックエンドAPI
        └── static/
            └── index.html        # Webフロントエンド画面
```

---

## 🚀 使い方

仮想環境を作成・有効化し、必要なパッケージをインストールして実行します。
※どのディレクトリから実行しても、パスは自動で正しく解決されます。

### 0. 環境構築（依存ライブラリのインストール）

```bash
# 仮想環境の作成と有効化（未作成の場合）
python3 -m venv .venv
source .venv/bin/activate

# 依存パッケージの一括インストール
pip install -r requirements.txt
```

---

### 1. データのダウンロードと前処理（初回セットアップ）

```bash
# ① Hugging Face から実写レシートデータセットをダウンロード（data/japanese_receipts/ に自動配置）
python src/training/download_dataset.py

# ② 画像の長辺1024pxリサイズおよび学習・評価用JSONL（train: 72件, test: 20件）の自動生成
python src/training/prepare_data.py
```

---

### 2. Webサービングの起動（ブラウザで試す）

```bash
python src/serving/app.py
# または
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```
ブラウザで **http://localhost:8000** （または `app.py` で設定したポート）を開くと、レシート画像をドラッグ＆ドロップして即座に抽出結果を確認できるUIが立ち上がります。

---

### 3. 未知のテストデータ（20件）に対する精度評価

```bash
python src/training/evaluate.py
```
学習に使っていないテスト画像に対して推論を行い、正解ラベルとAIの出力（店舗名・日時・合計金額）を比較表示します。

---

### 4. 再学習（ファインチューニング）

```bash
python src/training/train.py
```
`data/dataset.jsonl` を用いて、Qwen2-VL-2BのLLM層（Attention層）にLoRA学習を行い、成果物を `models/vlm-lora-final` に保存します。
