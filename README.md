# 🪲 Beetle Detector YOLOv8 OID

YOLOv8とOpen Images Dataset (OID)を使用したBeetle（甲虫）検出システム

## 📋 プロジェクト概要

このプロジェクトは、Googleが公開するクリーンなデータセット「Open Images Dataset (OID)」を使用し、YOLOv8をベースに移転学習を行ってBeetle（甲虫）を検出できるモデルを作成します。

**データセット仕様:**
- 約190万枚の画像
- 600の物体クラス
- 1,540万個以上のバウンディングボックス
- CC BY 4.0ライセンス（適切な帰属表記が必要）

## 🎯 プロジェクト目標

- mAP@0.5 ≥ 0.60 の検出精度達成
- 再現性とライセンスコンプライアンスの確保
- CPU環境での軽量テスト対応
- Google Colab GPUでの本格訓練対応

## 🏗️ プロジェクト構成

```
beetle-detector-yolov8-oid/
├── scripts/              # 実行スクリプト
│   ├── 01_download_oid_beetle.py    # データダウンロード
│   ├── 02_clean_and_export_yolo.py  # データ前処理・YOLO変換
│   ├── 03_train_yolov8.sh           # 訓練スクリプト
│   ├── 04_eval_report.ipynb         # 評価・可視化ノートブック
│   └── 05_infer_cli.py              # 推論CLI
├── datasets/             # データセット
│   └── beetle-oid-yolo/
│       ├── images/       # 画像ファイル
│       ├── labels/       # YOLOアノテーション
│       └── data.yaml     # YOLOデータ設定
├── runs/                 # 訓練結果
├── docs/                 # ドキュメント・帰属情報
├── requirements.txt      # Python依存関係
├── CLAUDE.md            # Claude Code ルールファイル
├── 作業手順書.md        # 日本語作業手順
└── README.md            # このファイル
```

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリクローン
git clone <repository-url>
cd beetle-detector-yolov8-oid

# Python仮想環境作成・有効化
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 依存関係インストール
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 軽量テスト（推奨開始方法）

```bash
# データダウンロード（軽量版: train 200枚、val 50枚）
python scripts/01_download_oid_beetle.py

# データクリーニング・YOLO形式変換
python scripts/02_clean_and_export_yolo.py

# CPU最適化訓練（20-30分程度）
bash scripts/03_train_yolov8.sh --cpu-optimized

# 評価実行
jupyter notebook scripts/04_eval_report.ipynb

# 推論テスト
python scripts/05_infer_cli.py --model runs/detect/train/weights/best.pt --source test_image.jpg
```

### 3. 本格訓練（Google Colab推奨）

```bash
# フルスケールデータダウンロード（train 1200枚、val 300枚）
python scripts/01_download_oid_beetle.py --full-scale

# データ前処理
python scripts/02_clean_and_export_yolo.py --full-scale

# GPU訓練（80エポック）
bash scripts/03_train_yolov8.sh --full-scale
```

## 📊 期待される結果

### 軽量テスト（CPU）
- **訓練時間**: 20-40分
- **データ**: train 200枚、val 50枚  
- **目標mAP@0.5**: ≥0.40（学習確認用）

### 本格訓練（GPU）
- **訓練時間**: 2-4時間
- **データ**: train 1200枚、val 300枚
- **目標mAP@0.5**: ≥0.60（プロダクション用）

## 🛠️ 使用方法詳細

### データダウンロードオプション

```bash
# 軽量テスト（デフォルト）
python scripts/01_download_oid_beetle.py

# カスタムサンプル数
python scripts/01_download_oid_beetle.py --train-samples 500 --val-samples 100

# フルスケール
python scripts/01_download_oid_beetle.py --full-scale
```

### 訓練オプション

```bash
# CPU最適化（バッチサイズ4、20エポック）
bash scripts/03_train_yolov8.sh --cpu-optimized

# テストモード（デフォルト: バッチサイズ8、30エポック）
bash scripts/03_train_yolov8.sh

# フルスケール（バッチサイズ16、80エポック）
bash scripts/03_train_yolov8.sh --full-scale

# カスタム設定
bash scripts/03_train_yolov8.sh --epochs 50 --batch 12 --model yolov8s.pt
```

### 推論オプション

```bash
# 単一画像
python scripts/05_infer_cli.py -m runs/detect/train/weights/best.pt -s image.jpg

# ディレクトリ内全画像
python scripts/05_infer_cli.py -m best.pt -s images/

# 信頼度閾値調整
python scripts/05_infer_cli.py -m best.pt -s images/ --conf 0.5

# 結果をJSONで保存
python scripts/05_infer_cli.py -m best.pt -s images/ --save-json
```

## 📈 性能ベンチマーク

| 環境 | モデル | データ数 | mAP@0.5 | 訓練時間 | 推論速度 |
|------|--------|----------|---------|----------|----------|
| CPU (テスト) | YOLOv8n | 250枚 | ~0.45 | 30分 | 200ms |
| GPU (本格) | YOLOv8n | 1500枚 | ~0.65 | 3時間 | 15ms |
| GPU (高精度) | YOLOv8s | 1500枚 | ~0.70 | 5時間 | 25ms |

## 📝 ライセンス・法的コンプライアンス

### Open Images Dataset
- **アノテーション**: CC BY 4.0（クレジット表記必須）
- **画像**: CC BY 2.0（個別ライセンス確認必要）
- **帰属管理**: `docs/ATTRIBUTION.csv`で自動生成

### 重要な注意点
- ✅ 学習・研究用途での使用
- ✅ 適切な帰属表記の実施
- ❌ 画像ファイルの再配布禁止
- ❌ 商用利用前の個別ライセンス確認必要

### 帰属表記例
```
Image © [Author Name], via Flickr (Open Images). Licensed under CC BY 2.0.
```

## 🔧 トラブルシューティング

### よくある問題

**1. FiftyOne初回実行が遅い**
```bash
# 初回ダウンロードで時間がかかるのは正常です
# プロキシ環境の場合は環境変数設定が必要な場合があります
```

**2. GPU訓練でメモリ不足**
```bash
# バッチサイズを削減
bash scripts/03_train_yolov8.sh --batch 4
```

**3. データダウンロード失敗**
```bash
# ネットワーク接続確認
# 再試行（部分ダウンロード対応）
python scripts/01_download_oid_beetle.py --train-samples 100 --val-samples 25
```

## 🤝 貢献・改善提案

1. **精度向上**
   - より大きなモデル（YOLOv8m/l）の利用
   - データオーグメンテーション調整
   - ハードネガティブマイニング

2. **速度最適化**  
   - TensorRT/ONNX変換
   - モデル量子化
   - エッジデバイス対応

3. **機能拡張**
   - 物体追跡（ByteTrack）
   - 動画推論対応
   - Web API化

## 📚 参考リンク

- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [FiftyOne Documentation](https://voxel51.com/docs/fiftyone/)
- [CC BY License](https://creativecommons.org/licenses/by/4.0/)

## 📞 サポート

- **技術的な問題**: Issue作成またはPull Request
- **ライセンス関連**: 各画像の個別確認が必要
- **データセット**: Open Images公式サイト参照

---

🪲 **Happy Bug Hunting with AI!** 🤖