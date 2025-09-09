# 要件定義書 — **beetle-detector-yolov8-oid**

**Author:** Murasan（むらさん）  
**Website:** https://murasan-net.com/  
**Repository (proposed):** `beetle-detector-yolov8-oid`

---

## 1. 背景・目的（Overview）
Google が公開する **Open Images Dataset (OID)** の「**Beetle**（甲虫）」クラスを抽出し、**YOLOv8（Ultralytics）** をベースにファインチューニングして、一般画像から甲虫を高精度に検出する単一クラス検出器を構築します。  
本プロジェクトは **再現性・法令遵守・軽量推論** を重視し、学習用スクリプト、評価・可視化、推論 CLI、ならびにライセンス帰属（アトリビューション）管理フローを含む**実務向けテンプレート**を提供します。

---

## 2. 成果物（Deliverables）
- 学習済み重み（`.pt`）: `yolov8n` を起点に `yolov8s/m` へ拡張可能
- スクリプト群: データ抽出／前処理／YOLO 形式エクスポート／学習／評価／推論
- ドキュメント:
  - `README.md`（実行手順, サンプル結果）
  - `LICENSE_NOTES.md`（OID ライセンス遵守、画像の帰属方法）
  - `ATTRIBUTION.csv`（学習・掲載画像の出典・作者・ライセンス一覧）
- 評価レポート: mAP, PR 曲線, 混同行列, 誤検出の視覚レビュー

---

## 3. スコープ（Scope）
- **データ**: Open Images Dataset V6/V7 の **Beetle** クラス（boxable）を **FiftyOne Dataset Zoo** で部分ダウンロード。  
  - 画像の再配布は行わず、学習用途に限定
  - 学習/検証分割（例: train 1,200 枚 / val 300 枚）をベースラインとする
- **モデル**: Ultralytics YOLOv8（`n` → 必要に応じ `s` / `m`）
- **用途**: 一般シーンの甲虫検出（単一クラス）。種レベル識別はスコープ外（将来拡張）。

---

## 4. ライセンス・法務要件（Legal）
- **OID アノテーション**: **CC BY 4.0**（クレジット表記が必要）  
- **OID 画像**: **CC BY 2.0 としてリスト**（ただし**各画像のライセンスは個別確認が必要**。Google は保証しない旨をサイトが明示）  
- **方針**:
  - 本リポジトリは**学習に必要な ID/アノテーション/導線のみ**を扱い、**画像ファイル自体は再配布しない**
  - 書籍/ブログ等で画像を掲載する際は、各画像ごとに **出典・作者・ライセンス表記** を行い、`ATTRIBUTION.csv` から自動生成可能にする
- **記載例（クレジット例）**:  
  `Image © <Author Name>, via Flickr (Open Images). Licensed under CC BY 2.0.`

> 参考:  
> - Open Images: https://storage.googleapis.com/openimages/web/index.html  
> - Ultralytics YOLOv8: https://docs.ultralytics.com  
> - FiftyOne Open Images: https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html#open-images

---

## 5. 受け入れ基準（Acceptance Criteria）
- **精度**: val セットで `mAP@0.5 ≥ 0.60` を初期達成（単一クラス）
- **再現性**: クリーンな環境から **ワンコマンド**でデータ抽出→学習→評価まで通る
- **ドキュメント**: 実行手順とライセンス遵守の明記、サンプル可視化を含む

---

## 6. 実行環境（Environment）
- OS: Ubuntu 20.04+/22.04+, Windows 11（WSL2 可）
- Python: 3.10+
- GPU: NVIDIA 8–24GB VRAM 推奨（CPU 学習は想定外）
- 主要依存関係: `ultralytics`, `torch`, `fiftyone`, `pandas`, `tqdm`

### セットアップ例
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install ultralytics fiftyone pandas tqdm
```

---

## 7. データ要件・前処理（Data & Preprocessing）
- **クラス**: `Beetle`（単一クラス検出）
- **サンプル数**: 初期は train 1,200 / val 300 を目安（調整可）
- **アノテーション**: OID → YOLO 形式（`class x_center y_center width height` 正規化）
- **クレンジング**:
  - 壊れ画像・極端に小さい bbox（例: 幅または高さ < 8px）を除外
  - OID の **非網羅ラベリング** に留意（評価時の誤検出判定を適切化）

### FiftyOne による OID 取得（例）
```python
import fiftyone as fo
import fiftyone.zoo as foz

train_ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Beetle"],
    max_samples=1200,
)

val_ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Beetle"],
    max_samples=300,
)
```

### 小さすぎる bbox の除去（例）
```python
def filter_small_boxes(dataset, min_side=8):
    to_delete = []
    for sample in dataset:
        det = sample.detections
        if not det:
            to_delete.append(sample.id); continue
        keep = []
        for d in det.detections:
            w = d.bounding_box[2] * sample.metadata.width
            h = d.bounding_box[3] * sample.metadata.height
            if w >= min_side and h >= min_side:
                keep.append(d)
        if keep:
            det.detections = keep
            sample.detections = det
            sample.save()
        else:
            to_delete.append(sample.id)
    if to_delete:
        dataset.delete_samples(to_delete)

filter_small_boxes(train_ds)
filter_small_boxes(val_ds)
```

### YOLO 形式へのエクスポート
```python
export_dir = "datasets/beetle-oid-yolo"

train_ds.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,  # YOLOv8 互換
    label_field="detections",
    split="train",
    classes=["Beetle"],
)

val_ds.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    split="val",
    classes=["Beetle"],
)
# data.yaml, images/{train,val}/, labels/{train,val}/ が生成される
```

---

## 8. 学習設計（Training）
- **ベース重み**: `yolov8n.pt`（軽量・高速ベースライン）→ 必要に応じて `yolov8s/m` へ
- **入力解像度**: `imgsz=640`（初期）。小物体対策で 768 も検討
- **オーグメンテーション**: flip/scale/HSV/Mosaic/MixUp（過度な崩れに注意）
- **エポック/バッチ**: `epochs=80` 目安, `batch=16–32`
- **最適化**: 既定（SGD/AdamW）で開始し、学習曲線を見て調整

### Ultralytics CLI（例）
```bash
yolo detect train \
  data=datasets/beetle-oid-yolo/data.yaml \
  model=yolov8n.pt \
  epochs=80 imgsz=640 batch=16
```

---

## 9. 評価と可視化（Evaluation）
- Ultralytics の `val` 出力で **mAP@0.5 / mAP@0.5:0.95**, PR 曲線, 混同行列を確認
- OID の非網羅ラベリングを考慮した評価を試す場合は **FiftyOne の Open Images 評価**を適用可能
- 誤検出の**エラーブラウジング**でデータ/オーグメント/閾値を見直す

---

## 10. 推論（Inference）
```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source="test_images/*.jpg" \
  conf=0.25
```
- Python API でも推論スクリプトを提供（バッチ処理、保存パス、閾値可変）

---

## 11. プロジェクト構成（Repository Layout: 提案）
```
beetle-detector-yolov8-oid/
├─ scripts/
│   ├─ 01_download_oid_beetle.py
│   ├─ 02_clean_and_export_yolo.py
│   ├─ 03_train_yolov8.sh
│   ├─ 04_eval_report.ipynb
│   └─ 05_infer_cli.py
├─ datasets/
│   └─ beetle-oid-yolo/
│       ├─ images/{train,val}/
│       ├─ labels/{train,val}/
│       └─ data.yaml
├─ runs/                # YOLO 出力（学習/検証/可視化）
├─ docs/
│   ├─ LICENSE_NOTES.md
│   └─ ATTRIBUTION.csv  # 自動生成を推奨
├─ README.md
└─ REQUIREMENTS.md      # 本ドキュメント
```

---

## 12. 非機能要件（Non-Functional）
- **性能**: A100/3090 で `yolov8n` 100FPS 目安（参考値）。CPU/Pi は将来検討
- **拡張性**: 複数クラス/種レベル識別への拡張容易性
- **再現性**: 乱数シード, 依存関係ピン留め, データ取得の決定的処理
- **可搬性**: Dockerfile/UV/Pipenv 等の環境固定をオプション提供
- **ドキュメント性**: コマンド一発, 画像付サンプル, FAQ

---

## 13. リスクと対策（Risks & Mitigations）
| リスク | 影響 | 対策 |
|---|---|---|
| 画像ライセンスの不一致 | 掲載不可/信用低下 | `ATTRIBUTION.csv` による出典管理、掲載前の個別確認、画像の再配布禁止 |
| 非網羅ラベリング | 過剰 FP と評価誤差 | FiftyOne の Open Images 評価、ハードネガティブの見直し |
| 小物体・遠景 | 検出漏れ | 高解像度学習/タイル推論/オーグメント調整 |
| ドメインギャップ | 実運用で精度低下 | 追加データ収集・継続学習・閾値最適化 |

---

## 14. マイルストーン（Milestones）
1. **M1**: 環境構築／OID 抽出／YOLO 形式化（2–3日）  
2. **M2**: `yolov8n` ベースライン学習・評価（1–2日）  
3. **M3**: 誤検出分析・オーグメント調整（2–3日）  
4. **M4**: `yolov8s/m` への拡張とベストモデル確定（2–3日）  
5. **M5**: 推論 CLI/README 仕上げ・ライセンス文書化（1–2日）

---

## 15. 将来拡張（Future Work）
- **日本在来種（クワガタ/カブトムシ）** への特化（追加アノテーション or 二段階分類）
- エッジ最適化（TensorRT/ONNX、NPU/Hailo 等）
- 物体追跡（ByteTrack/OC-SORT）や動画推論の追加

---

## 16. 参考リンク（References）
- Open Images (Official): https://storage.googleapis.com/openimages/web/index.html  
- Ultralytics YOLOv8 Docs: https://docs.ultralytics.com  
- FiftyOne Open Images Zoo: https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html#open-images  
- FiftyOne YOLO Export: https://voxel51.com/docs/fiftyone/integrations/using_yolo_datasets.html
