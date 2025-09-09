# Claude Code Rules for Beetle Detector YOLOv8 Project

## Project Overview
YOLOv8をベースにGoogleが公開しているクリーンなデータセット「Open Images Dataset (OID)」のデータを使って移転学習を行い、Beetle（甲虫）を検出できるモデルを作成します。

「Open Images Dataset (OID)」は約190万枚の画像に対して、600の物体クラスにわたる1,540万個以上のバウンディングボックスが提供されています。このデータセットを使用してBeetleクラスのみを対象とした単一クラス検出器を構築し、高精度での甲虫検出と適切なライセンスコンプライアンスを実現することを目標とします。

開発言語はPythonを使用します。

## Development Environment
- Python: 3.10+
- GPU: NVIDIA 8-24GB VRAM recommended
- OS: Ubuntu 20.04+/22.04+, Windows 11 (WSL2 supported)

## Key Dependencies
- ultralytics (YOLOv8)
- fiftyone (Open Images Dataset access)
- torch, torchvision
- pandas, tqdm, opencv-python

## Project Structure
```
beetle-detector-yolov8-oid/
├─ scripts/
│   ├─ 01_download_oid_beetle.py
│   ├─ 02_clean_and_export_yolo.py
│   ├─ 03_train_yolov8.sh
│   ├─ 04_eval_report.ipynb
│   └─ 05_infer_cli.py
├─ datasets/
├─ docs/
├─ runs/
├─ README.md
└─ CLAUDE.md
```

## Commands to Run
### Setup
```bash
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Training
```bash
bash scripts/03_train_yolov8.sh
```

### Evaluation
```bash
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/beetle-oid-yolo/data.yaml
```

### Inference
```bash
python scripts/05_infer_cli.py --model runs/detect/train/weights/best.pt --source test_images/
```

## Important Notes
- 作業開始時は必ず要件定義書「REQUIREMENTS_beetle-detector-yolov8-oid.md」を確認すること
- Never redistribute OID images directly
- Always maintain proper license attribution in ATTRIBUTION.csv
- Target mAP@0.5 ≥ 0.60 for validation
- Use single class detection (Beetle only)
- Follow CC BY 4.0 license requirements for annotations
- Each image requires individual license verification

## Legal Compliance
- OID annotations: CC BY 4.0
- OID images: CC BY 2.0 (individual verification required)
- No image redistribution in repository
- Maintain attribution records for any published images