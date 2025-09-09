#!/bin/bash

# YOLOv8 Beetle Detection Training Script
# CPU最適化版（軽量テスト用）とGPU版（本格訓練用）を提供

set -e  # エラー時に終了

echo "=== YOLOv8 Beetle Detection Training ==="

# 引数解析
MODE="test"  # デフォルトはテストモード
EPOCHS=30
BATCH=8
IMGSZ=640
MODEL="yolov8n.pt"
DATA="datasets/beetle-oid-yolo/data.yaml"
PROJECT="runs/detect"
NAME="train"

# オプション解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --full-scale)
            MODE="full"
            EPOCHS=80
            BATCH=16
            echo "🚀 Full scale training mode"
            shift
            ;;
        --cpu-optimized)
            MODE="cpu"
            EPOCHS=20
            BATCH=4
            echo "💻 CPU optimized mode"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data)
            DATA="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --full-scale      Full scale training (80 epochs, batch 16)"
            echo "  --cpu-optimized   CPU optimized training (20 epochs, batch 4)"
            echo "  --epochs N        Number of epochs (default: 30)"
            echo "  --batch N         Batch size (default: 8)"
            echo "  --imgsz N         Image size (default: 640)"
            echo "  --model PATH      Model path (default: yolov8n.pt)"
            echo "  --data PATH       Data yaml path (default: datasets/beetle-oid-yolo/data.yaml)"
            echo "  --name NAME       Experiment name (default: train)"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Training parameters:"
echo "  Mode: $MODE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH"
echo "  Image size: $IMGSZ"
echo "  Model: $MODEL"
echo "  Data: $DATA"
echo "  Project: $PROJECT"
echo "  Name: $NAME"

# 環境チェック
echo ""
echo "=== Environment Check ==="

# データファイル存在確認
if [ ! -f "$DATA" ]; then
    echo "❌ Data file not found: $DATA"
    echo "Please run: python scripts/02_clean_and_export_yolo.py first"
    exit 1
fi

echo "✅ Data file found: $DATA"

# GPU確認
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    DEVICE="0"  # GPU使用
else
    echo "💻 GPU not found, using CPU"
    DEVICE="cpu"
fi

# 仮想環境確認
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment not detected"
    echo "Activating .venv..."
    source .venv/bin/activate
fi

# Python/ultralytics確認
echo ""
echo "=== Package Verification ==="
python -c "import ultralytics; print(f'✅ Ultralytics version: {ultralytics.__version__}')" 2>/dev/null || {
    echo "❌ Ultralytics not found. Installing..."
    pip install ultralytics
}

python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')" 2>/dev/null || {
    echo "❌ PyTorch not found"
    exit 1
}

# 学習開始
echo ""
echo "=== Training Started ==="
echo "Start time: $(date)"

# CPU最適化設定
if [ "$DEVICE" = "cpu" ]; then
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    echo "🔧 CPU optimization enabled (4 threads)"
fi

# YOLOv8学習実行
yolo detect train \
    data="$DATA" \
    model="$MODEL" \
    epochs=$EPOCHS \
    batch=$BATCH \
    imgsz=$IMGSZ \
    device="$DEVICE" \
    project="$PROJECT" \
    name="$NAME" \
    save=True \
    plots=True \
    verbose=True \
    patience=10 \
    seed=42 \
    deterministic=True

TRAIN_EXIT_CODE=$?

echo ""
echo "=== Training Completed ==="
echo "End time: $(date)"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training successful!"
    
    # 結果確認
    RESULT_DIR="$PROJECT/$NAME"
    if [ -d "$RESULT_DIR" ]; then
        echo ""
        echo "Results saved to: $RESULT_DIR"
        echo "Key files:"
        
        if [ -f "$RESULT_DIR/weights/best.pt" ]; then
            echo "  ✅ Best weights: $RESULT_DIR/weights/best.pt"
        fi
        
        if [ -f "$RESULT_DIR/weights/last.pt" ]; then
            echo "  ✅ Last weights: $RESULT_DIR/weights/last.pt"
        fi
        
        if [ -f "$RESULT_DIR/results.png" ]; then
            echo "  ✅ Training curves: $RESULT_DIR/results.png"
        fi
        
        if [ -f "$RESULT_DIR/confusion_matrix.png" ]; then
            echo "  ✅ Confusion matrix: $RESULT_DIR/confusion_matrix.png"
        fi
        
        # 自動評価実行
        echo ""
        echo "=== Auto Validation ==="
        if [ -f "$RESULT_DIR/weights/best.pt" ]; then
            echo "Running validation with best weights..."
            yolo detect val \
                model="$RESULT_DIR/weights/best.pt" \
                data="$DATA" \
                device="$DEVICE" \
                plots=True \
                save_json=True
        fi
        
        echo ""
        echo "🎉 Training pipeline completed successfully!"
        echo "Next steps:"
        echo "  1. Check results in: $RESULT_DIR"
        echo "  2. Run inference: python scripts/05_infer_cli.py --model $RESULT_DIR/weights/best.pt"
        echo "  3. View evaluation: jupyter notebook scripts/04_eval_report.ipynb"
        
    else
        echo "⚠️  Result directory not found: $RESULT_DIR"
    fi
    
    exit 0
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi