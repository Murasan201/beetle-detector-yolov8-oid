#!/usr/bin/env python3
"""
FiftyOneデータセットからYOLO形式へのエクスポート
データクリーニング（小さいバウンディングボックス除去）も含む
"""

import os
import argparse
import fiftyone as fo
from pathlib import Path
import pandas as pd

def filter_small_boxes(dataset, min_side=8, min_area=64):
    """
    小さすぎるバウンディングボックスを除去
    
    Args:
        dataset: FiftyOne dataset
        min_side (int): 最小サイドサイズ（ピクセル）
        min_area (int): 最小面積（ピクセル^2）
    """
    print(f"=== 小さいバウンディングボックス除去開始 ===")
    print(f"Min side: {min_side}px, Min area: {min_area}px²")
    
    to_delete = []
    total_boxes_before = 0
    total_boxes_after = 0
    
    for sample in dataset:
        det = sample.detections
        if not det:
            to_delete.append(sample.id)
            continue
            
        total_boxes_before += len(det.detections)
        keep = []
        
        for d in det.detections:
            # 正規化座標から実際のピクセル座標に変換
            w = d.bounding_box[2] * sample.metadata.width
            h = d.bounding_box[3] * sample.metadata.height
            area = w * h
            
            # サイズと面積の条件をチェック
            if w >= min_side and h >= min_side and area >= min_area:
                keep.append(d)
        
        if keep:
            det.detections = keep
            sample.detections = det
            sample.save()
            total_boxes_after += len(keep)
        else:
            # すべてのバウンディングボックスが小さすぎる場合は削除
            to_delete.append(sample.id)
    
    # 不要なサンプルを削除
    if to_delete:
        dataset.delete_samples(to_delete)
    
    print(f"削除されたサンプル数: {len(to_delete)}")
    print(f"バウンディングボックス: {total_boxes_before} → {total_boxes_after}")
    print(f"除去率: {((total_boxes_before - total_boxes_after) / total_boxes_before * 100):.1f}%")
    
    return len(to_delete), total_boxes_before - total_boxes_after

def export_to_yolo(train_dataset, val_dataset, export_dir="datasets/beetle-oid-yolo"):
    """
    YOLO形式でエクスポート
    
    Args:
        train_dataset: 訓練用FiftyOne dataset
        val_dataset: 検証用FiftyOne dataset  
        export_dir: エクスポート先ディレクトリ
    """
    print(f"=== YOLO形式エクスポート開始 ===")
    print(f"Export directory: {export_dir}")
    
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 訓練データをエクスポート
        print("\n--- 訓練データエクスポート中 ---")
        train_dataset.export(
            export_dir=str(export_path),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="detections",
            split="train",
            classes=["Beetle"],
        )
        
        # 検証データをエクスポート
        print("--- 検証データエクスポート中 ---")
        val_dataset.export(
            export_dir=str(export_path), 
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="detections", 
            split="val",
            classes=["Beetle"],
        )
        
        # data.yamlファイルの確認・修正
        data_yaml = export_path / "data.yaml"
        if data_yaml.exists():
            print(f"✅ data.yaml created: {data_yaml}")
            
            # data.yamlの内容を表示
            with open(data_yaml, 'r') as f:
                content = f.read()
                print("data.yaml content:")
                print(content)
        else:
            print("❌ data.yaml not found")
            
        # ディレクトリ構造確認
        print(f"\n=== エクスポート完了 ===")
        print("Directory structure:")
        for item in sorted(export_path.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(export_path)
                print(f"  {rel_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ エクスポートエラー: {e}")
        return False

def create_attribution_csv(train_dataset, val_dataset, output_path="docs/ATTRIBUTION.csv"):
    """
    使用した画像の帰属情報CSVを作成（ライセンス遵守用）
    
    Args:
        train_dataset: 訓練用dataset
        val_dataset: 検証用dataset
        output_path: CSV出力パス
    """
    print(f"=== 帰属情報CSV作成開始 ===")
    
    attribution_data = []
    
    # 両方のデータセットから情報を収集
    for split, dataset in [("train", train_dataset), ("val", val_dataset)]:
        for sample in dataset:
            # Open Imagesのメタデータから情報を抽出
            image_id = sample.get("open_images_id", "unknown")
            # 他のメタデータも利用可能であれば追加
            attribution_data.append({
                "split": split,
                "image_id": image_id,
                "filename": os.path.basename(sample.filepath),
                "source": "Open Images Dataset V7",
                "license": "CC BY 2.0 (individual verification required)",
                "attribution": "Image from Open Images Dataset V7, licensed under CC BY 2.0",
                "url": f"https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0cyf8&id={image_id}",
                "usage_date": pd.Timestamp.now().strftime("%Y-%m-%d")
            })
    
    # CSV作成
    df = pd.DataFrame(attribution_data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ 帰属情報CSV作成完了: {output_path}")
    print(f"   Total records: {len(attribution_data)}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Clean and export FiftyOne datasets to YOLO format")
    parser.add_argument("--train-dataset", default="oid_beetle_train_200",
                       help="Name of train dataset in FiftyOne")
    parser.add_argument("--val-dataset", default="oid_beetle_val_50", 
                       help="Name of val dataset in FiftyOne")
    parser.add_argument("--export-dir", default="datasets/beetle-oid-yolo",
                       help="Export directory for YOLO format")
    parser.add_argument("--min-side", type=int, default=8,
                       help="Minimum side length for bounding boxes (pixels)")
    parser.add_argument("--min-area", type=int, default=64,
                       help="Minimum area for bounding boxes (pixels^2)")
    parser.add_argument("--skip-cleaning", action="store_true",
                       help="Skip data cleaning step")
    parser.add_argument("--full-scale", action="store_true",
                       help="Use full scale dataset names")
    
    args = parser.parse_args()
    
    # フルスケールモードの場合はデータセット名を調整
    if args.full_scale:
        args.train_dataset = "oid_beetle_train_1200"
        args.val_dataset = "oid_beetle_val_300"
        print("🚀 Full scale mode enabled")
    
    print(f"Train dataset: {args.train_dataset}")
    print(f"Val dataset: {args.val_dataset}")
    
    try:
        # FiftyOneデータセットを読み込み
        print("=== FiftyOneデータセット読み込み中 ===")
        train_dataset = fo.load_dataset(args.train_dataset)
        val_dataset = fo.load_dataset(args.val_dataset)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # データクリーニング（オプション）
        if not args.skip_cleaning:
            print("\n--- データクリーニング実行 ---")
            deleted_train, removed_boxes_train = filter_small_boxes(
                train_dataset, args.min_side, args.min_area
            )
            deleted_val, removed_boxes_val = filter_small_boxes(
                val_dataset, args.min_side, args.min_area  
            )
            
            print(f"クリーニング後 - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        else:
            print("⏭️  データクリーニングをスキップ")
        
        # YOLO形式にエクスポート
        success = export_to_yolo(train_dataset, val_dataset, args.export_dir)
        
        if success:
            # 帰属情報CSV作成（ライセンス遵守）
            attribution_path = create_attribution_csv(train_dataset, val_dataset)
            
            print("✅ 全処理完了！")
            print("次のステップ: bash scripts/03_train_yolov8.sh")
            return 0
        else:
            print("❌ エクスポート失敗")
            return 1
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("FiftyOneデータセットが存在するか確認してください")
        print("利用可能なデータセット:")
        for name in fo.list_datasets():
            print(f"  - {name}")
        return 1

if __name__ == "__main__":
    exit(main())