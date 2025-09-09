#!/usr/bin/env python3
"""
Open Images Dataset (OID) からBeetleクラスのデータをダウンロード
FiftyOneを使用してデータを取得し、軽量テスト用の設定も提供
"""

import os
import argparse
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path

def download_beetle_data(train_samples=200, val_samples=50, test_mode=True):
    """
    OIDからBeetleクラスのデータをダウンロード
    
    Args:
        train_samples (int): 訓練用サンプル数
        val_samples (int): 検証用サンプル数  
        test_mode (bool): 軽量テストモード
    """
    
    print(f"=== Open Images Dataset Beetle データダウンロード開始 ===")
    print(f"Test Mode: {test_mode}")
    print(f"Train samples: {train_samples}, Val samples: {val_samples}")
    
    # データセット保存ディレクトリ作成
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    try:
        # 訓練データのダウンロード
        print("\n--- 訓練データダウンロード中 ---")
        train_ds = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=["Beetle"],
            max_samples=train_samples,
            dataset_name=f"oid_beetle_train_{train_samples}"
        )
        
        print(f"訓練データ: {len(train_ds)} samples downloaded")
        
        # 検証データのダウンロード  
        print("\n--- 検証データダウンロード中 ---")
        val_ds = foz.load_zoo_dataset(
            "open-images-v7", 
            split="validation",
            label_types=["detections"],
            classes=["Beetle"],
            max_samples=val_samples,
            dataset_name=f"oid_beetle_val_{val_samples}"
        )
        
        print(f"検証データ: {len(val_ds)} samples downloaded")
        
        # データ統計表示
        print(f"\n=== ダウンロード完了 ===")
        print(f"Train: {len(train_ds)} images")
        print(f"Val: {len(val_ds)} images")
        print(f"Total: {len(train_ds) + len(val_ds)} images")
        
        # FiftyOneでデータセット確認（オプション）
        if test_mode:
            print("\n--- サンプルデータ確認 ---")
            print("Train dataset info:")
            train_ds.print_summary()
            
            # 最初のサンプルの詳細
            first_sample = train_ds.first()
            if first_sample.detections:
                print(f"First sample detections: {len(first_sample.detections.detections)}")
                
        return train_ds, val_ds
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("FiftyOneの初回実行時はダウンロードに時間がかかる場合があります")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="OID Beetle data downloader")
    parser.add_argument("--train-samples", type=int, default=200, 
                       help="Number of training samples (default: 200 for test)")
    parser.add_argument("--val-samples", type=int, default=50,
                       help="Number of validation samples (default: 50 for test)")
    parser.add_argument("--full-scale", action="store_true",
                       help="Download full scale dataset (1200 train, 300 val)")
    parser.add_argument("--no-test", action="store_true",
                       help="Disable test mode verbose output")
    
    args = parser.parse_args()
    
    # フルスケールモードの場合
    if args.full_scale:
        train_samples = 1200
        val_samples = 300
        test_mode = False
        print("🚀 Full scale mode enabled")
    else:
        train_samples = args.train_samples
        val_samples = args.val_samples
        test_mode = not args.no_test
        print("🧪 Test mode enabled (lightweight)")
    
    # ダウンロード実行
    train_ds, val_ds = download_beetle_data(
        train_samples=train_samples,
        val_samples=val_samples, 
        test_mode=test_mode
    )
    
    if train_ds and val_ds:
        print("✅ データダウンロード成功！")
        print("次のステップ: python scripts/02_clean_and_export_yolo.py")
    else:
        print("❌ データダウンロード失敗")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())