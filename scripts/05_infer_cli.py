#!/usr/bin/env python3
"""
Beetle Detection 推論CLI
学習済みYOLOv8モデルを使用して画像からBeetleを検出
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import time

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

def load_model(model_path: str) -> YOLO:
    """
    学習済みYOLOv8モデルを読み込み
    
    Args:
        model_path: モデルファイルパス (.pt)
        
    Returns:
        YOLO: 読み込み済みモデル
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Device: {'GPU' if next(model.model.parameters()).is_cuda else 'CPU'}")
    print(f"   Classes: {model.names}")
    
    return model

def process_single_image(model: YOLO, image_path: str, conf: float = 0.25, 
                        save_result: bool = True, output_dir: str = "inference_results") -> Dict[str, Any]:
    """
    単一画像での推論実行
    
    Args:
        model: YOLOv8モデル
        image_path: 入力画像パス
        conf: 信頼度閾値
        save_result: 結果画像を保存するか
        output_dir: 出力ディレクトリ
        
    Returns:
        Dict: 検出結果情報
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # 推論実行
    start_time = time.time()
    results = model(image_path, conf=conf, verbose=False)
    inference_time = time.time() - start_time
    
    result = results[0]
    
    # 検出結果の解析
    detections = []
    if result.boxes is not None:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            detection = {
                "class_id": int(box.cls.cpu().numpy()[0]),
                "class_name": model.names[int(box.cls.cpu().numpy()[0])],
                "confidence": float(box.conf.cpu().numpy()[0]),
                "bbox": box.xyxy.cpu().numpy()[0].tolist(),  # [x1, y1, x2, y2]
            }
            detections.append(detection)
    
    # 結果画像保存
    output_path = None
    if save_result and detections:
        os.makedirs(output_dir, exist_ok=True)
        
        # アノテーション付き画像生成
        annotated_img = result.plot()
        
        # 保存パス生成
        input_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{input_name}_result.jpg")
        
        # BGR -> RGB変換して保存
        cv2.imwrite(output_path, annotated_img)
    
    # 結果情報
    result_info = {
        "image_path": image_path,
        "inference_time_ms": inference_time * 1000,
        "num_detections": len(detections),
        "detections": detections,
        "output_path": output_path
    }
    
    return result_info

def process_batch_images(model: YOLO, image_paths: List[str], conf: float = 0.25,
                        save_results: bool = True, output_dir: str = "inference_results") -> List[Dict[str, Any]]:
    """
    複数画像での一括推論
    
    Args:
        model: YOLOv8モデル
        image_paths: 入力画像パスのリスト
        conf: 信頼度閾値
        save_results: 結果画像を保存するか
        output_dir: 出力ディレクトリ
        
    Returns:
        List[Dict]: 各画像の検出結果情報
    """
    
    print(f"=== Batch inference: {len(image_paths)} images ===")
    
    all_results = []
    total_detections = 0
    total_time = 0
    
    for i, image_path in enumerate(image_paths, 1):
        try:
            print(f"Processing [{i}/{len(image_paths)}]: {Path(image_path).name}")
            
            result_info = process_single_image(
                model, image_path, conf, save_results, output_dir
            )
            
            all_results.append(result_info)
            total_detections += result_info["num_detections"]
            total_time += result_info["inference_time_ms"]
            
            print(f"  Detections: {result_info['num_detections']}, "
                  f"Time: {result_info['inference_time_ms']:.2f}ms")
            
        except Exception as e:
            print(f"  ❌ Error processing {image_path}: {e}")
            continue
    
    # 統計情報
    avg_time = total_time / len(all_results) if all_results else 0
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    
    print(f"\n=== Batch Results ===")
    print(f"Processed images: {len(all_results)}/{len(image_paths)}")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {avg_time:.2f}ms")
    print(f"Average FPS: {avg_fps:.2f}")
    
    return all_results

def save_results_json(results: List[Dict[str, Any]], output_path: str):
    """
    推論結果をJSONファイルに保存
    
    Args:
        results: 推論結果のリスト
        output_path: 出力JSONファイルパス
    """
    
    # サマリー情報追加
    summary = {
        "total_images": len(results),
        "total_detections": sum(r["num_detections"] for r in results),
        "avg_inference_time_ms": np.mean([r["inference_time_ms"] for r in results]),
        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_path}")

def get_image_files(source: str) -> List[str]:
    """
    指定されたパスから画像ファイルを取得
    
    Args:
        source: ファイル/ディレクトリパス、またはワイルドカード
        
    Returns:
        List[str]: 画像ファイルパスのリスト
    """
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    source_path = Path(source)
    
    if source_path.is_file():
        # 単一ファイル
        if source_path.suffix.lower() in image_extensions:
            image_files.append(str(source_path))
    elif source_path.is_dir():
        # ディレクトリ
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        image_files = [str(f) for f in image_files]
    else:
        # ワイルドカードパターン
        from glob import glob
        image_files = glob(source)
        image_files = [f for f in image_files if Path(f).suffix.lower() in image_extensions]
    
    return sorted(image_files)

def main():
    parser = argparse.ArgumentParser(
        description="Beetle Detection Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 単一画像での推論
  python scripts/05_infer_cli.py --model runs/detect/train/weights/best.pt --source test.jpg

  # ディレクトリ内の全画像で推論  
  python scripts/05_infer_cli.py --model runs/detect/train/weights/best.pt --source images/

  # ワイルドカード指定
  python scripts/05_infer_cli.py --model runs/detect/train/weights/best.pt --source "images/*.jpg"

  # 信頼度閾値調整
  python scripts/05_infer_cli.py --model best.pt --source images/ --conf 0.5

  # 結果保存なし（速度重視）
  python scripts/05_infer_cli.py --model best.pt --source images/ --no-save
        """
    )
    
    parser.add_argument("--model", "-m", required=True,
                       help="Path to trained YOLOv8 model (.pt)")
    parser.add_argument("--source", "-s", required=True,
                       help="Input source (image file, directory, or wildcard)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", "-o", default="inference_results",
                       help="Output directory for results (default: inference_results)")
    parser.add_argument("--no-save", action="store_true",
                       help="Do not save annotated images (faster)")
    parser.add_argument("--save-json", action="store_true",
                       help="Save results as JSON file")
    parser.add_argument("--device", default="",
                       help="Device to use (cpu, 0, 1, etc.). Auto-detect if empty")
    
    args = parser.parse_args()
    
    try:
        # モデル読み込み
        model = load_model(args.model)
        
        # デバイス設定
        if args.device:
            model.to(args.device)
            print(f"Using device: {args.device}")
        
        # 入力画像取得
        image_files = get_image_files(args.source)
        
        if not image_files:
            print(f"❌ No image files found in: {args.source}")
            return 1
            
        print(f"Found {len(image_files)} image(s)")
        
        # 推論実行
        save_results = not args.no_save
        
        if len(image_files) == 1:
            # 単一画像
            result = process_single_image(
                model, image_files[0], args.conf, save_results, args.output_dir
            )
            results = [result]
            
            print(f"\n=== Results ===")
            print(f"Image: {result['image_path']}")
            print(f"Detections: {result['num_detections']}")
            print(f"Inference time: {result['inference_time_ms']:.2f}ms")
            
            if result['detections']:
                print("Detected objects:")
                for i, det in enumerate(result['detections'], 1):
                    print(f"  {i}. {det['class_name']}: {det['confidence']:.3f}")
            
        else:
            # 複数画像
            results = process_batch_images(
                model, image_files, args.conf, save_results, args.output_dir
            )
        
        # JSON保存（オプション）
        if args.save_json:
            json_path = os.path.join(args.output_dir, "inference_results.json")
            os.makedirs(args.output_dir, exist_ok=True)
            save_results_json(results, json_path)
        
        print(f"\n✅ Inference completed successfully!")
        if save_results:
            print(f"📁 Results saved in: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())