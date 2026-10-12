from pathlib import Path
import argparse
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model on val/test split.")

    parser.add_argument("--weights", type=str, default="runs/train/yolov8n/weights/best.pt",
                        help="Path to trained model weights.")
    parser.add_argument("--data", type=str, default="data/rice_leaf_disease.yaml",
                        help="Path to dataset YAML.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size.")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size. Use batch=1 for final test if you want stable latency/memory.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device, e.g. '0', or 'cpu'.")
    parser.add_argument("--project", type=str, default="runs/test",
                        help="Folder to save test/evaluation results.")
    parser.add_argument("--name", type=str, default="yolov8n_scratch_paddy",
                        help="Evaluation run name.")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold used during evaluation.")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold used for NMS during evaluation.")

    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = ROOT / args.weights
    data_path = ROOT / args.data

    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found: {weights_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(weights_path))

    results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(ROOT / args.project),
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        plots=True,
    )

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Split: {args.split}")
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")


if __name__ == "__main__":
    main()
