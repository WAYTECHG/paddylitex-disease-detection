from pathlib import Path
import argparse
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for paddy disease detection.")

    parser.add_argument("--model", type=str, default="cfg/training/yolov8n.yaml",
                        help="Path to model YAML or .pt file.")
    parser.add_argument("--data", type=str, default="data/rice_leaf_disease.yaml",
                        help="Path to dataset YAML.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size.")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device, e.g. '0', or 'cpu'.")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Folder to save training results.")
    parser.add_argument("--name", type=str, default="yolov8n_scratch_paddy",
                        help="Experiment name.")
    parser.add_argument("--patience", type=int, default=100,
                        help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last checkpoint.")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained weights only if the selected model supports it.")

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = ROOT / args.model
    data_path = ROOT / args.data

    if not model_path.exists() and not args.model.endswith(".pt"):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path) if model_path.exists() else args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(ROOT / args.project),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        plots=True,
        resume=args.resume,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
