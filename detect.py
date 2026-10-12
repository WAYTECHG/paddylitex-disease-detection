from pathlib import Path
import argparse
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO prediction for qualitative analysis.")

    parser.add_argument("--weights", type=str, default="runs/train/yolov8n/weights/best.pt",
                        help="Path to trained model weights.")
    parser.add_argument("--source", type=str, default="datasets/rice_leaf_disease/images/test",
                        help="Image, folder, video, or webcam source.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for NMS.")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device, e.g. '0', or 'cpu'.")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Folder to save prediction results.")
    parser.add_argument("--name", type=str, default="yolov8n",
                        help="Prediction run name.")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save prediction labels as .txt files.")
    parser.add_argument("--save-conf", action="store_true",
                        help="Save confidence scores in prediction labels.")

    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = ROOT / args.weights
    source_path = ROOT / args.source

    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found: {weights_path}")

    # Source can also be a URL, webcam index, etc., so only convert to local path if it exists.
    source = str(source_path) if source_path.exists() else args.source

    model = YOLO(str(weights_path))

    model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(ROOT / args.project),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )


if __name__ == "__main__":
    main()
