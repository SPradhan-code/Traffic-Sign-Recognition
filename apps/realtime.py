"""Webcam demo:  python apps/realtime.py [--camera 0] [--threshold 0.8]
Hold a sign inside the green box. Press q to quit."""
import argparse
import sys
from pathlib import Path

# Ensure project root and src/ are in sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402

from tsr.config import load_config  # noqa: E402
from tsr.predictor import Predictor  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    cfg = load_config()
    predictor = Predictor(cfg["model_path"], cfg["img_size"])
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
        roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        top = predictor.predict(roi, top_k=1)[0]
        if top["confidence"] >= args.threshold:
            text = f'{top["label"]} ({top["confidence"]:.0%})'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Traffic Sign Recognition (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
