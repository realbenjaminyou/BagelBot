import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
from ultralytics import YOLOWorld

PROMPTS = [
    "a bagel sliced in half, cut side facing the camera",
    "a small round plastic tub of cream cheese with a peel back foil lid",
]

# Several descriptions per class reduce prompt wording sensitivity.
BAGEL_DESCRIPTIONS = [
    "a close-up photo of a sliced bagel covered with a thick layer of cream cheese spread",
    "a bagel half with uneven white cream cheese smeared across the cut bread surface",
    "an open bagel with creamy white cheese spread on the top of the bread",
    "a close-up photo of a plain sliced bagel with no topping on it",
    "a bagel half with bare toasted bread and an uncovered porous cut surface",
    "a bagel without cream cheese, butter or other spreads",
]

TUB_DESCRIPTIONS = [
    "a small plastic food tub containing cream cheese with an opened foil lid",
    "a single serving container of white cream cheese, foil seal partially peeled back",
    "a little round tub of plain yogurt",
    "a small tub of butter or margarine",
    "a small round container holding sauce or dip",
]

COLORS = {
    "BAGEL WITH SPREAD": (0, 180, 0),
    "PLAIN BAGEL": (255, 150, 0),
    "CREAM CHEESE TUB": (220, 70, 220),
    "BAGEL - UNCERTAIN": (0, 210, 255),
    "TUB - UNVERIFIED": (0, 210, 255),
}


def text_features(model, tokenizer, prompts, device):
    tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return emb


def scores_for_crop(crop, model, preprocess, text_emb, device):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (emb @ text_emb.T).squeeze(0).float().cpu().numpy()


def classify_bagel(crop, model, preprocess, embeddings, device, margin):
    # Compare the top matching wording for each competing visual condition.
    similarities = scores_for_crop(crop, model, preprocess, embeddings, device)
    spread = float(np.max(similarities[:3]))
    plain = float(np.max(similarities[3:]))
    gap = spread - plain
    if abs(gap) < margin:
        return "BAGEL - UNCERTAIN", gap
    return ("BAGEL WITH SPREAD" if gap > 0 else "PLAIN BAGEL"), gap


def verify_tub(crop, model, preprocess, embeddings, device, margin):
    scores = scores_for_crop(crop, model, preprocess, embeddings, device)
    cream_cheese = float(np.max(scores[:2]))
    alternatives = float(np.max(scores[2:]))
    gap = cream_cheese - alternatives
    if gap < margin:
        return "TUB - UNVERIFIED", gap
    return "CREAM CHEESE TUB", gap


def main():
    parser = argparse.ArgumentParser(description="Live bagel and cream cheese vision")
    parser.add_argument("--camera", type=int, default=0, help="Camera number, usually 0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO detection threshold")
    parser.add_argument("--margin", type=float, default=0.035,
                        help="Minimum OpenCLIP score gap before assigning a specific label")
    parser.add_argument("--every", type=int, default=3, help="Infer every Nth frame (CPU speed)")
    parser.add_argument("--output", default="captures", help="Screenshot folder")
    args = parser.parse_args()
    if args.every < 1 or args.margin < 0 or not 0 <= args.conf <= 1:
        parser.error("Require --every >= 1, --margin >= 0, and 0 <= --conf <= 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}. The first run downloads the pretrained models.")
    print("Loading YOLO-World...")
    detector = YOLOWorld("yolov8s-worldv2.pt")
    detector.set_classes(PROMPTS)

    print("Loading image-text visual classifier...")
    classifier, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    classifier.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    bagel_vectors = text_features(classifier, tokenizer, BAGEL_DESCRIPTIONS, device)
    tub_vectors = text_features(classifier, tokenizer, TUB_DESCRIPTIONS, device)

    # CAP_DSHOW works well on Windows; use the portable default elsewhere.
    import sys
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}. Try --camera 1 or close other camera apps.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Webcam ready. Press S to save a screenshot, Q or Esc to exit.")
    output = Path(args.output)
    count = 0
    previous = []
    last_infer = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Webcam did not return a frame.")
                break
            count += 1
            height, width = frame.shape[:2]
            if count == 1 or count % args.every == 0:
                previous = []
                start = time.perf_counter()
                result = detector.predict(frame, conf=args.conf, imgsz=640, verbose=False)[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        category = int(box.cls.item())
                        if category not in (0, 1):
                            continue
                        detection_score = float(box.conf.item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        if x2 - x1 < 20 or y2 - y1 < 20:
                            continue
                        crop = frame[y1:y2, x1:x2]
                        if category == 0:
                            label, gap = classify_bagel(
                                crop, classifier, preprocess, bagel_vectors, device, args.margin
                            )
                        else:
                            label, gap = verify_tub(
                                crop, classifier, preprocess, tub_vectors, device, args.margin
                            )
                        previous.append((x1, y1, x2, y2, label, detection_score, gap))
                last_infer = time.perf_counter() - start

            for x1, y1, x2, y2, label, confidence, gap in previous:
                color = COLORS[label]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 14, 2)
                cv2.putText(frame, f"{label}  detection={confidence:.2f}",
                            (x1, max(22, y1 - 11)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"center={center}  visual gap={gap:+.3f}",
                            (x1, min(height - 8, y2 + 19)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, color, 1, cv2.LINE_AA)

            cv2.rectangle(frame, (0, 0), (width, 34), (24, 24, 24), -1)
            cv2.putText(frame, f"LIVE | {device} | model {last_infer:.2f}s | S: save  Q: quit | provisional labels",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow("Bagel & Cream Cheese - Live Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                output.mkdir(parents=True, exist_ok=True)
                filename = output / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(filename), frame)
                print("Saved", filename)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
