import uuid
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection


MODEL_NAME = "google/owlvit-base-patch32"
_processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
_model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
_model.eval()

DEFAULT_LABELS = ["knife", "pistol", "gun", "handgun"]



LABEL_SYNONYMS = {
    "knife": [
        "knife",
        "kitchen knife",
        "chef knife",
        "blade",
        "dagger",
        "combat knife",
        "folding knife",
    ],
    "pistol": [
        "pistol",
        "semi-automatic pistol",
        "handgun",
        "sidearm",
        "gun",
        "firearm",
    ],
    "handgun": [
        "handgun",
        "pistol",
        "gun",
        "firearm",
        "sidearm",
    ],
    "gun": [
        "gun",
        "firearm",
        "handgun",
        "pistol",
        "revolver",
    ],
    "revolver": [
        "revolver",
        "handgun",
        "pistol",
        "gun",
        "firearm",
    ],
}


def _normalize(label: str) -> str:
    return label.strip().lower()


def expand_labels(user_labels: List[str]) -> List[str]:

    expanded: List[str] = []
    seen = set()

    for raw in user_labels:
        key = _normalize(raw)
        candidates = LABEL_SYNONYMS.get(key, [raw])

        for c in candidates:
            c_norm = _normalize(c)
            if c_norm not in seen:
                seen.add(c_norm)
                expanded.append(c)

    return expanded or DEFAULT_LABELS


def prefer_display_label(model_label: str, user_labels: List[str]) -> str:

    ml = _normalize(model_label)
    user_norm = {_normalize(x) for x in user_labels}

    if ml == "handgun":
        if "revolver" in user_norm:
            return "revolver"
        if "pistol" in user_norm:
            return "pistol"


    if ml in {"gun", "firearm"}:
        if "revolver" in user_norm:
            return "revolver"
        if "pistol" in user_norm:
            return "pistol"

    return model_label


def run_detection(
    image_path: str,
    results_dir: str,
    labels: Optional[List[str]] = None,
    threshold: float = 0.2,
) -> Tuple[str, List[dict]]:

    user_labels = labels or DEFAULT_LABELS
    labels_expanded = expand_labels(user_labels)

    img = Image.open(image_path).convert("RGB")
    inputs = _processor(text=[labels_expanded], images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = _model(**inputs)

    target_sizes = torch.tensor([img.size[::-1]])


    post_fn = getattr(_processor, "post_process_object_detection", None)
    if post_fn is None:
        post_fn = getattr(_processor, "post_process_grounded_object_detection")

    results = post_fn(
        outputs=outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    label_ids = results["labels"].cpu().tolist()

    detections: List[dict] = []
    for box, score, label_id in zip(boxes, scores, label_ids):
        idx = int(label_id)
        raw_label = labels_expanded[idx] if 0 <= idx < len(labels_expanded) else "unknown"


        display_label = prefer_display_label(raw_label, user_labels)

        detections.append(
            {
                "label": display_label,
                "score": float(score),
                "box": [float(x) for x in box],
                "prompt": raw_label,  
            }
        )


    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], width=3)
        text = f'{det["label"]}: {det["score"]:.2f}'
        draw.text((x1, max(0, y1 - 12)), text, font=font)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = results_path / out_name
    img.save(out_path, "JPEG", quality=95)

    return str(out_path), detections
