# services/inference.py
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection


MODEL_NAME = "google/owlvit-base-patch32"
_processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
_model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
_model.eval()

# Базовые labels по умолчанию (можно менять в UI)
DEFAULT_LABELS = ["knife", "pistol", "revolver", "gun", "handgun", "blade"]

# Синонимы для "ensemble labels" (важно: не перебарщиваем, чтобы не засорять запрос)
SYNONYMS: Dict[str, List[str]] = {
    "knife": ["knife", "kitchen knife", "chef knife", "hunting knife", "dagger", "blade"],
    "blade": ["blade", "knife blade"],
    "pistol": ["pistol", "handgun", "semi-automatic pistol", "gun"],
    "handgun": ["handgun", "pistol", "gun"],
    "revolver": ["revolver", "pistol", "gun", "handgun"],
    "gun": ["gun", "pistol", "handgun", "revolver", "firearm"],
}

# Шаблоны фраз — помогают OWL-ViT (zero-shot) лучше "понять" текст
PROMPT_TEMPLATES = [
    "{x}",
    "a photo of {x}",
    "a close-up of {x}",
    "{x} on the ground",
    "{x} in a person hand",
]

# -------------------------
# Utility: IoU + NMS
# -------------------------
def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _nms(dets: List[Dict[str, Any]], iou_thr: float = 0.5) -> List[Dict[str, Any]]:
    # NMS делаем отдельно по label, чтобы knife не "гасил" pistol и т.д.
    out: List[Dict[str, Any]] = []
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets:
        by_label.setdefault(d["label"], []).append(d)

    for label, items in by_label.items():
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        keep = []
        for d in items:
            ok = True
            for k in keep:
                if _iou(d["box"], k["box"]) >= iou_thr:
                    ok = False
                    break
            if ok:
                keep.append(d)
        out.extend(keep)

    # общая сортировка по score
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# -------------------------
# Queries: ensemble + mapping (нормализация)
# -------------------------
def _build_queries(user_labels: List[str]) -> Tuple[List[str], List[str]]:
    """
    Возвращает:
      - queries: список текстовых запросов для модели
      - query_to_base: для каждого query — к какому "базовому" label он относится (что покажем пользователю)
    """
    base_labels = [x.strip().lower() for x in user_labels if x.strip()]
    if not base_labels:
        base_labels = [x.lower() for x in DEFAULT_LABELS]

    queries: List[str] = []
    query_to_base: List[str] = []

    for base in base_labels:
        # берём синонимы или сам base
        variants = SYNONYMS.get(base, [base])

        # ограничим количество вариантов, чтобы не раздувать слишком сильно
        variants = list(dict.fromkeys(variants))[:6]

        # добавляем шаблоны фраз
        for v in variants:
            for tpl in PROMPT_TEMPLATES:
                q = tpl.format(x=v)
                queries.append(q)
                query_to_base.append(base)

    # удаляем точные дубли (с сохранением соответствия query_to_base)
    seen = set()
    uniq_q = []
    uniq_map = []
    for q, b in zip(queries, query_to_base):
        key = (q, b)
        if key in seen:
            continue
        seen.add(key)
        uniq_q.append(q)
        uniq_map.append(b)

    return uniq_q, uniq_map


# -------------------------
# Core inference on one image
# -------------------------
def _infer_once(img: Image.Image, queries: List[str], q2base: List[str], threshold: float) -> List[Dict[str, Any]]:
    inputs = _processor(text=[queries], images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = _model(**inputs)

    target_sizes = torch.tensor([img.size[::-1]])  # (h,w)

    # В разных версиях transformers метод может называться по-разному:
    if hasattr(_processor, "post_process_object_detection"):
        results = _processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
    else:
        # актуальный для новых версий
        results = _processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    label_ids = results["labels"].cpu().tolist()

    dets: List[Dict[str, Any]] = []
    for box, score, lid in zip(boxes, scores, label_ids):
        base_label = q2base[lid] if 0 <= lid < len(q2base) else "object"
        dets.append(
            {
                "label": base_label,  # нормализуем: показываем базовый label из UI
                "score": float(score),
                "box": [float(x) for x in box],  # x1,y1,x2,y2
            }
        )
    return dets


# -------------------------
# Smart crops/tiles (лучший фикс для "маленьких объектов")
# -------------------------
def _generate_crops(w: int, h: int) -> List[Tuple[int, int, int, int]]:
    """
    Возвращает список прямоугольников (left, top, right, bottom) для прогонов.
    """
    crops = [(0, 0, w, h)]  # full

    # Center crop (примерно 70% кадра)
    cw = int(w * 0.7)
    ch = int(h * 0.7)
    cx1 = max(0, (w - cw) // 2)
    cy1 = max(0, (h - ch) // 2)
    crops.append((cx1, cy1, cx1 + cw, cy1 + ch))

    # 2x2 tiles с небольшим overlap
    overlap = 0.15
    tw = int(w / 2 * (1 + overlap))
    th = int(h / 2 * (1 + overlap))

    xs = [0, max(0, w - tw)]
    ys = [0, max(0, h - th)]

    for y in ys:
        for x in xs:
            crops.append((x, y, min(w, x + tw), min(h, y + th)))

    # убираем дубли
    crops = list(dict.fromkeys(crops))
    return crops


def run_detection(
    image_path: str,
    results_dir: str,
    labels: List[str] = None,
    threshold: float = 0.2,
) -> Tuple[str, List[dict]]:
    """
    Возвращает:
      - путь к сохранённому изображению с боксами
      - список детекций (label, score, box)
    """
    labels = labels or DEFAULT_LABELS
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 1) Готовим запросы (ensemble + templates) + маппинг к базовым labels
    queries, q2base = _build_queries(labels)

    # 2) Генерим smart-crops и прогоняем модель на каждом
    crops = _generate_crops(w, h)

    all_dets: List[Dict[str, Any]] = []
    for (x1, y1, x2, y2) in crops:
        crop_img = img.crop((x1, y1, x2, y2))
        dets = _infer_once(crop_img, queries, q2base, threshold=threshold)

        # переносим координаты из crop -> original
        for d in dets:
            bx1, by1, bx2, by2 = d["box"]
            d["box"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
            all_dets.append(d)

    # 3) NMS, чтобы убрать дубли
    detections = _nms(all_dets, iou_thr=0.5)

    # 4) Рисуем боксы
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
