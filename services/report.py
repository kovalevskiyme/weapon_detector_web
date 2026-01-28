from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def history_to_excel(history: List[Dict[str, Any]], out_path: str) -> str:
    rows = []
    for item in history:
        dets = item.get("detections") or []
        det_text = "; ".join(
            [f"{d.get('label')}({float(d.get('score', 0)):.2f})" for d in dets]
        )

        rows.append(
            {
                "ts": item.get("ts"),
                "filename": item.get("filename"),
                "labels": item.get("labels"),
                "threshold": item.get("threshold"),
                "count": item.get("count"),
                "ms": item.get("ms"),
                "detections": det_text,
                "result_image": item.get("result_image"),
            }
        )

    df = pd.DataFrame(rows)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False, sheet_name="history")
    return str(out)


def history_to_pdf(history: List[Dict[str, Any]], out_path: str) -> str:

    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)


    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    ]
    font_path = None
    for p in font_paths:
        if Path(p).exists():
            font_path = p
            break


    font_name = "DejaVuSans"
    if font_path:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    else:

        font_name = "Helvetica"

    doc = SimpleDocTemplate(
        str(out),
        pagesize=landscape(A4),
        leftMargin=20,
        rightMargin=20,
        topMargin=20,
        bottomMargin=20,
    )

    styles = getSampleStyleSheet()


    styles["Title"].fontName = font_name
    styles["Normal"].fontName = font_name

    story = []
    story.append(Paragraph("Отчёт: История запросов детекции опасных предметов", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Всего записей: {len(history)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    data = [[
        "ts", "filename", "labels", "thr", "count", "ms", "detections", "result_image"
    ]]

    for item in history:
        dets = item.get("detections") or []
        det_text = ", ".join(
            [f"{d.get('label')}({float(d.get('score', 0)):.2f})" for d in dets[:5]]
        )
        if len(dets) > 5:
            det_text += ", ..."

        data.append([
            str(item.get("ts", "")),
            str(item.get("filename", "")),
            str(item.get("labels", "")),
            str(item.get("threshold", "")),
            str(item.get("count", "")),
            str(item.get("ms", "")),
            det_text,
            str(item.get("result_image", "")),
        ])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), font_name),
        ("FONTNAME", (0, 1), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    story.append(table)
    doc.build(story)

    return str(out)
