from pathlib import Path
import shutil
import time

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from services.inference import run_detection, DEFAULT_LABELS
from services.history import append_history, read_history, now_iso
from services.report import history_to_excel, history_to_pdf


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"
TEMPLATES_DIR = BASE_DIR / "templates"


HISTORY_PATH = (BASE_DIR.parent / "data" / "history.json").resolve()

REPORTS_DIR = (BASE_DIR.parent / "data" / "reports").resolve()

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Weapon Detector (Photo MVP)")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def parse_threshold(value) -> float:

    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", ".")
    return float(s)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "labels": ", ".join(DEFAULT_LABELS),
            "result_image": None,
            "detections": None,
            "threshold": 0.2,
        },
    )


@app.post("/detect", response_class=HTMLResponse)
async def detect(
    request: Request,
    file: UploadFile = File(...),
    threshold: str = Form("0.2"),
    labels: str = Form("knife,pistol,gun,handgun"),
):
    thr = parse_threshold(threshold)


    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    upload_path = UPLOADS_DIR / safe_name

    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    label_list = [x.strip() for x in labels.split(",") if x.strip()]

    t0 = time.perf_counter()
    out_path, detections = run_detection(
        image_path=str(upload_path),
        results_dir=str(RESULTS_DIR),
        labels=label_list,
        threshold=thr,
    )
    ms = int((time.perf_counter() - t0) * 1000)


    result_rel = "/static/results/" + Path(out_path).name


    append_history(
        str(HISTORY_PATH),
        {
            "ts": now_iso(),
            "filename": safe_name,
            "labels": ", ".join(label_list),
            "threshold": thr,
            "count": len(detections),
            "ms": ms,
            "detections": detections,
            "result_image": result_rel,
        },
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "labels": ", ".join(label_list),
            "result_image": result_rel,
            "detections": detections,
            "threshold": thr,
        },
    )


@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    items = read_history(str(HISTORY_PATH))
    items = list(reversed(items))[:50]  

    return templates.TemplateResponse(
        "history.html",
        {"request": request, "items": items},
    )


@app.post("/history/clear")
def clear_history():

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text("[]", encoding="utf-8")
    return RedirectResponse(url="/history", status_code=303)


@app.get("/export/excel")
def export_excel():
    items = read_history(str(HISTORY_PATH))
    out_path = REPORTS_DIR / "history.xlsx"
    history_to_excel(items, str(out_path))

    return FileResponse(
        path=str(out_path),
        filename="history.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/export/pdf")
def export_pdf():
    items = read_history(str(HISTORY_PATH))
    out_path = REPORTS_DIR / "history.pdf"
    history_to_pdf(items, str(out_path))

    return FileResponse(
        path=str(out_path),
        filename="history.pdf",
        media_type="application/pdf",
    )
