import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def ensure_history_file(history_path: str) -> None:
    p = Path(history_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("[]", encoding="utf-8")


def append_history(history_path: str, record: Dict[str, Any]) -> None:
    ensure_history_file(history_path)
    p = Path(history_path)

    try:
        data: List[Dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = []
    except Exception:
        data = []

    data.append(record)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_history(history_path: str) -> List[Dict[str, Any]]:
    ensure_history_file(history_path)
    p = Path(history_path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
