# utils.py
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional


def _safe_strip(s: str) -> str:
    return s.strip("\ufeff").strip()


def iter_text_gz(
    path: str,
    mode: str = "line",
    min_chars: int = 30,
    limit: Optional[int] = None,
    encoding: str = "utf-8",
) -> Iterator[Dict]:
    """
    Lire un .gz contenant du texte brut et produire des "documents".

    mode:
      - "line": chaque ligne non vide = 1 document
      - "paragraph": regroupe les lignes jusqu'à une ligne vide => 1 document

    Retourne des dicts: {"id": int, "title": str, "text": str, "label": str}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    if mode not in {"line", "paragraph"}:
        raise ValueError("mode must be 'line' or 'paragraph'")

    doc_id = 0
    yielded = 0

    with gzip.open(path, "rt", encoding=encoding, errors="replace") as f:
        if mode == "line":
            for raw in f:
                if limit is not None and yielded >= limit:
                    break
                line = _safe_strip(raw)
                if not line:
                    continue
                if len(line) < min_chars:
                    continue

                title = (line[:80] + "…") if len(line) > 80 else line
                yield {"id": doc_id, "title": title, "text": line, "label": "unknown"}
                doc_id += 1
                yielded += 1

        else:  # paragraph
            buff = []
            for raw in f:
                if limit is not None and yielded >= limit:
                    break

                line = _safe_strip(raw)
                if line:
                    buff.append(line)
                    continue

                # ligne vide => fin de paragraphe
                if buff:
                    text = "\n".join(buff).strip()
                    buff = []
                    if len(text) < min_chars:
                        continue
                    title = (text[:80] + "…") if len(text) > 80 else text
                    yield {"id": doc_id, "title": title, "text": text, "label": "unknown"}
                    doc_id += 1
                    yielded += 1

            # dernier paragraphe
            if buff and (limit is None or yielded < limit):
                text = "\n".join(buff).strip()
                if len(text) >= min_chars:
                    title = (text[:80] + "…") if len(text) > 80 else text
                    yield {"id": doc_id, "title": title, "text": text, "label": "unknown"}


def save_jsonl(path: str, records: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data
