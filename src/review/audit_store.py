"""Append-only JSONL review audit chain with hash verification."""
from __future__ import annotations
import hashlib
import json
import os
import threading

_lock = threading.Lock()


def append_audit(path: str, payload: dict) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with _lock:
        previous = _last_hash(path)
        entry = {**payload, "prev_hash": previous}
        canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        entry["hash"] = hashlib.sha256((previous + canonical).encode("utf-8")).hexdigest()
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        return entry


def verify_audit(path: str) -> dict:
    previous = "0" * 64
    if not os.path.exists(path):
        return {"valid": True, "entries": 0, "first_broken_index": None}
    with open(path, encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            try:
                entry = json.loads(line)
                digest = entry.pop("hash")
                if entry.get("prev_hash") != previous:
                    raise ValueError
                canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                expected = hashlib.sha256((previous + canonical).encode("utf-8")).hexdigest()
                if digest != expected:
                    raise ValueError
                previous = digest
            except Exception:
                return {"valid": False, "entries": index, "first_broken_index": index}
    return {"valid": True, "entries": index + 1 if 'index' in locals() else 0,
            "first_broken_index": None, "last_hash": previous}


def _last_hash(path: str) -> str:
    result = verify_audit(path)
    if not result["valid"]:
        raise ValueError(f"Audit chain is broken at index {result['first_broken_index']}")
    return result.get("last_hash", "0" * 64)
