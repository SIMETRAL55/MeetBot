# prepare_docs.py
"""
Robust preparer for diarization/transcript outputs.

Accepts either:
 - a JSON array file: [ {start,end,speaker,text}, ... ]
 - a JSONL file: one JSON object per line

Produces: prepared/<audio_basename>.jsonl (one JSON object per line)

Each output doc:
{
  "id": "<audio_basename>_<idx>",
  "text": "<SPEAKER_LABEL: text>"   # speaker label included when available
  "metadata": {
     "audio_file": "<audio_basename>",
     "speaker": "<SPEAKER_LABEL or unknown>",
     "start": <float|null>,
     "end": <float|null>,
     "chunk_id": idx
  }
}
"""
from pathlib import Path
import json
import re
from typing import List, Dict, Any
import logging
import sys

logger = logging.getLogger("prepare_docs")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Za-z0-9_\- ]{1,40}):\s*(.*)$")  # captures "SPEAKER_00: rest..."


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load either a JSON array or a JSONL file.
    Returns list of dicts (raw segments).
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    first_char = text.lstrip()[0]
    if first_char == "[":
        try:
            arr = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array in {path}: {e}") from e
        if not isinstance(arr, list):
            raise ValueError(f"Expected JSON array in {path}")
        return arr

    docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e.msg}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {lineno} of {path}, got {type(obj)}")
            docs.append(obj)
    return docs


def _extract_speaker_and_text(raw_text: str):
    """
    If text begins with a speaker prefix "SPEAKER_00: ..." extract it.
    Returns (speaker_or_none, cleaned_text)
    """
    if not raw_text:
        return None, ""
    m = SPEAKER_PREFIX_RE.match(raw_text)
    if m:
        speaker = m.group(1).strip()
        text = m.group(2).strip()
        return speaker, text
    return None, raw_text.strip()


def _get_timestamp(seg: Dict[str, Any]):
    """
    Robust extraction of start/end from various possible shapes:
      - seg['start'], seg['end']
      - seg['timestamp'] = [start, end]
      - seg['ts'] or seg['time']
    Returns (start_or_None, end_or_None)
    """
    # direct keys
    if "start" in seg or "end" in seg:
        try:
            s = seg.get("start")
            e = seg.get("end")
            return (float(s) if s is not None else None, float(e) if e is not None else None)
        except Exception:
            pass

    # timestamp array
    ts = seg.get("timestamp") or seg.get("timestamps") or seg.get("time") or seg.get("ts")
    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
        try:
            return (float(ts[0]) if ts[0] is not None else None, float(ts[1]) if ts[1] is not None else None)
        except Exception:
            pass

    # not available
    return None, None


def prepare_documents(result_json_path: str, out_dir: str = "prepared") -> List[Dict[str, Any]]:
    p = Path(result_json_path)
    if not p.exists():
        raise FileNotFoundError(f"result json not found: {result_json_path}")

    audio_basename = p.stem
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_file = out_dir_p / f"{audio_basename}.jsonl"

    try:
        raw_segments = _load_json_or_jsonl(p)
    except Exception as e:
        logger.error("Failed to load input file: %s", e)
        raise

    docs: List[Dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments):
        # pull text (prefer explicit 'text' field)
        raw_text = seg.get("text") or seg.get("transcript") or seg.get("chunk_text") or ""
        raw_text = raw_text.strip()

        # pull speaker if explicitly present
        speaker = seg.get("speaker") or seg.get("spk") or seg.get("speaker_label")
        if speaker is None:
            # try to extract from the raw_text if text starts with "SPEAKER_xx: ..."
            extracted_spk, cleaned = _extract_speaker_and_text(raw_text)
            if extracted_spk:
                speaker = extracted_spk
                raw_text = cleaned  # remove the speaker prefix from text
        # normalize empty speaker
        if not speaker:
            speaker = "unknown"

        # timestamps
        start, end = _get_timestamp(seg)

        # construct the text used for embedding/search: include speaker label if we have it and not 'unknown'
        if speaker and speaker.lower() != "unknown":
            doc_text = f"{speaker}: {raw_text}".strip()
        else:
            doc_text = raw_text

        doc = {
            "id": f"{audio_basename}_{idx}",
            "text": doc_text,
            "metadata": {
                "audio_file": audio_basename,
                "speaker": speaker,
                "start": float(start) if start is not None else None,
                "end": float(end) if end is not None else None,
                "chunk_id": idx
            }
        }
        docs.append(doc)

    # write JSONL cleanly (one JSON object per line)
    with out_file.open("w", encoding="utf-8") as outfh:
        for d in docs:
            outfh.write(json.dumps(d, ensure_ascii=False) + "\n")

    logger.info("Prepared %d documents -> %s", len(docs), out_file)
    return docs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare diarization JSON -> prepared JSONL")
    parser.add_argument("result_json", help="Path to diarization/transcript JSON (array) or JSONL")
    parser.add_argument("--out-dir", default="prepared", help="Directory to write prepared JSONL")
    args = parser.parse_args()

    try:
        prepare_documents(args.result_json, out_dir=args.out_dir)
    except Exception as exc:
        logger.error("prepare_documents failed: %s", exc)
        sys.exit(1)
