import json
import os
from typing import Any, Dict, List


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_start is None or b_start is None:
        return 0.0
    if a_end is None or b_end is None:
        return 0.0
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    return max(0.0, right - left)


def split_transcript_chunk(chunk: Dict[str, Any], cut_times: List[float]) -> List[Dict[str, Any]]:
    start = chunk.get("start")
    end = chunk.get("end")
    text = chunk.get("text", "").strip()
    if start is None or end is None or end <= start:
        return [chunk]

    cuts = [t for t in cut_times if start < t < end]
    if not cuts:
        return [chunk]

    times = [start] + cuts + [end]
    words = text.split()
    if not words:
        return [{"start": times[i], "end": times[i + 1], "text": ""} for i in range(len(times) - 1)]

    total_dur = end - start
    word_dur = total_dur / len(words)
    word_times = [(start + word_dur * i, start + word_dur * (i + 1), w) for i, w in enumerate(words)]

    pieces = []
    for i in range(len(times) - 1):
        sub_s, sub_e = times[i], times[i + 1]
        sub_words = [w for ws, we, w in word_times if sub_s <= ((ws + we) / 2) < sub_e]
        pieces.append({"start": sub_s, "end": sub_e, "text": " ".join(sub_words)})
    return pieces


def build_speaker_transcript(diarization_segments: List[Dict[str, Any]], whisper_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm_whispers = []
    for w in whisper_segments:
        s = w.get("start")
        e = w.get("end")
        if s is None or e is None:
            continue
        norm_whispers.append({"start": float(s), "end": float(e), "text": w.get("text", "").strip()})

    cut_times = sorted(
        {
            t
            for d in diarization_segments
            for t in (d.get("start"), d.get("end"))
            if t is not None
        }
    )

    split_whispers = []
    for w in norm_whispers:
        for s in split_transcript_chunk(w, cut_times):
            if s.get("end") is not None and s.get("end") > s.get("start"):
                split_whispers.append(s)

    assigned = []
    for w in split_whispers:
        best_spk = None
        best_ov = 0.0
        for d in diarization_segments:
            ov = overlap(w["start"], w["end"], d.get("start"), d.get("end"))
            if ov > best_ov:
                best_ov = ov
                best_spk = d.get("speaker")

        if best_spk is None:
            best_dist = float("inf")
            for d in diarization_segments:
                ds = d.get("start")
                if ds is None:
                    continue
                dist = abs(w["start"] - ds)
                if dist < best_dist:
                    best_dist = dist
                    best_spk = d.get("speaker")

        assigned.append({"start": w["start"], "end": w["end"], "speaker": best_spk, "text": w["text"]})

    assigned.sort(key=lambda x: x["start"])
    merged: List[Dict[str, Any]] = []
    for seg in assigned:
        if not merged:
            merged.append(seg.copy())
            continue
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"]:
            prev["end"] = max(prev["end"], seg["end"])
            if seg["text"]:
                prev["text"] = f"{prev['text'].rstrip()} {seg['text'].lstrip()}".strip()
        else:
            merged.append(seg.copy())

    return merged


def format_result_as_json(transcript: List[Dict[str, Any]], audio_path: str, output_dir: str = "results") -> List[Dict[str, Any]]:
    speaker_map: Dict[str, str] = {}
    next_id = 1
    out: List[Dict[str, Any]] = []

    for item in transcript:
        sp = item.get("speaker", "unknown")
        if sp not in speaker_map:
            speaker_map[sp] = f"Speaker {next_id}"
            next_id += 1
        out.append(
            {
                "start": round(item.get("start", 0.0), 2),
                "end": round(item.get("end", 0.0), 2),
                "speaker": speaker_map[sp],
                "text": item.get("text", ""),
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{base}.json")

    if os.path.exists(output_path):
        i = 1
        while True:
            alt = os.path.join(output_dir, f"{base}_{i}.json")
            if not os.path.exists(alt):
                output_path = alt
                break
            i += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out
