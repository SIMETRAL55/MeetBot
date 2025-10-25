from typing import List, Dict, Any
import os 
import json

def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_start is None or b_start is None:
        return 0.0
    # If either end is None, treat that side as 0 overlap
    if a_end is None or b_end is None:
        return 0.0
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    return max(0.0, right - left)

def split_transcript_chunk(
    chunk: Dict[str, Any],
    cut_times: List[float]
) -> List[Dict[str, Any]]:
    """
    Split a transcription chunk into smaller parts at diarization boundaries (cut_times).
    """
    start = chunk.get("start")
    end = chunk.get("end")
    text = chunk.get("text", "").strip()
    # If start or end is None or invalid, no split
    if start is None or end is None or end <= start:
        return [chunk]
    cuts = [t for t in cut_times if start < t < end]
    if not cuts:
        return [chunk]
    times = [start] + cuts + [end]
    total_dur = end - start
    # split text proportionally by time slices
    words = text.split()
    if not words:
        pieces = []
        for i in range(len(times)-1):
            pieces.append({"start": times[i], "end": times[i+1], "text": ""})
        return pieces
    word_dur = total_dur / len(words)
    word_times = []
    for i, w in enumerate(words):
        w_s = start + word_dur * i
        w_e = w_s + word_dur
        word_times.append((w_s, w_e, w))
    pieces = []
    for i in range(len(times)-1):
        sub_s = times[i]
        sub_e = times[i+1]
        sub_words = []
        for (ws, we, w) in word_times:
            mid = (ws + we) / 2
            if sub_s <= mid < sub_e:
                sub_words.append(w)
        pieces.append({"start": sub_s, "end": sub_e, "text": " ".join(sub_words)})
    return pieces

def build_speaker_transcript(
    diarization_segments: List[Dict[str, Any]],
    whisper_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Align whisper segments (with start, end, text) to diarization speaker segments.
    Return merged speaker transcript segments.
    """
    # Step 0: normalize whisper segments (they already have start/end/text)
    norm_whispers = []
    for w in whisper_segments:
        # only include those with valid start & end
        s = w.get("start")
        e = w.get("end")
        if s is None or e is None:
            # you could still include, but skip for now
            continue
        norm_whispers.append({"start": float(s), "end": float(e), "text": w.get("text", "").strip()})

    # Step 1: collect diarization boundary times to split whisper chunks
    cut_times = []
    for d in diarization_segments:
        ds = d.get("start")
        de = d.get("end")
        if ds is not None:
            cut_times.append(ds)
        if de is not None:
            cut_times.append(de)
    cut_times = sorted(set(cut_times))

    # Step 2: split whisper chunks at those boundary times
    split_whispers = []
    for w in norm_whispers:
        subs = split_transcript_chunk(w, cut_times)
        for s in subs:
            if s.get("end") is not None and s.get("end") > s.get("start"):
                split_whispers.append(s)

    # Step 3: assign each small whisper piece to speaker with max overlap
    assigned = []
    for w in split_whispers:
        best_spk = None
        best_ov = 0.0
        for d in diarization_segments:
            ds = d.get("start")
            de = d.get("end")
            spk = d.get("speaker")
            ov = overlap(w["start"], w["end"], ds, de)
            if ov > best_ov:
                best_ov = ov
                best_spk = spk
        if best_spk is None:
            # fallback: choose nearest diarization by start time difference
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

    # Step 4: sort by time
    assigned.sort(key=lambda x: x["start"])

    # Step 5: merge segments that are contiguous and same speaker
    merged = []
    for seg in assigned:
        if not merged:
            merged.append(seg.copy())
        else:
            prev = merged[-1]
            if seg["speaker"] == prev["speaker"]:
                # extend
                prev["end"] = max(prev["end"], seg["end"])
                if seg["text"]:
                    if prev["text"]:
                        prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
                    else:
                        prev["text"] = seg["text"]
            else:
                merged.append(seg.copy())

    return merged

def format_result_as_json(transcript: List[Dict[str, Any]], audio_path: str, output_dir: str = "results") -> str:
    speaker_map = {}
    next_id = 1
    out = []
    for item in transcript:
        sp = item.get("speaker", "unknown")
        if sp not in speaker_map:
            speaker_map[sp] = f"Speaker {next_id}"
            next_id += 1
        out.append({
            "start": round(item.get("start", 0.0), 2),
            "end": round(item.get("end", 0.0), 2),
            "speaker": speaker_map[sp],
            "text": item.get("text", "")
        })

    # ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # build output filename
    base = os.path.splitext(os.path.basename(audio_path))[0]
    fname = base + ".json"
    output_path = os.path.join(output_dir, fname)

    # if file already exists, warn or adjust name
    if os.path.exists(output_path):
        # you can choose warning or auto-rename
        print(f"⚠️ Warning: output file already exists: {output_path}")
        # Option 1: suffix with counter
        i = 1
        while True:
            alt = os.path.join(output_dir, f"{base}_{i}.json")
            if not os.path.exists(alt):
                output_path = alt
                break
            i += 1

    # write JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
        
    return out
