import json
import tempfile
import unittest
from pathlib import Path

from meetbot.infra.cache.hf_cache import cache_path_for, load_from_cache, save_to_cache
from meetbot.services.alignment_service import format_result_as_json, overlap


class AlignmentAndCacheTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(overlap(0, 5, 3, 8), 2.0)
        self.assertEqual(overlap(0, 1, 2, 3), 0.0)

    def test_cache_roundtrip(self):
        payload = {"x": 1, "y": [1, 2]}
        p = save_to_cache("modelA", "tests/audio.wav", payload, extra={"lang": "en"})
        loaded = load_from_cache("modelA", "tests/audio.wav", extra={"lang": "en"})
        self.assertEqual(payload, loaded)
        self.assertEqual(cache_path_for("modelA", "tests/audio.wav", extra={"lang": "en"}), p)

    def test_format_result_writes_same_shape_as_return(self):
        transcript = [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00", "text": "hello"}]
        with tempfile.TemporaryDirectory() as td:
            out = format_result_as_json(transcript, "sample.mp3", output_dir=td)
            files = list(Path(td).glob("sample*.json"))
            self.assertTrue(files)
            written = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(out, written)


if __name__ == "__main__":
    unittest.main()
