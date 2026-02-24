from __future__ import annotations

import os
from pathlib import Path

from qwen_tts.tts_api import TTSEngine

MODULE_PATH = "qwen_tts.tts_api"
CLASS_NAME = "TTSEngine"
LOAD_METHOD = "tts_load"
INFER_METHOD = "tts_inference"

MODEL_PATH = os.environ.get("QWEN_TTS_TEST_MODEL_PATH", "models/Qwen3-TTS-12Hz-0.6B-Base")
REFERENCE_AUDIO_PATH = os.environ.get(
    "QWEN_TTS_TEST_REFERENCE_AUDIO_PATH",
    "qwen3_test_qwen3_tts_dataset/data/ref_audio.wav",
)
REFERENCE_TEXT = os.environ.get("QWEN_TTS_TEST_REFERENCE_TEXT") or None
TEST_TEXT = "This is a validation run from test_tts_api.py."


def main() -> int:
    engine = TTSEngine()

    output_dir = Path("tts_test_outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "integration_test.wav"

    engine.tts_load(
        model_path=MODEL_PATH,
        reference_audio_path=REFERENCE_AUDIO_PATH,
        reference_text=REFERENCE_TEXT,
    )

    result = engine.tts_inference(
        text=TEST_TEXT,
        output_path=str(output_path),
        model_path=MODEL_PATH,
        reference_audio_path=REFERENCE_AUDIO_PATH,
        reference_text=REFERENCE_TEXT,
        x_vector_only_mode=True,
        do_sample=False,
        max_new_tokens=1024,
    )
    returned_path = result if isinstance(result, Path) else Path(result)
    if not returned_path.is_absolute():
        returned_path = (Path.cwd() / returned_path).resolve()

    if not returned_path.exists():
        raise FileNotFoundError(f"Expected output file does not exist: {returned_path}")
    if returned_path.stat().st_size <= 0:
        raise ValueError(f"Output file is empty: {returned_path}")

    print(f"SUCCESS: generated audio at {returned_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
