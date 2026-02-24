# API Documentation

## Purpose
Document the class-based callable wrapper API (`TTSEngine`) for using Qwen 3 TTS as a simple load-then-generate component across Base, CustomVoice, and VoiceDesign models.

## Key inputs/config
- `model_path` (`str | Path`): Local directory for the Qwen3-TTS model to load. Required for first load/inference in offline mode.
- `reference_audio_path` (`str | Path | None`): Local reference audio file for Base voice-clone models. Optional for `custom_voice` and `voice_design`.
- `reference_text` (`str | None`): Optional transcript for the reference audio. Not required when Base inference uses `x_vector_only_mode=True`.
- `device` (`str | None`): Runtime target, e.g. `cuda`, `cuda:0`, `cpu`, `mps`. Defaults to `cuda:0` if available, otherwise `cpu`.
- `dtype` (`str | None`): Optional load dtype (`float32`, `float16`, `bfloat16`, aliases supported).
- `attn_implementation` (`str | None`): Optional attention backend forwarded to Qwen model loading.
- `local_files_only` (`bool`): Defaults to `True` in the wrapper for offline/local-path usage.
- `text` (`str | list[str]`): Input text (single or batch) to synthesize. Empty values are rejected.
- `output_path` (`str | Path | None`): Explicit output file path to write generated audio.
- `output_dir` (`str | Path | None`): Optional output directory if `output_path` is not provided.
- `language` (`str | list[str]`): Language hint(s), defaults to `"Auto"`.
- `speaker` (`str | list[str] | None`): Required for `custom_voice` models.
- `instruct` (`str | list[str] | None`): Required for `voice_design`; optional for `custom_voice`.
- `x_vector_only_mode` (`bool`): Base-model voice clone option, defaults to `True`.
- `seed` (`int | None`): Seeds Torch/Torch CUDA before inference for more repeatable behavior.
- `**generate_kwargs`: Additional generation arguments forwarded to the underlying Qwen generator method for the loaded model type.

## Key outputs/behavior
- `TTSEngine.tts_load(...)` loads and caches a `Qwen3TTSModel` instance using `Qwen3TTSModel.from_pretrained(...)`.
- `TTSEngine.tts_inference(...)` dispatches to one of:
  - `generate_voice_clone(...)` for Base models
  - `generate_custom_voice(...)` for CustomVoice models
  - `generate_voice_design(...)` for VoiceDesign models
- Output directories are created automatically.
- The wrapper writes the first generated waveform (`wavs[0]`) to the requested output path.
- The returned path is validated to exist and be non-empty before returning.
- Repeated `tts_load(...)` calls with the same model/device/dtype/attention settings are cached.
- The wrapper is designed for offline/local paths and rejects URL `reference_audio_path` values.

## Public Class

### `qwen_tts.tts_api.TTSEngine`

Stable class-based wrapper for offline/local Qwen3-TTS inference.

### Constructor

```python
TTSEngine()
```

No required constructor arguments.

### `get_loaded_model_info()`

Returns lightweight metadata for the currently loaded model:
- `model_path`
- `model_type` (`base`, `custom_voice`, `voice_design`)
- `model_size` (for example `1b7`)
- `device`
- `dtype`

## Methods

### `tts_load(...)`

```python
def tts_load(
    *,
    model_path,
    reference_audio_path=None,
    reference_text=None,
    device=None,
    dtype=None,
    attn_implementation=None,
    local_files_only=True,
) -> None
```

Purpose:
- Load and cache a local Qwen3-TTS model and store optional reference-audio state used by Base voice cloning.

Parameter notes:
- `model_path`: Must exist and be a local directory in this wrapper.
- `reference_audio_path`: Optional at load time, but required for Base-model inference.
- `reference_text`: Optional reference transcript for Base voice clone.
- `device`: If omitted, wrapper auto-selects `cuda:0` when CUDA is available, otherwise `cpu`.
- `dtype`: String aliases are normalized to Torch dtypes.
- `local_files_only`: Defaults to `True`, supporting offline behavior.

Model-specific behavior:
- Base model loads will raise if no `reference_audio_path` is provided.
- CustomVoice / VoiceDesign models can load without a reference audio path.

Errors:
- `FileNotFoundError` for missing model directory or reference audio file.
- `ValueError` for invalid path type (e.g., non-directory model path, URL reference in offline mode).
- `RuntimeError` if `Qwen3TTSModel.from_pretrained(...)` fails.

### `tts_inference(...)`

```python
def tts_inference(
    *,
    text,
    output_path=None,
    output_dir=None,
    model_path=None,
    reference_audio_path=None,
    reference_text=None,
    language="Auto",
    speaker=None,
    instruct=None,
    x_vector_only_mode=True,
    seed=1234,
    **generate_kwargs,
) -> Path
```

Purpose:
- Generate speech audio from text using the loaded (or lazily loaded) Qwen3-TTS model and write the result to a file path.

Parameter notes:
- `text`: Required non-empty string or list of non-empty strings.
- `output_path`: Preferred explicit file path. If omitted, wrapper falls back to `output_dir / "qwen3_tts_output.wav"` or `tts_outputs/qwen3_tts_output.wav`.
- `model_path`: Optional override for lazy load or model switching.
- `reference_audio_path`: Optional override; required for Base models unless already stored.
- `reference_text`: Optional override for stored reference transcript.
- `language`: `str` or `list[str]`, forwarded to the selected Qwen generation method.
- `speaker`: Required for `custom_voice` models (for example `"Vivian"`).
- `instruct`: Required for `voice_design`; optional for `custom_voice`.
- `x_vector_only_mode`: Base-model only option; ignored by CustomVoice/VoiceDesign dispatch.
- `seed`: If not `None`, seeds `torch.manual_seed(...)` and `torch.cuda.manual_seed_all(...)` (when CUDA is available).
- `**generate_kwargs`: Passed directly to the selected generator:
  - Base -> `generate_voice_clone(...)`
  - CustomVoice -> `generate_custom_voice(...)`
  - VoiceDesign -> `generate_voice_design(...)`

Behavior details:
- If the engine is not already loaded, `tts_inference(...)` calls `tts_load(...)` internally when enough inputs are provided.
- The wrapper auto-detects model type from the loaded Qwen model metadata and dispatches accordingly.
- The wrapper writes only the first waveform (`wavs[0]`) returned by Qwen3-TTS.
- The return value is an absolute `Path` to the written audio file.

Model-type requirements enforced by the wrapper:
- Base (`model_type == "base"`):
  - requires `reference_audio_path` (load-time or inference-time)
  - uses `reference_text` when provided
- CustomVoice (`model_type == "custom_voice"`):
  - requires `speaker`
  - accepts optional `instruct`
- VoiceDesign (`model_type == "voice_design"`):
  - requires `instruct`

Errors:
- `ValueError` for empty text, invalid text list items, or missing model-specific inputs (`speaker` / `instruct` / Base reference audio).
- `RuntimeError` if synthesis fails or no waveforms are returned.
- `RuntimeError` if file writing fails.
- `FileNotFoundError` / `ValueError` if output validation fails after generation.

## Minimal Usage Examples

### Base (Voice Clone)

```python
from qwen_tts.tts_api import TTSEngine

engine = TTSEngine()
engine.tts_load(
    model_path="models/Qwen3-TTS-12Hz-1.7B-Base",
    reference_audio_path="qwen3_test_qwen3_tts_dataset/data/ref_audio.wav",
)
audio_path = engine.tts_inference(
    text="This is a Base-model voice clone call.",
    output_path="tts_test_outputs/base_example.wav",
    x_vector_only_mode=True,
)
print(audio_path)
```

### CustomVoice

```python
from qwen_tts.tts_api import TTSEngine

engine = TTSEngine()
engine.tts_load(model_path="models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
audio_path = engine.tts_inference(
    text="This is a CustomVoice call.",
    output_path="tts_test_outputs/custom_voice_example.wav",
    speaker="Vivian",
    language="English",
    instruct="Speak clearly in a calm tone.",
)
print(audio_path)
```

CustomVoice model note:
- `TTSEngine` works with the provided Qwen CustomVoice models (for example, the published Hugging Face / local-downloaded `Qwen3-TTS-12Hz-1.7B-CustomVoice`).
- It is also an expected wrapper target for a CustomVoice model directory produced by your own fine-tune run, as long as it loads successfully through `Qwen3TTSModel.from_pretrained(...)` and reports a compatible `custom_voice` model type.

### VoiceDesign

```python
from qwen_tts.tts_api import TTSEngine

engine = TTSEngine()
engine.tts_load(model_path="models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
audio_path = engine.tts_inference(
    text="This is a VoiceDesign call.",
    output_path="tts_test_outputs/voice_design_example.wav",
    language="English",
    instruct="A warm narrator voice with steady pacing and gentle energy.",
)
print(audio_path)
```

## Related files
- `qwen_tts/tts_api.py` (wrapper implementation)
- `qwen_tts/inference/qwen3_tts_model.py` (underlying Qwen3 TTS wrapper used by `TTSEngine`)
- `qwen_tts/__init__.py` (exports `TTSEngine`)
- `test_tts_api.py` (project-root integration test for the Base-model wrapper flow)
