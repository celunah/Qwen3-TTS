# Architecture

## Purpose
Describe the class-based wrapper architecture added for callable Qwen 3 TTS usage, with emphasis on the load-then-generate flow, multi-model dispatch, and how it maps onto the existing `qwen_tts` code.

## Key inputs/config
- `model_path`: Local Qwen3-TTS model directory selected by the caller at runtime.
- `reference_audio_path`: Local reference audio file used for Base voice-clone prompts (not required for CustomVoice/VoiceDesign).
- `reference_text`: Optional transcript for reference audio.
- `device`, `dtype`, `attn_implementation`, `local_files_only`: Wrapper load-time options forwarded into model loading.
- `text`, `language`, `speaker`, `instruct`, `x_vector_only_mode`, `seed`, `generate_kwargs`: Wrapper inference options.
- `output_path` / `output_dir`: Destination control for generated audio files.

## Key outputs/behavior
- Provides a stable, black-box style API (`TTSEngine`) for local/offline inference across Base, CustomVoice, and VoiceDesign models.
- Encapsulates direct access to `Qwen3TTSModel.from_pretrained(...)` and the model-specific generation methods.
- Caches the loaded model, model metadata, and reference settings inside the engine instance.
- Returns a validated, non-empty generated audio file path.

## High-Level Flow

### 1. Load phase (`TTSEngine.tts_load`)
- Validates `model_path` as an existing local directory.
- Validates `reference_audio_path` as an existing local file and rejects URL paths (offline wrapper behavior) when provided.
- Resolves runtime device (`cuda:0` if available, else `cpu`, unless caller overrides).
- Normalizes `dtype` string aliases to Torch dtypes.
- Calls `Qwen3TTSModel.from_pretrained(...)` with:
  - `device_map`
  - `dtype`
  - `local_files_only=True` by default
  - optional `attn_implementation`
- Stores loaded state on the engine instance:
  - `_tts`
  - `_loaded_model_path`
  - `_reference_audio_path`
  - `_reference_text`
  - `_device`
  - `_dtype_name`
  - `_attn_implementation`
  - `_local_files_only`
  - `_loaded_model_type`
  - `_loaded_model_size`
- Base model loads require `reference_audio_path`; CustomVoice and VoiceDesign loads do not.

### 2. Inference phase (`TTSEngine.tts_inference`)
- Validates non-empty `text`.
- If engine is not loaded (or caller passes overrides), re-enters `tts_load(...)`.
- Resolves an absolute output path and creates parent directories.
- Seeds Torch RNGs when `seed` is provided.
- Dispatches based on the loaded model type (`_loaded_model_type`):
  - `base` -> `self._tts.generate_voice_clone(...)` (requires Base reference audio)
  - `custom_voice` -> `self._tts.generate_custom_voice(...)` (requires `speaker`)
  - `voice_design` -> `self._tts.generate_voice_design(...)` (requires `instruct`)
- Writes the first returned waveform (`wavs[0]`) to `output_path` with `soundfile.write(...)`.
- Validates file existence and non-zero size, then returns the absolute path.

## Component Map (Wrapper-Focused)

### `qwen_tts/tts_api.py`
Purpose:
- Owns the public wrapper class `TTSEngine`.

Key responsibilities:
- Input validation for local/offline path usage
- Device/dtype normalization
- Lazy/cached model loading
- Inference execution and output-file writing
- Error normalization with user-facing messages

### `qwen_tts/inference/qwen3_tts_model.py`
Purpose:
- Existing repository wrapper around Qwen3 TTS models and processors.

Relevant wrapper integration points:
- `Qwen3TTSModel.from_pretrained(...)` for model+processor loading
- `Qwen3TTSModel.generate_voice_clone(...)` for Base voice cloning
- `Qwen3TTSModel.generate_custom_voice(...)` for CustomVoice generation
- `Qwen3TTSModel.generate_voice_design(...)` for VoiceDesign generation

Notes:
- `TTSEngine` intentionally wraps these calls so callers do not need to import low-level repo internals.

### `qwen_tts/__init__.py`
Purpose:
- Exposes `TTSEngine` as part of the package import surface.

### `test_tts_api.py`
Purpose:
- Project-root integration test that exercises the Base-model wrapper path:
  - import
  - `tts_load(...)`
  - `tts_inference(...)`
  - output path validation

## Call Flow (Concrete)

```text
Caller
  -> qwen_tts.tts_api.TTSEngine()
  -> TTSEngine.tts_load(model_path, optional reference_audio_path, ...)
      -> Qwen3TTSModel.from_pretrained(model_path, ...)
      -> detect/store loaded model metadata (type/size)
      -> cache engine state
  -> TTSEngine.tts_inference(text, output_path, model-specific args, ...)
      -> dispatch by model type:
           base -> generate_voice_clone(...)
           custom_voice -> generate_custom_voice(...)
           voice_design -> generate_voice_design(...)
      -> soundfile.write(output_path, wavs[0], sample_rate)
      -> validate output file exists and is non-empty
      -> return Path(output_path)
```

## Scope Boundaries

- This architecture doc focuses on the callable wrapper API and its integration points.
- It does not attempt to fully document all Qwen3-TTS core model internals, tokenizer internals, or training code.
- For advanced internals, inspect:
  - `qwen_tts/core/`
  - `qwen_tts/inference/`
  - `examples/`
  - `finetuning/`

## Related files
- `qwen_tts/tts_api.py`
- `qwen_tts/inference/qwen3_tts_model.py`
- `qwen_tts/__init__.py`
- `test_tts_api.py`
- `README.md`
