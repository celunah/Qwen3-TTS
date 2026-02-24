# AI_CHANGELOG

## 2026-02-23
- Added `docs/AUTHOR_NOTES.md` to capture documentation constraints and preferences for the Qwen3-TTS callable wrapper API.
- Updated `README.md` with a new "Simple Class-Based API (Load -> Generate)" section for `qwen_tts.tts_api.TTSEngine`, including an offline/local-path example.
- Added `docs/API.md` documenting `TTSEngine`, including `tts_load(...)` and `tts_inference(...)` parameters, behavior, and error handling.
- Added `docs/ARCHITECTURE.md` describing the wrapper-focused call flow from `TTSEngine` into `Qwen3TTSModel.from_pretrained(...)` and `generate_voice_clone(...)`.
- Verified and reconciled docs after `TTSEngine` gained multi-model dispatch support (`base`, `custom_voice`, `voice_design`).
- Fixed doc mismatches where the wrapper was previously described as Base-only and where `reference_audio_path` was incorrectly documented as universally required.
- Updated `README.md`, `docs/API.md`, and `docs/ARCHITECTURE.md` to document model-specific inference arguments (`speaker`, `instruct`) and local 1.7B wrapper validation.
- Removed session-specific validation/provenance sections from `docs/API.md` and `docs/ARCHITECTURE.md`; kept that information in `AI_CHANGELOG.md` instead.
- Clarified in `docs/API.md` that `TTSEngine` CustomVoice support applies to both the provided published CustomVoice models and compatible CustomVoice model directories produced by user fine-tune runs.
