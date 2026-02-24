# Author Notes

## Purpose
Guide documentation updates for this repo so they describe the project as a simple, class-based, callable API wrapper for Qwen 3 TTS.

## Notes
- Document this project as an easy-to-call Python API for Qwen 3 TTS, not just a training/inference repo.
- Emphasize the class-based API design (`TTSEngine`) and the two-step workflow: load first, then generate.
- Keep docs focused on "black box" usage: callers provide model path, input/reference audio, text, and output audio path.
- Highlight that `tts_load(...)` should accept the model location explicitly (user chooses which model to load).
- Highlight that `tts_inference(...)` should accept an output audio path explicitly and return the generated file path.
- Prioritize practical usage examples over internal implementation details in README-style docs.
- When documenting parameters, make `model_path`, `reference_audio_path`, and `output_path` very clear and easy to find.
- Describe offline/local-path usage clearly when applicable.
- Keep wording readable and user-facing; avoid overloading docs with low-level Qwen internals unless needed for troubleshooting.
