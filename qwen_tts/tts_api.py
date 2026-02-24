from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import soundfile as sf
import torch

from .inference.qwen3_tts_model import Qwen3TTSModel


PathLike = Union[str, Path]
TextLike = Union[str, list[str]]


class TTSEngine:
    """Stable class-based TTS API for offline/local Qwen3-TTS inference."""

    def __init__(self) -> None:
        self._tts: Optional[Qwen3TTSModel] = None
        self._loaded_model_path: Optional[Path] = None
        self._reference_audio_path: Optional[Path] = None
        self._reference_text: Optional[str] = None
        self._device: Optional[str] = None
        self._dtype_name: Optional[str] = None
        self._attn_implementation: Optional[str] = None
        self._local_files_only: bool = True
        self._loaded_model_type: Optional[str] = None
        self._loaded_model_size: Optional[str] = None

    def tts_load(
        self,
        *,
        model_path: PathLike,
        reference_audio_path: Optional[PathLike] = None,
        reference_text: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        local_files_only: bool = True,
    ) -> None:
        model_dir = self._validate_local_model_path(model_path)
        resolved_device = self._resolve_device(device)
        normalized_dtype = self._normalize_dtype_name(dtype, resolved_device)
        ref_audio = (
            self._validate_local_audio_path(reference_audio_path)
            if reference_audio_path is not None
            else None
        )

        if (
            self._tts is not None
            and self._loaded_model_path == model_dir
            and self._device == resolved_device
            and self._dtype_name == normalized_dtype
            and self._attn_implementation == attn_implementation
            and self._local_files_only == bool(local_files_only)
        ):
            if ref_audio is not None:
                self._reference_audio_path = ref_audio
            if reference_text is not None:
                self._reference_text = reference_text
            if self._loaded_model_type == "base" and self._reference_audio_path is None:
                raise ValueError(
                    "Base model is loaded but no `reference_audio_path` is set. "
                    "Pass `reference_audio_path` to `tts_load(...)` or `tts_inference(...)`."
                )
            return

        dtype_obj = self._resolve_dtype(resolved_device, normalized_dtype)
        load_kwargs: dict[str, Any] = {
            "device_map": resolved_device,
            "dtype": dtype_obj,
            "local_files_only": bool(local_files_only),
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        try:
            self._tts = Qwen3TTSModel.from_pretrained(str(model_dir), **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Qwen3-TTS model from '{model_dir}' on device '{resolved_device}'."
            ) from exc

        model_meta = getattr(self._tts, "model", None)
        loaded_model_type = getattr(model_meta, "tts_model_type", None)
        loaded_model_size = getattr(model_meta, "tts_model_size", None)

        self._loaded_model_path = model_dir
        self._reference_audio_path = ref_audio
        self._reference_text = reference_text
        self._device = resolved_device
        self._dtype_name = normalized_dtype
        self._attn_implementation = attn_implementation
        self._local_files_only = bool(local_files_only)
        self._loaded_model_type = str(loaded_model_type) if loaded_model_type is not None else None
        self._loaded_model_size = str(loaded_model_size) if loaded_model_size is not None else None

        if self._loaded_model_type == "base" and self._reference_audio_path is None:
            raise ValueError(
                "Loaded Base model requires `reference_audio_path` for voice cloning. "
                "Pass a local reference audio file to `tts_load(...)`."
            )

    def tts_inference(
        self,
        *,
        text: TextLike,
        output_path: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        model_path: Optional[PathLike] = None,
        reference_audio_path: Optional[PathLike] = None,
        reference_text: Optional[str] = None,
        language: Union[str, list[str]] = "Auto",
        speaker: Optional[Union[str, list[str]]] = None,
        instruct: Optional[Union[str, list[str]]] = None,
        x_vector_only_mode: bool = True,
        seed: Optional[int] = 1234,
        **generate_kwargs: Any,
    ) -> Path:
        self._validate_text_input(text)

        if model_path is not None or reference_audio_path is not None or self._tts is None:
            if model_path is None and self._loaded_model_path is None:
                raise ValueError("`model_path` is required before first inference.")
            self.tts_load(
                model_path=model_path or self._loaded_model_path,
                reference_audio_path=reference_audio_path or self._reference_audio_path,
                reference_text=self._reference_text if reference_text is None else reference_text,
                device=self._device,
                dtype=self._dtype_name,
                attn_implementation=self._attn_implementation,
                local_files_only=self._local_files_only,
            )
        elif reference_text is not None:
            self._reference_text = reference_text
        if reference_audio_path is not None and self._tts is not None:
            self._reference_audio_path = self._validate_local_audio_path(reference_audio_path)

        if self._tts is None or self._loaded_model_path is None:
            raise RuntimeError("TTS engine is not loaded. Call `tts_load(...)` first.")

        out_path = self._resolve_output_path(output_path=output_path, output_dir=output_dir)

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        try:
            wavs, sample_rate = self._run_model_type_inference(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                x_vector_only_mode=x_vector_only_mode,
                generate_kwargs=generate_kwargs,
            )
        except Exception as exc:
            model_type = self._loaded_model_type or "unknown"
            raise RuntimeError(
                f"Qwen3-TTS synthesis failed for model type '{model_type}'."
            ) from exc

        if not wavs:
            raise RuntimeError("Qwen3-TTS returned no audio waveforms.")

        try:
            sf.write(str(out_path), wavs[0], int(sample_rate))
        except Exception as exc:
            raise RuntimeError(f"Failed to write synthesized audio to '{out_path}'.") from exc

        if not out_path.exists():
            raise FileNotFoundError(f"Synthesis output was not created: {out_path}")
        if out_path.stat().st_size <= 0:
            raise ValueError(f"Synthesis output is empty: {out_path}")
        return out_path

    def get_loaded_model_info(self) -> dict[str, Optional[str]]:
        """Return lightweight metadata for the currently loaded model."""
        return {
            "model_path": str(self._loaded_model_path) if self._loaded_model_path else None,
            "model_type": self._loaded_model_type,
            "model_size": self._loaded_model_size,
            "device": self._device,
            "dtype": self._dtype_name,
        }

    def _run_model_type_inference(
        self,
        *,
        text: TextLike,
        language: Union[str, list[str]],
        speaker: Optional[Union[str, list[str]]],
        instruct: Optional[Union[str, list[str]]],
        x_vector_only_mode: bool,
        generate_kwargs: dict[str, Any],
    ) -> tuple[list[Any], int]:
        if self._tts is None:
            raise RuntimeError("TTS engine is not loaded.")

        model_type = self._loaded_model_type
        if model_type == "base":
            if self._reference_audio_path is None:
                raise ValueError(
                    "Base model inference requires `reference_audio_path`. "
                    "Pass it to `tts_load(...)` or `tts_inference(...)`."
                )
            return self._tts.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=str(self._reference_audio_path),
                ref_text=self._reference_text,
                x_vector_only_mode=bool(x_vector_only_mode),
                **generate_kwargs,
            )

        if model_type == "custom_voice":
            if speaker is None or (isinstance(speaker, str) and not speaker.strip()):
                raise ValueError(
                    "CustomVoice model inference requires `speaker` "
                    "(for example: `speaker='Vivian'`)."
                )
            return self._tts.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                **generate_kwargs,
            )

        if model_type == "voice_design":
            if instruct is None or (isinstance(instruct, str) and not instruct.strip()):
                raise ValueError(
                    "VoiceDesign model inference requires `instruct` "
                    "(natural-language voice/style description)."
                )
            return self._tts.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                **generate_kwargs,
            )

        raise ValueError(
            f"Unsupported or unknown loaded model type '{model_type}'. "
            "Expected one of: base, custom_voice, voice_design."
        )

    def _validate_text_input(self, text: TextLike) -> None:
        if isinstance(text, str):
            if not text.strip():
                raise ValueError("`text` must be a non-empty string.")
            return
        if isinstance(text, list):
            if not text:
                raise ValueError("`text` list must not be empty.")
            bad_idx = [i for i, item in enumerate(text) if not isinstance(item, str) or not item.strip()]
            if bad_idx:
                raise ValueError(f"`text` contains empty or invalid items at indexes: {bad_idx}")
            return
        raise ValueError("`text` must be a string or list of strings.")

    def _validate_local_model_path(self, model_path: PathLike) -> Path:
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {path}. Offline mode requires a local model directory."
            )
        if not path.is_dir():
            raise ValueError(f"Model path must be a directory in offline mode: {path}")
        return path

    def _validate_local_audio_path(self, reference_audio_path: PathLike) -> Path:
        ref = str(reference_audio_path)
        if ref.startswith("http://") or ref.startswith("https://"):
            raise ValueError("Offline mode is enabled; `reference_audio_path` must be a local file path.")
        path = Path(reference_audio_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Reference audio path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Reference audio path must be a file: {path}")
        return path

    def _resolve_device(self, requested: Optional[str]) -> str:
        if requested:
            req = requested.strip().lower()
            if req == "cuda" and torch.cuda.is_available():
                return "cuda:0"
            if req.startswith("cuda") and not torch.cuda.is_available():
                raise ValueError("CUDA was requested but no CUDA device is available.")
            if req == "mps":
                mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
                if not mps_ok:
                    raise ValueError("MPS was requested but is not available.")
            return requested
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _normalize_dtype_name(self, dtype: Optional[str], device: str) -> str:
        if dtype is None:
            return "bfloat16" if str(device).startswith("cuda") else "float32"
        return dtype.lower()

    def _resolve_dtype(self, device: str, dtype: Optional[str]) -> torch.dtype:
        normalized = self._normalize_dtype_name(dtype, device)
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported dtype '{dtype}'. Use one of: {sorted(mapping)}")
        return mapping[normalized]

    def _resolve_output_path(
        self,
        *,
        output_path: Optional[PathLike],
        output_dir: Optional[PathLike],
    ) -> Path:
        if output_path is not None:
            out = Path(output_path).expanduser()
        else:
            base = Path(output_dir).expanduser() if output_dir is not None else Path("tts_outputs")
            out = base / "qwen3_tts_output.wav"
        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
