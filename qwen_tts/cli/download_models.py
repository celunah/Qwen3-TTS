from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


DEFAULT_MODEL_REPOS: tuple[str, ...] = (
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the standard Qwen3-TTS Hugging Face model set into a local directory. "
            "Each repo is downloaded into a subfolder under the provided path."
        )
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Destination directory that will contain downloaded model subfolders.",
    )
    return parser


def _repo_local_dir(base_path: Path, repo_id: str) -> Path:
    # Use the repo name as the local subfolder (e.g. Qwen3-TTS-12Hz-1.7B-Base).
    return base_path / repo_id.split("/")[-1]


def download_qwen3_tts_models(destination_root: Path, repos: Iterable[str] = DEFAULT_MODEL_REPOS) -> list[Path]:
    destination_root = destination_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    downloaded_paths: list[Path] = []
    for repo_id in repos:
        local_dir = _repo_local_dir(destination_root, repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            resume_download=True,
        )
        downloaded_paths.append(local_dir)
    return downloaded_paths


def main() -> int:
    args = build_parser().parse_args()
    destination_root = Path(args.path)
    downloaded_paths = download_qwen3_tts_models(destination_root)

    print("\nDownloaded model directories:")
    for path in downloaded_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
