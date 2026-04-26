# coding=utf-8
import argparse
import json
import re
from pathlib import Path

import librosa
from faster_whisper import WhisperModel

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

def normalize_speaker_name(speaker: str) -> str:
    spaced = speaker.replace(" ", "_").lower()
    cleaned = re.sub(r"[^a-z_]", "", spaced)
    collapsed = re.sub(r"_+", "_", cleaned)
    return collapsed.strip("_")

def find_speaker_dirs(input_dir: Path) -> list[Path]:
    speaker_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    if not speaker_dirs:
        raise ValueError(f"No speaker folders found in: {input_dir}")
    return sorted(speaker_dirs, key=lambda p: p.name.lower())

def find_audio_files(speaker_dir: Path) -> list[Path]:
    files = [p for p in speaker_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files, key=lambda p: p.name.lower())

def load_transcript_if_exists(audio_path: Path) -> str | None:
    txt_path = audio_path.with_suffix(".txt")
    if not txt_path.exists():
        return None
    text = txt_path.read_text(encoding="utf-8").strip()
    return text or None

def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
    language: str | None,
) -> str:
    segments, _ = model.transcribe(
        str(audio_path),
        language=None if language in (None, "", "Auto", "auto") else language,
        vad_filter=True,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text

def get_duration_seconds(audio_path: Path) -> float:
    duration = librosa.get_duration(path=str(audio_path))
    return float(duration)

def build_dataset(
    input_dir: Path,
    output_jsonl: Path,
    whisper_model_name: str,
    device: str,
    language_value: str,
    whisper_language: str | None,
    use_existing_txt: bool,
) -> None:
    speaker_dirs = find_speaker_dirs(input_dir)

    # Find existing transcripts or fall back to Whisper
    needs_whisper = True
    if use_existing_txt:
        needs_whisper = False
        for speaker_dir in speaker_dirs:
            for audio_path in find_audio_files(speaker_dir):
                if load_transcript_if_exists(audio_path) is None:
                    needs_whisper = True
                    break
            if needs_whisper:
                break

    # Initialize Whisper if there are missing transcripts
    model = None
    if needs_whisper:
        model = WhisperModel(
            whisper_model_name,
            device=device,
            compute_type="bfloat16",
        )

    rows: list[dict] = []
    stats: dict[str, dict[str, float | int]] = {}

    sample_count: int = 0
    dataset_length: float = 0.0

    # Populate the dataset
    for speaker_dir in speaker_dirs:
        speaker_name = speaker_dir.name
        audio_files = find_audio_files(speaker_dir)

        if not audio_files:
            print(f"[WARN] Speaker '{speaker_name}' has no audio files, skipping.")
            continue

        ref_audio = max(audio_files, key=get_duration_seconds)

        stats[speaker_name] = {
            "files": 0,
            "seconds": 0.0,
        }

        for audio_path in audio_files:
            text = load_transcript_if_exists(audio_path) if use_existing_txt else None

            # Load existing transcripts or transcribe with Whisper
            if text is None:
                if model is None:
                    raise RuntimeError("Whisper model was not initialized.")

                text = transcribe_audio(
                    model=model,
                    audio_path=audio_path,
                    language=whisper_language,
                )

            # Nothing transcribed, skip this row
            if not text:
                print(f"[WARN] Empty transcript for {audio_path}, skipping.")
                continue

            duration = get_duration_seconds(audio_path)

            rows.append(
                {
                    "audio": str(audio_path.resolve()),
                    "ref_audio": str(ref_audio.resolve()),
                    "text": text,
                    "speaker": normalize_speaker_name(speaker_name),
                    "language": language_value,
                }
            )

            stats[speaker_name]["files"] += 1
            stats[speaker_name]["seconds"] += duration

            sample_count += 1
            dataset_length += duration

            print(f"Speaker: {speaker_name}")
            print(text)
            print(f"Samples: {sample_count} ({dataset_length:.2f}s)")

    if not rows:
        raise ValueError("No dataset rows were created.")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved dataset to: {output_jsonl}")
    print(f"Total samples: {len(rows)}\n")
    print("Per-speaker stats:")
    for speaker_name, speaker_stats in sorted(stats.items()):
        files = int(speaker_stats["files"])
        seconds = float(speaker_stats["seconds"])
        minutes = seconds / 60.0
        print(f"  {speaker_name}: {files} files | {seconds:.2f}s | {minutes:.2f} min")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Location of input data")
    parser.add_argument(
        "--output_jsonl", type=str, default="train.jsonl", help="Output JSONL file name"
    )
    parser.add_argument(
        "--whisper_model", type=str, default="large-v3", help="Which Whisper model to use"
    )
    parser.add_argument("--device", type=str, default="cuda", help="CUDA device to use")
    parser.add_argument(
        "--language", type=str, default="Auto", help="The dataset's target language"
    )
    parser.add_argument(
        "--whisper_language",
        type=str,
        default=None,
        help="Language hint for Whisper, e.g. en, de, fr. Leave unset for auto-detection.",
    )
    parser.add_argument(
        "--no_use_existing_txt",
        action="store_true",
        help="Ignore .txt files next to audio and always transcribe with Whisper.",
    )
    args = parser.parse_args()

    build_dataset(
        input_dir=Path(args.input_dir),
        output_jsonl=Path(args.output_jsonl),
        whisper_model_name=args.whisper_model,
        device=args.device,
        language_value=args.language,
        whisper_language=args.whisper_language,
        use_existing_txt=not args.no_use_existing_txt,
    )


if __name__ == "__main__":
    main()
