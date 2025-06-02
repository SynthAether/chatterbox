#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python inference_chatter.py   --txt_dir  ./text_files/   --output_dir ./outputs  --gain_db -3   --audio_prompt  ./prompts/test.wav --ckpt_dir ./logdir

try:
    import sox
except ImportError:
    sox = None  # We'll check later if user asked for gain but sox isn't installed

def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio for each .txt in a folder using ChatterboxTTS."
    )
    parser.add_argument(
        "--txt_dir",
        type=str,
        required=True,
        help="Directory containing .txt files, each with one utterance per file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where synthesized .wav files will be saved.",
    )
    parser.add_argument(
        "--audio_prompt",
        type=str,
        default=None,
        help="(Optional) Path to a .wav file to use as the voice prompt for cloning.",
    )
    parser.add_argument(
        "--gain_db",
        type=float,
        default=None,
        help="(Optional) Gain normalization target in dB (e.g. -3.0). Requires python-sox.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="(Optional) Path to directory with local checkpoint files for ChatterboxTTS.",
    )
    
    args = parser.parse_args()

    # 1. Validate txt_dir and output_dir
    txt_dir = Path(args.txt_dir)
    if not txt_dir.is_dir():
        raise ValueError(f"{txt_dir} is not a valid directory")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. If user requested gain normalization but sox isn't importable, abort early
    if args.gain_db is not None and sox is None:
        raise ImportError(
            "You requested --gain_db, but python-sox is not installed. "
            "Please run 'pip install sox' and try again."
        )

    # 3. Load the pretrained ChatterboxTTS model
    #    If you want CPU only, pass --device cpu. Otherwise, default is cuda if available.
    print(f"Loading ChatterboxTTS model on device '{args.device}'...")
    #model = ChatterboxTTS.from_pretrained(device=args.device)
    if args.ckpt_dir:
        model = ChatterboxTTS.from_local(args.ckpt_dir, device=args.device)
    else:
        model = ChatterboxTTS.from_pretrained(device=args.device)    

    # 4. Iterate over every .txt file in txt_dir
    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {txt_dir}. Exiting.")
        return

    for txt_path in txt_files:
        # Read the entire text content (strip leading/trailing whitespace)
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            print(f"Skipping empty file: {txt_path.name}")
            continue

        # 5. Generate waveform. If an audio_prompt is provided, include it.
        if args.audio_prompt:
            wav = model.generate(text, audio_prompt_path=args.audio_prompt)
        else:
            wav = model.generate(text)

        # 6. Save the raw waveform to output_dir/{stem}.wav
        out_wav_path = output_dir / f"{txt_path.stem}.wav"
        # torchaudio.save expects a Tensor of shape [channels, time], but
        # ChatterboxTTS.generate(...) typically returns a 1D Tensor (time,).
        # torchaudio will interpret 1D as a single channel.
        ta.save(str(out_wav_path), wav, model.sr)

        # 7. If gain_db is specified, apply sox normalization and overwrite
        if args.gain_db is not None:
            # Build a temporary filename in the same folder
            temp_norm_path = output_dir / f"{txt_path.stem}.norm.wav"
            transformer = sox.Transformer()
            transformer.norm(args.gain_db)
            # Run the sox effect: reads out_wav_path, writes temp_norm_path
            transformer.build(str(out_wav_path), str(temp_norm_path))
            # Replace the original file with the normalized one
            temp_norm_path.replace(out_wav_path)

        print(f"[Synthesized] {txt_path.name} -> {out_wav_path.name}")

if __name__ == "__main__":
    main()
