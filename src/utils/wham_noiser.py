import librosa
import numpy as np
import os
import random
import soundfile as sf
import argparse


def mix_audio(input_audio_path, noise_audio_path, output_audio_path, sr=16_000):
    audio, _ = librosa.load(input_audio_path, sr=sr)
    noise, _ = librosa.load(noise_audio_path, sr=sr)
    if len(noise) < len(audio):
        repeat_times = len(audio) // len(noise) + 1
        noise = np.tile(noise, repeat_times)
    noise = noise[:len(audio)]
    mixed_audio = audio + noise
    sf.write(output_audio_path, mixed_audio, sr)


def main(input_dir, noise_dir, output_dir):
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                audio_files.append(full_path)
    noise_files = []
    for root, dirs, files in os.walk(noise_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                noise_files.append(os.path.join(root, file))
    os.makedirs(output_dir, exist_ok=True)
    for input_audio_path in audio_files:
        noise_audio_path = random.choice(noise_files)
        relative_path = os.path.relpath(input_audio_path, input_dir)
        output_audio_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        mix_audio(input_audio_path, noise_audio_path, output_audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для наложения шума на аудиофайлы.")
    parser.add_argument("--mix_dir", type=str, help="Директория с исходными аудиофайлами.")
    parser.add_argument("--noise_dir", type=str, help="Директория со шумовыми файлами.")
    parser.add_argument("--out_dir", type=str, help="Выходная директория для обработанных файлов.")

    args = parser.parse_args()

    main(args.mix_dir, args.noise_dir, args.out_dir)
