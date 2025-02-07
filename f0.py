import os
import json
import argparse
import time
import numpy as np
import librosa
import scipy.signal as signal

def time_function(func, *args, **kwargs):
    """Wrapper to time function execution."""
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    print(f"{func.__name__} took {elapsed_time:.4f} seconds")
    return result

def high_pass_filter(audio_signal, sample_rate, cutoff, order=5):
    """Apply a high-pass filter to the audio signal."""
    sos = signal.butter(order, cutoff, btype='high', fs=sample_rate, output='sos')
    return signal.sosfilt(sos, audio_signal, axis=-1)

def compute_f0(audio_signal, sample_rate, fmin, window_ms):
    """Compute F0 values per channel and report valid F0 percentages."""

    window_size = int((window_ms / 1000) * sample_rate)
    num_channels = audio_signal.shape[0]

    f0_values = []
    time_steps = []


    for ch in range(num_channels):
        start_time = time.time()
        total_windows = 0
        valid_f0_count = 0
        channel_f0, channel_time = [], []
        for start in range(0, len(audio_signal[ch]) - window_size, window_size):
            end = start + window_size
            window = audio_signal[ch, start:end]
            f0, _, _ = librosa.pyin(window, fmin=fmin, fmax=5000, sr=sample_rate)
            valid_f0 = f0[~np.isnan(f0)]
            total_windows += 1
            if len(valid_f0) > 0:
                valid_f0_count += 1
                channel_f0.append(np.mean(valid_f0))
            else:
                channel_f0.append(0.0)
            channel_time.append(start / sample_rate)

        f0_values.append(channel_f0)
        if not time_steps:  # Store time steps only once
            time_steps = channel_time

        valid_f0_percentage = (valid_f0_count / total_windows) * 100 if total_windows > 0 else 0
        elapsed_time = time.time() - start_time
        print(f"    channel {ch} took {elapsed_time:.4f} seconds, {valid_f0_percentage:.2f}% valid f0")

    return np.array(time_steps), np.array(f0_values)

def process_wav_file(filename, hp_filter, window_ms):
    """Load WAV, apply processing, and save F0 data to JSON."""
    print(f"Processing file: {filename}")
    print(f"High-pass filter frequency: {hp_filter} Hz")
    print(f"Window size: {window_ms} ms")

    audio_signal, sample_rate = time_function(librosa.load, filename, sr=None, mono=False)

    if audio_signal.ndim == 1:
        audio_signal = np.expand_dims(audio_signal, axis=0)  # Convert mono to shape (1, N)

    # High-pass filter
    filtered_audio = time_function(high_pass_filter, audio_signal, sample_rate, cutoff=hp_filter)

    # Compute F0 values per channel
    print(f"Computing f0_values:")
    f0_time_steps, f0_values = time_function(compute_f0, filtered_audio, sample_rate, hp_filter, window_ms)

    # Compute F0 values for the mono mix
    print(f"Computing f0_values_mono:")
    mono_signal = np.mean(filtered_audio, axis=0)
    _, f0_values_mono = time_function(compute_f0, np.expand_dims(mono_signal, axis=0), sample_rate, hp_filter, window_ms)

    # Save results to JSON
    base_name, _ = os.path.splitext(filename)
    output_filename = f"{base_name}_f0.json"

    data = {
        "high_pass_filter": hp_filter,
        "window_ms": window_ms,
        "sample_rate": sample_rate,
        "f0_time_steps": f0_time_steps.tolist(),
        "f0_values": {str(i): f0_values[i].tolist() for i in range(len(f0_values))},
        "f0_values_mono": f0_values_mono[0].tolist(),
    }

    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved F0 analysis to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract F0 values from a WAV file.")
    parser.add_argument("wav_file", type=str, help="Path to the input WAV file")
    parser.add_argument("--hp_filter", type=int, default=512, help="High-pass filter cutoff frequency (default: 512 Hz)")
    parser.add_argument("--window_ms", type=int, default=100, help="Window size in milliseconds (default: 100 ms)")
    args = parser.parse_args()

    process_wav_file(args.wav_file, args.hp_filter, args.window_ms)