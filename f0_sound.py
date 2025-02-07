import os
import json
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageDraw
import subprocess
import sys
import matplotlib.animation as animation

def parse_time(time_str):
    """Convert mm:ss format to seconds."""
    if time_str is None:
        return None
    try:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid time format: {time_str}. Use mm:ss.")

def load_json(json_file):
    """Load JSON file containing f0 analysis."""
    with open(json_file, "r") as f:
        return json.load(f)

def format_time(x, _):
    """Format x-axis ticks as mm:ss."""
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f"{minutes}:{seconds:02d}"

def plot_audio_and_f0(wav_file, start_time=None, clip_length=None, f0_cutoff=1000):
    """Plot waveform and spectrogram with F0 markings for each channel and the mono mix."""
    
    # Load corresponding JSON file
    json_file = os.path.splitext(wav_file)[0] + "_f0.json"
    if not os.path.exists(json_file):
        print(f"Error: JSON file {json_file} not found. Run the F0 analysis first.")
        return

    data = load_json(json_file)
    
    # Load WAV file
    audio_signal, sample_rate = librosa.load(wav_file, sr=None, mono=False)
    
    # Ensure multi-channel format
    if audio_signal.ndim == 1:
        audio_signal = np.expand_dims(audio_signal, axis=0)  # Convert mono to shape (1, N)
    
    num_channels = audio_signal.shape[0]
    total_duration = len(audio_signal[0]) / sample_rate

    # Default values for start_time and clip_length
    if start_time is None:
        start_time = 0
    if clip_length is None:
        clip_length = total_duration

    # Ensure start_time + clip_length does not exceed audio duration
    end_time = min(start_time + clip_length, total_duration)

    # Extract data from JSON
    f0_time_steps = np.array(data["f0_time_steps"])
    f0_values = {int(k): np.array(v) for k, v in data.get("f0_values", {}).items()}
    f0_values_mono = np.array(data["f0_values_mono"])

    # Compute time vectors
    time_audio = np.linspace(0, total_duration, len(audio_signal[0]))

    # Define time window for plotting
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Filter data strictly within start_time and end_time
    mask_audio = (time_audio >= start_time) & (time_audio <= end_time)
    mask_f0 = (f0_time_steps >= start_time) & (f0_time_steps <= end_time)
    time_audio = time_audio[mask_audio]  # Apply mask

    # Determine unified y-axis limits for all signal waveforms
    signal_min = np.min(audio_signal[:, mask_audio]) if num_channels > 1 else np.min(audio_signal[mask_audio])
    signal_max = np.max(audio_signal[:, mask_audio]) if num_channels > 1 else np.max(audio_signal[mask_audio])

    # Create plots: One waveform & spectrogram for each channel + one for mono
    fig, axes = plt.subplots(num_channels * 2 + 2, 1, figsize=(14, num_channels * 4), sharex=True, constrained_layout=True)

    for i in range(num_channels):
        # Plot waveform
        axes[i * 2].plot(time_audio, audio_signal[i][mask_audio], color="b", alpha=0.7)
        axes[i * 2].set_ylabel(f"Ch {i} Signal")
        axes[i * 2].set_ylim(signal_min, signal_max)  # Standardize y-axis for all signals
        axes[i * 2].grid(True, linestyle='--', alpha=0.5)
        axes[i * 2].set_xlim(start_time, end_time)  # Ensure x-axis limits match selection

        # Overlay detected F0 values as black vertical markers on waveforms (if above cutoff)
        for t, f0 in zip(f0_time_steps[mask_f0], f0_values.get(i, [])[mask_f0]):
            if f0 > f0_cutoff:
                axes[i * 2].axvspan(t, t + 0.1, color="black", alpha=0.3)  # 100ms width

        # Compute spectrogram and set correct time coordinates
        S = librosa.feature.melspectrogram(y=audio_signal[i][start_sample:end_sample], sr=sample_rate, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time',
                                 x_coords=np.linspace(start_time, end_time, S.shape[1]),
                                 y_axis='mel', ax=axes[i * 2 + 1], cmap='magma')
        axes[i * 2 + 1].set_ylabel(f"Ch {i} Spec")
        axes[i * 2 + 1].set_xlim(start_time, end_time)

        # Overlay detected F0 values on spectrogram (if above cutoff)
        for t, f0 in zip(f0_time_steps[mask_f0], f0_values.get(i, [])[mask_f0]):
            if f0 > f0_cutoff:
                axes[i * 2 + 1].axvspan(t, t + 0.1, color="white", alpha=0.3)
                axes[i * 2 + 1].scatter(t, f0, color='cyan', s=20, edgecolors='black')  # Mark exact F0 location

    # Compute and plot mono waveform
    mono_signal = np.mean(audio_signal, axis=0)
    axes[-2].plot(time_audio, mono_signal[mask_audio], color="g", alpha=0.7)
    axes[-2].set_ylabel("Mono Signal")
    axes[-2].set_ylim(signal_min, signal_max)  # Standardize y-axis
    axes[-2].grid(True, linestyle='--', alpha=0.5)
    axes[-2].set_xlim(start_time, end_time)

    # Compute and plot mono spectrogram
    S_mono = librosa.feature.melspectrogram(y=mono_signal[start_sample:end_sample], sr=sample_rate, n_mels=128, fmax=8000)
    S_mono_dB = librosa.power_to_db(S_mono, ref=np.max)
    librosa.display.specshow(S_mono_dB, sr=sample_rate, x_axis='time',
                             x_coords=np.linspace(start_time, end_time, S_mono.shape[1]),
                             y_axis='mel', ax=axes[-1], cmap='magma')
    axes[-1].set_ylabel("Mono Spec")
    axes[-1].set_xlim(start_time, end_time)

    # Overlay detected F0 values on mono spectrogram (if above cutoff)
    for t, f0 in zip(f0_time_steps[mask_f0], f0_values_mono[mask_f0]):
        if f0 > f0_cutoff:
            axes[-1].axvspan(t, t + 0.1, color="white", alpha=0.3)
            axes[-1].scatter(t, f0, color='cyan', s=20, edgecolors='black')  # Mark exact F0 location

    axes[-1].set_xlabel("Time (mm:ss)")
    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    plt.xticks(rotation=45)  # Rotate tick labels for readability
    
    # plt.subplots_adjust(left=0.05, right=0.05, top=0.05, bottom=0.05)
    temp_png   = "temp.png"
    plt.savefig(temp_png, bbox_inches='tight', dpi=100)
    plt.close(fig)


    img = Image.open(temp_png)
    width, height = img.size


    # Convert start_time and end_time to pixel coordinates
    start_x_pixel = axes[-1].transData.transform((start_time, 0))[0]
    end_x_pixel = axes[-1].transData.transform((end_time, 0))[0]

    # Get tick padding size in points and convert to pixels
    tick_padding_pts = axes[-1].xaxis.get_tick_padding()  # Padding size in points
    tick_padding_px = (tick_padding_pts / 72) * fig.dpi  # Convert to pixels

    # Adjust the pixel positions to account for tick mark width
    start_x_pixel += tick_padding_px
    end_x_pixel += tick_padding_px


    temp_video = "temp_video.mp4"
    temp_audio = "temp_audio.wav"
    output_video = os.path.splitext(wav_file)[0] + "_f0.mp4"
    fps = 30    
    
    subprocess.run(
        f'ffmpeg -y -i "{wav_file}" -ss {start_time} -t {clip_length} "{temp_audio}"',
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
    )


    # Precompute frame timestamps
    duration_frames = fps * int(clip_length)
    frame_times = np.linspace(start_time, start_time + clip_length, duration_frames)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(img)
    ax.axis('off')

    red_line = ax.axvline(0, color='r', linewidth=2)

    def update(frame):
        current_time = frame_times[frame]
        x_position = int(start_x_pixel + ((current_time - start_time) / clip_length) * (end_x_pixel - start_x_pixel))
        red_line.set_xdata([x_position, x_position])
        return red_line,

    ani = animation.FuncAnimation(fig, update, frames=duration_frames, blit=True)
    FFMpegWriter = animation.FFMpegWriter(fps=fps, metadata=dict(title="Audio Analysis"))
    ani.save(temp_video, writer=FFMpegWriter, dpi=100)

    subprocess.run(
        f'ffmpeg -y -i "{temp_audio}" -i "{temp_video}" -c:v libx264 -c:a aac -strict experimental "{output_video}"',
        shell=True, stdout=sys.stdout, stderr=sys.stderr
    )

    subprocess.run(
        f'ffmpeg -y -i "{temp_audio}" -i "{temp_video}" -c:v libx264 -c:a aac -strict experimental "{output_video}"',
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    # os.remove(temp_video)
    # os.remove(temp_audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot F0 values for each channel and the mono mix.")
    parser.add_argument("wav_file", type=str, help="Path to the input WAV file")
    parser.add_argument("-st", "--start_time", type=parse_time, default=None, help="Start time in mm:ss format (default: start of file)")
    parser.add_argument("-t", "--clip_length", type=parse_time, default=None, help="Duration of the clip in mm:ss format (default: full file length)")
    parser.add_argument("-c","--f0_cutoff", type=int, default=1000, help="Minimum F0 frequency to display (default: 1000 Hz)")
    
    args = parser.parse_args()
    plot_audio_and_f0(args.wav_file, args.start_time, args.clip_length, args.f0_cutoff)