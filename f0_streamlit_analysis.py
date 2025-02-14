import pandas as pd
import streamlit as st
import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm
import ruptures as rpt
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

class F0Analysis:

    def __init__(self, df, cutoff, segment_length, cluster_penalty, progress_bar=None, progress_text=None):

        self.df              = df
        self.filenames       = df.loc[df["name_index"] == 0, ["filepath", "f0","Room"]].dropna().values.tolist()

        self.bird_name_lookup = {
            filepath: {name_index: bird for name_index, bird in zip(group["name_index"], group["Bird"])}
            for filepath, group in df.groupby("filepath")
        }

        self.cutoff          = cutoff
        self.segment_length  = segment_length
        self.cluster_penalty = cluster_penalty
        self.progress_bar = progress_bar  # Streamlit Progress Bar
        self.progress_text = progress_text  # Streamlit Progress Text
        self.compute_f0_statistics()


    def compute_f0_statistics(self):
        all_f0_values = []
        zero_segments = 0
        below_cutoff_segments = 0
        above_cutoff_segments = 0
        all_segment_percentages = []

        bird_total_time = {}       # Total time recorded for each bird
        bird_active_time = {}      # Time each bird has f0 > cutoff
        bird_cage_noise_time = {}  # Time each bird has 0 < f0 <= cutoff

        total_files = len(self.filenames)  # Total number of files for progress tracking

        for index, filename in enumerate(self.filenames):
            if self.progress_text:
                self.progress_text.text(f"Processing file {index + 1} of {total_files}: {os.path.basename(filename[0])}")

            if self.progress_bar:
                self.progress_bar.progress((index + 1) / total_files)

            data = load_json(filename[1])
            time_steps = np.array(data["f0_time_steps"])
            f0_values = {int(k): np.array(v) for k, v in data["f0_values"].items()}

            if len(f0_values) < 1:
                continue  # Ensure at least one channel exists

            num_channels = len(f0_values)
            smoothing_samples = int(self.segment_length / (time_steps[1] - time_steps[0]))

            # Initialize bird tracking per file
            filepath = filename[0]
            bird_mapping = self.bird_name_lookup.get(filepath, {})

            for channel, f0 in f0_values.items():
                bird_name = bird_mapping.get(channel, f"Unknown_{channel}")  # Get bird name or default

                # Ensure bird is in tracking dictionaries
                if bird_name not in bird_total_time:
                    bird_total_time[bird_name] = 0
                    bird_active_time[bird_name] = 0
                    bird_cage_noise_time[bird_name] = 0

                all_f0_values.extend(f0[f0 > data["high_pass_filter"]])  # Collect all non-zero f0 values

                # Process segments
                for i in range(0, len(time_steps) - smoothing_samples, smoothing_samples):
                    segment = f0[i : i + smoothing_samples]
                    segment_duration = smoothing_samples * (time_steps[1] - time_steps[0])

                    bird_total_time[bird_name] += segment_duration
                    above_cutoff = np.sum(segment > self.cutoff) / smoothing_samples
                    cage_noise = np.sum((segment > 0) & (segment <= self.cutoff)) / smoothing_samples

                    bird_active_time[bird_name] += above_cutoff * segment_duration
                    bird_cage_noise_time[bird_name] += cage_noise * segment_duration

                    all_segment_percentages.append(above_cutoff)

                # Count segments for statistical analysis
                for i in range(0, len(time_steps) - smoothing_samples, smoothing_samples):
                    segment_values = np.concatenate([f0_values[ch][i : i + smoothing_samples] for ch in range(num_channels)])

                    if np.all(segment_values == 0):
                        zero_segments += 1
                    elif np.any(segment_values > self.cutoff):
                        above_cutoff_segments += 1
                    else:
                        below_cutoff_segments += 1

        # 🟢 Create DataFrame with total song time, cage noise time, and percentages
        bird_activity_data = []
        for bird in bird_total_time:
            total_time = bird_total_time[bird]
            song_time = bird_active_time[bird]
            cage_noise_time = bird_cage_noise_time[bird]

            avg_activity = (song_time / total_time) * 100 if total_time > 0 else 0
            avg_cage_noise = (cage_noise_time / total_time) * 100 if total_time > 0 else 0

            bird_activity_data.append({
                "Bird": bird,
                "Song %": avg_activity,  # Percentage of time spent singing
                "Cage Noise %": avg_cage_noise,  # Percentage of time in cage noise
                "Song": seconds_to_hms(song_time),  # Absolute song time
                "Cage Noise": seconds_to_hms(cage_noise_time),  # Absolute cage noise time
                "Total Time": seconds_to_hms(total_time)  # Human-readable total time
            })

        # 🟢 Convert to DataFrame
        self.bdf = pd.DataFrame(bird_activity_data)

        # 🟢 Compute histogram
        # need to filter f0==0 values
        hist_counts, bins = np.histogram(all_segment_percentages, bins=10) if all_segment_percentages else (np.zeros(10), np.linspace(0, 1, 11))

        # 🟢 Store results
        self.zero_segments = zero_segments
        self.below_cutoff_segments = below_cutoff_segments
        self.above_cutoff_segments = above_cutoff_segments
        self.all_f0_values = np.array(all_f0_values)
        self.hist_counts = hist_counts
        self.bins = bins


        # 🟢 Final update of progress bar (marking completion)
        if self.progress_text:
            self.progress_text.text("")
        if self.progress_bar:
            self.progress_bar.progress(1.0)

    def plot_pie_chart(self):
        labels = ["No f0", "no bird sound", "some bird sound"]
        sizes = [self.zero_segments, self.below_cutoff_segments, self.above_cutoff_segments]
        colors = ["gray", "red", "blue"]
        
        filtered_sizes = [size for size in sizes if size > 0]
        filtered_labels = [labels[i] for i in range(len(sizes)) if sizes[i] > 0]
        filtered_colors = [colors[i] for i in range(len(sizes)) if sizes[i] > 0]
        
        if not filtered_sizes:
            st.warning("No valid data for pie chart.")
            return
        
        fig, ax = plt.subplots(figsize=(8,6))
        ax.pie(filtered_sizes, labels=filtered_labels, autopct="%1.1f%%", colors=filtered_colors, startangle=90, wedgeprops={"edgecolor": "black"}, textprops={'fontsize': 14   })
        
        st.pyplot(fig)

    def plot_stacked_bar_chart(self):
        if np.sum(self.hist_counts) == 0:
            st.warning("No valid F0 data available for histogram.")
            return
        
        categories = [f"{int(self.bins[i] * 100)}% - {int(self.bins[i+1] * 100)}%" for i in range(len(self.bins) - 1)]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(categories, self.hist_counts, color=plt.cm.tab10.colors[:len(categories)])
        
        ax.set_xlabel("Percentage of time bird sound identified")
        ax.set_ylabel("Number of segments")
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
        
        for bar, count in zip(bars, self.hist_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(count)), ha='center', va='bottom', fontsize=8)
        
        st.pyplot(fig)

    def plot_combined_histogram(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.hist(self.all_f0_values, bins=50, alpha=0.75, edgecolor='black')
        
        for i in range(len(bars[0])):
            bar_color = 'red' if bars[1][i] < self.cutoff else 'blue'
            ax.patches[i].set_color(bar_color)
        
        ax.set_xlabel("F0 Values")
        ax.set_ylabel("Frequency Count")
        ax.grid(True)
        
        st.pyplot(fig)





    def plot_2d_heatmap(self, progress_bar=None, progress_text=None):
        df = self.compute_2d_histogram(progress_bar=progress_bar, progress_text=progress_text)
        
        x_values = df["Song Percentage"].values
        y_values = df["Dominance Score"].values
        
        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = ax.hist2d(x_values, y_values, bins=[20, 20], cmap="plasma")
        
        # Auto-detect scaling method
        max_bin = np.max(heatmap[0])
        if max_bin > 100:
            norm = LogNorm()
        elif max_bin > 10:
            norm = PowerNorm(gamma=0.5)
        else:
            norm = None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = ax.hist2d(x_values, y_values, bins=[20, 20], cmap="plasma", norm=norm)
        
        ax.set_xlabel("Percentage of time bird sound identified in clip")
        ax.set_ylabel("Dominance Score (0 = balanced, 1 = one bird dominant)")
        
        cbar = plt.colorbar(heatmap[3], ax=ax)
        cbar.set_label("Number of Segments")

        return fig, df



    def compute_2d_histogram(self, progress_bar=None, progress_text=None):
        segment_data = []

        filenames = self.filenames
        cutoff = self.cutoff
        segment_length = self.segment_length
        cluster_penalty = self.cluster_penalty
        
        for file_number, filename in enumerate( filenames ):
            video_file = filename[0]
            data = load_json(filename[1])
            room = filename[2]
            time_steps = np.array(data["f0_time_steps"])
            f0_values = {int(k): np.array(v) for k, v in data["f0_values"].items()}
            
            if len(f0_values) < 2:
                continue
            
            left_f0 = f0_values[0]
            right_f0 = f0_values[1]
            
            segments = int(segment_length / (time_steps[1] - time_steps[0]))
            x_values, y_values, timestamps = [], [], []
            
            for i in range(0, len(time_steps) - segments, segments):
                left_segment = left_f0[i:i + segments]
                right_segment = right_f0[i:i + segments]
                
                left_above = np.sum(left_segment > cutoff) / segments
                right_above = np.sum(right_segment > cutoff) / segments
                
                segment_percentage = max(left_above, right_above)
                balance_score = abs(left_above - right_above) / (left_above + right_above + 1e-6)
                
                x_values.append(segment_percentage)
                y_values.append(balance_score)
                timestamps.append(time_steps[i])
            
            # Ensure a segment starts at the beginning
            if timestamps[0] != time_steps[0]:
                timestamps.insert(0, time_steps[0])
                x_values.insert(0, x_values[0])
                y_values.insert(0, y_values[0])
            
            # Apply PELT for 2D Change Point Detection
            activity_matrix = np.column_stack((x_values, y_values))
            algo_2d = rpt.Pelt(model="rbf").fit(activity_matrix)
            change_points = algo_2d.predict(pen=cluster_penalty)
            
            # Ensure a segment covers the first and last portion of the file
            if change_points[0] != 0:
                change_points.insert(0, 0)

            if change_points[-1] < len(timestamps):
                change_points.append(len(timestamps))
            
            for j in range(len(change_points) - 1):
                start_time = timestamps[change_points[j]]
                end_time = timestamps[change_points[j + 1] - 1] if change_points[j + 1] < len(timestamps) else timestamps[-1]
                avg_segment_percentage = np.mean(x_values[change_points[j]:change_points[j + 1]])
                avg_balance_score = np.mean(y_values[change_points[j]:change_points[j + 1]])
                
                segment_data.append({
                    "Room": room,
                    "Start": seconds_to_hms(start_time),
                    "Length": seconds_to_hms(end_time - start_time),
                    "Song Percentage": avg_segment_percentage * 100,
                    "Dominance Score": avg_balance_score * 100,
                    "filepath": video_file,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "f0" : filename[1]
                })
                if (progress_bar is not None) :
                    progress_bar.progress( file_number / len(filenames))
                if (progress_text is not None):
                    progress_text.text(f"Processing file {file_number + 1} of {len(filenames)}")

        return pd.DataFrame(segment_data)


def generate_subtitles(data, format="vtt", cutoff=1000, include_cage=True):
    f0_time_steps = data["f0_time_steps"]
    f0_values = data["f0_values"]

    f0_values = {int(k): v for k, v in f0_values.items()}

    subtitles = []
    prev_text = None
    start_time = None

    def format_time(seconds, srt=False):
        ms = int((seconds % 1) * 1000)
        total_seconds = int(seconds)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        if srt:
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        return f"{m:02}:{s:02}.{ms:03}"

    for i in range(len(f0_time_steps)):
        current_time = f0_time_steps[i]
        left_bird = f0_values[0][i]
        right_bird = f0_values[1][i]


        left_text = None
        right_text = None
        text = None

        if left_bird >= cutoff :
            left_text = "left bird"
        elif left_bird > 0  and include_cage :
            left_text = "left cage"

        if right_bird >= cutoff :
            right_text = "right bird"
        elif right_bird > 0 and include_cage :
            right_text = "right cage"

        if left_text and right_text:
            text = f"{left_text} and {right_text}"
        elif left_text:
            text = left_text    
        elif right_text:
            text = right_text

        if text != prev_text:
            if prev_text is not None:
                end_time = current_time
                if format == "srt":
                    subtitle_entry = (
                        f"{len(subtitles) + 1}\n"
                        f"{format_time(start_time, srt=True)} --> {format_time(end_time, srt=True)}\n"
                        f"{prev_text}\n"
                    )
                else:
                    subtitle_entry = (
                        f"{format_time(start_time)} --> {format_time(end_time)}\n"
                        f"{prev_text}\n"
                    )
                subtitles.append(subtitle_entry)
            prev_text = text
            start_time = current_time if text else None

    if prev_text is not None:
        end_time = f0_time_steps[-1]
        if format == "srt":
            subtitle_entry = (
                f"{len(subtitles) + 1}\n"
                f"{format_time(start_time, srt=True)} --> {format_time(end_time, srt=True)}\n"
                f"{prev_text}\n"
            )
        else:
            subtitle_entry = (
                f"{format_time(start_time)} --> {format_time(end_time)}\n"
                f"{prev_text}\n"
            )
        subtitles.append(subtitle_entry)

    if format == "srt":
        return "\n".join(subtitles)
    else:
        return "WEBVTT\n\n" + "\n".join(subtitles)


def play_video(event, df, cutoff, include_cage):
    if event.selection.rows:
        selected_row_index = event.selection.rows[0]  # Get the index of the selected row
        selected_room = df.iloc[selected_row_index]["Room"]

        video_file = df.iloc[selected_row_index]["filepath"]
        wav_file = video_file.replace(".mp4", ".wav")  # Generate corresponding WAV filename
        start_time = df.iloc[selected_row_index].get("start_time", 0)  # Get start_time if available

        if pd.notna(video_file) and os.path.exists(video_file):
            st.write(f"### Playing Video for {selected_room}")

            srt_output = None
            f0_filepath = df.iloc[selected_row_index]["f0"]

            if pd.notna(f0_filepath):
                with open(f0_filepath, "r") as f:
                    srt_output = generate_subtitles(json.load(f), format="srt", cutoff=cutoff, include_cage=include_cage)

            # Play video
            st.video(video_file, start_time=start_time, subtitles=srt_output)

            # 🟢 Download buttons for Video and WAV file
            with open(video_file, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="⬇️ Download Video",
                data=video_bytes,
                file_name=os.path.basename(video_file),
                mime="video/mp4"
            )

            if os.path.exists(wav_file):
                with open(wav_file, "rb") as f:
                    wav_bytes = f.read()
                st.download_button(
                    label="🎵⬇️ Download Audio (WAV)",
                    data=wav_bytes,
                    file_name=os.path.basename(wav_file),
                    mime="audio/wav"
                )
            else:
                st.warning("⚠️ WAV file not found for this video.")

        else:
            st.warning("⚠️ No video available for this selection.")
    else:
        st.warning("⚠️ Select a row to play a video.")

def plot_audio_and_f0( sdf_row , cutoff=1000):
    """Plot waveform and spectrogram with F0 markings for each channel and the mono mix."""
    
    json_file = sdf_row["f0"]
    wav_file = sdf_row["filepath"].replace("mp4", "wav")
    start_time = sdf_row["start_time"]
    clip_length = sdf_row["duration"]

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
            if f0 > cutoff:
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
            if f0 > cutoff:
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
        if f0 > cutoff:
            axes[-1].axvspan(t, t + 0.1, color="white", alpha=0.3)
            axes[-1].scatter(t, f0, color='cyan', s=20, edgecolors='black')  # Mark exact F0 location

    axes[-1].set_xlabel("Time (mm:ss)")

    def xaxis_format(x, _):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f"{minutes}:{seconds:02d}"

    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(xaxis_format))
    plt.xticks(rotation=45)  # Rotate tick labels for readability

    st.pyplot(fig)



#plot_audio_and_f0(args.wav_file, args.start_time, args.clip_length, args.f0_cutoff)

