import pandas as pd
import streamlit as st
import os
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
from datetime import time, timedelta
import datetime
import re
import io
import zipfile


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

                if bird_name.startswith("Unknown"):
                    continue

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
                    above_cutoff = np.sum( (segment > self.cutoff[0]) & (segment <= self.cutoff[1]) ) / smoothing_samples
                    cage_noise = np.sum( (segment > 0) & (segment <= self.cutoff[0]) ) / smoothing_samples

                    bird_active_time[bird_name] += above_cutoff * segment_duration
                    bird_cage_noise_time[bird_name] += cage_noise * segment_duration

                    all_segment_percentages.append(above_cutoff)

                # Count segments for statistical analysis
                for i in range(0, len(time_steps) - smoothing_samples, smoothing_samples):
                    segment_values = np.concatenate([f0_values[ch][i : i + smoothing_samples] for ch in range(num_channels)])

                    if np.all( (segment_values == 0) | (segment_values > self.cutoff[1]) ):
                        zero_segments += 1
                    elif np.any( (segment_values > self.cutoff[0]) & (segment_values < self.cutoff[1]) ):
                        above_cutoff_segments += 1
                    else:
                        below_cutoff_segments += 1

        # 🟢 Create DataFrame with total Call time, cage noise time, and percentages
        bird_activity_data = []
        for bird in bird_total_time:
            total_time = bird_total_time[bird]
            Call_time = bird_active_time[bird]
            cage_noise_time = bird_cage_noise_time[bird]

            avg_activity = (Call_time / total_time) * 100 if total_time > 0 else 0
            avg_cage_noise = (cage_noise_time / total_time) * 100 if total_time > 0 else 0

            bird_activity_data.append({
                "Bird": bird,
                "Call %": avg_activity,  # Percentage of time spent singing
                "Cage Noise %": avg_cage_noise,  # Percentage of time in cage noise
                "Call": seconds_to_hms(Call_time),  # Absolute Call time
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
            if bars[1][i] < self.cutoff[0] :
                bar_color = 'red' 
            elif bars[1][i] <= self.cutoff[1] :
                bar_color = 'blue' 
            else :
                bar_color = 'grey'
            ax.patches[i].set_color(bar_color)
        
        ax.set_xlabel("F0 Values")
        ax.set_ylabel("Frequency Count")
        ax.grid(True)
        
        st.pyplot(fig)





    def plot_2d_heatmap(self, progress_bar=None, progress_text=None):
        df, sdf = self.compute_2d_histogram(progress_bar=progress_bar, progress_text=progress_text)
        
        x_values = df["Call Percentage"].values
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

        return fig, df , sdf



    def compute_2d_histogram(self, progress_bar=None, progress_text=None):
        clip_data    = []
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
                
                left_above  = np.sum( (left_segment  > cutoff[0]) & (left_segment  <= cutoff[1]) ) / segments
                right_above = np.sum( (right_segment > cutoff[0]) & (right_segment <= cutoff[1]) ) / segments
                
                segment_percentage = max(left_above, right_above)
                dominance_score    = abs(left_above - right_above) / (left_above + right_above + 1e-6)
                
                x_values.append(segment_percentage)
                y_values.append(dominance_score)
                timestamps.append(time_steps[i])

                segment_data.append({
                    "Room": room,
                    "time": time_steps[i],
                    "Call Percentage": segment_percentage * 100,
                    "Dominance Score": dominance_score * 100,
                    "filepath": video_file,
                    "f0" : filename[1]
                })
            
            # Ensure a segment starts at the beginning
            if len(timestamps) == 0 :
                continue

            if timestamps[0] != time_steps[0]:
                timestamps.insert(0, time_steps[0])
                x_values.insert(0, x_values[0])
                y_values.insert(0, y_values[0])
            
            # Apply PELT for 2D Change Point Detection
            activity_matrix = np.column_stack((x_values, y_values))
            try  :
                algo_2d = rpt.Pelt(model="rbf").fit(activity_matrix)
                change_points = algo_2d.predict(pen=cluster_penalty)

            
            except rpt.exceptions.BadSegmentationParameters:
                print(f"Segmentation failed: BadSegmentationParameters. Skipping segmentation for file {filename[1]} ")          
                # Fallback: Default to no change points or full sequence range
                change_points = [0, len(activity_matrix)]  # Default to single segment    
                
            # Ensure a segment covers the first and last portion of the file
            if change_points[0] != 0:
                change_points.insert(0, 0)

            if change_points[-1] < len(timestamps):
                change_points.append(len(timestamps))
            
            for j in range(len(change_points) - 1):
                start_time = timestamps[change_points[j]]
                end_time = timestamps[change_points[j + 1] - 1] if change_points[j + 1] < len(timestamps) else timestamps[-1]
                avg_clip_percentage = np.mean(x_values[change_points[j]:change_points[j + 1]])
                avg_dominance_score = np.mean(y_values[change_points[j]:change_points[j + 1]])
                
                clip_data.append({
                    "Room": room,
                    "Start": seconds_to_hms(start_time),
                    "Length": seconds_to_hms(end_time - start_time),
                    "Call Percentage": avg_clip_percentage * 100,
                    "Dominance Score": avg_dominance_score * 100,
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
        
        self.cdf = pd.DataFrame(clip_data)
        self.sdf = pd.DataFrame(segment_data)

        return self.cdf, self.sdf

    def plot_diagnostics(self, event, df):
        if event and event.selection.rows:

            selected_row_index = event.selection.rows[0]  
            filepath = df.iloc[selected_row_index]["filepath"]
            f0 = df.iloc[selected_row_index]["f0"]

            start_time = df.iloc[selected_row_index].get("start_time", 0)  #start_time in seconds
            end_time   = df.iloc[selected_row_index].get("end_time", 0)  #end_time in seconds

            # Load JSON from f0 file
            data = load_json(f0)
            time_steps = np.array(data["f0_time_steps"])
            f0_values = {int(k): np.array(v) for k, v in data["f0_values"].items()}
            
            # 🟢 Compute time differences for f0 > cutoff
            time_diffs = {ch: np.diff(time_steps[ (f0_values[ch] > self.cutoff[0]) & (f0_values[ch] <= self.cutoff[1]) ]) for ch in f0_values}


            # 🟢 Scatter Plot: Time differences vs time
            st.markdown("###### Scatter Plot of Time Differences Between Calls")
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
            for ch, diffs in time_diffs.items():
                scatter_times = time_steps[ (f0_values[ch] > self.cutoff[0]) & (f0_values[ch] <= self.cutoff[1]) ][1:]  # Shifted to match diff
                if len(diffs) > 0 and len(scatter_times) == len(diffs):
                    ax_scatter.scatter(scatter_times, diffs, alpha=0.5, label=f"{self.bird_name_lookup.get(filepath, {}).get(ch, ch)}")
            
            # 🔹 Add vertical grey lines for start_time and end_time
            ax_scatter.axvline(start_time, color="grey", linestyle="dashed", linewidth=1.5, label="Start Time")
            ax_scatter.axvline(end_time, color="grey", linestyle="dashed", linewidth=1.5, label="End Time")

            # Add labels for start and end time in hh:mm:ss
            ax_scatter.text(start_time, ax_scatter.get_ylim()[1] * 0.95, seconds_to_hms(start_time), 
                            rotation=90, color="black", verticalalignment="top", fontsize=9)
            ax_scatter.text(end_time, ax_scatter.get_ylim()[1] * 0.95, seconds_to_hms(end_time), 
                            rotation=90, color="black", verticalalignment="top", fontsize=9)

            # Formatting for time axis
            ax_scatter.set_xlabel("Time (hh:mm:ss)")
            min_time = int(time_steps.min())
            max_time = int(time_steps.max())
            tick_positions = np.arange(min_time, max_time, step=60)  # Every 60 seconds (1 min)
            tick_labels = [seconds_to_hms(t) for t in tick_positions]
            ax_scatter.set_xticks(tick_positions)
            ax_scatter.set_xticklabels(tick_labels, rotation=45, fontsize=9)
            
            ax_scatter.set_ylabel("Time between f0 events (seconds)")
            ax_scatter.legend()
            st.pyplot(fig_scatter)


            # 🟢 Generate separate normalized 2D histograms for each bird
            for ch, f0_vals in f0_values.items():
                bird_name = self.bird_name_lookup.get(filepath, {}).get(ch, f"Unknown_{ch}")
                valid_mask = (f0_vals > self.cutoff[0]) & (f0_vals <= self.cutoff[1])  # Ignore f0 values that are 0
                if np.sum(valid_mask) == 0:
                    continue

                st.markdown(f"###### Normalized 2D Histogram of f0 (Bird: {bird_name})")
                fig_2d, ax_2d = plt.subplots(figsize=(10, 4))
                heatmap, x_edges, y_edges = np.histogram2d(time_steps[valid_mask], f0_vals[valid_mask], bins=[50, 50])
                heatmap = heatmap / np.sum(heatmap)  # Normalize

                X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
                cbar = ax_2d.pcolormesh(X, Y, heatmap.T, cmap="Greys")
                ax_2d.set_xlabel("Time (hh:mm:ss)")

                # 🔹 Add vertical grey lines for start_time and end_time
                ax_2d.axvline(start_time, color="grey", linestyle="dashed", linewidth=1.5, label="Start Time")
                ax_2d.axvline(end_time, color="grey", linestyle="dashed", linewidth=1.5, label="End Time")

                # Add labels for start and end time in hh:mm:ss
                ax_2d.text(start_time, ax_2d.get_ylim()[1] * 0.95, seconds_to_hms(start_time), 
                        rotation=90, color="black", verticalalignment="top", fontsize=9)
                ax_2d.text(end_time, ax_2d.get_ylim()[1] * 0.95, seconds_to_hms(end_time), 
                        rotation=90, color="black", verticalalignment="top", fontsize=9)


                # Formatting x-axis with time labels
                min_time = int(time_steps.min())
                max_time = int(time_steps.max())
                tick_positions = np.arange(min_time, max_time, step=60)  # Every 60 seconds (1 min)
                tick_labels = [seconds_to_hms(t) for t in tick_positions]
                ax_2d.set_xticks(tick_positions)
                ax_2d.set_xticklabels(tick_labels, rotation=45, fontsize=9)

                ax_2d.set_ylabel("f0 Frequency (Hz)")

                # 🔥 Colorbar BELOW the plot
                fig_2d.colorbar(cbar, ax=ax_2d, orientation='horizontal', pad=0.2, label="Normalized Density")

                st.pyplot(fig_2d)

            col1, col2 = st.columns(2)  # Middle column is wider for better fit
            with col1:

                tbins = st.slider(
                    "FIlter important time differences",
                    min_value=0.0, max_value=20.0, value=(0.2, 2.5), step=.1,
                    help = "bird calls are within this range, cage noise is above and below this range"
                )

                # Filter differences < 2.5 seconds for each channel
                filtered_diffs = [diffs[ (diffs < tbins[1]) & (diffs > tbins[0]) ] for diffs in time_diffs.values() if len(diffs) > 0]
                if filtered_diffs:
                    all_filtered = np.concatenate(filtered_diffs)
                else:
                    all_filtered = np.array([])
                
                # Compute bin edges using 25 bins
                bins = np.histogram_bin_edges(all_filtered, bins=25)
                


                st.markdown(f"###### Normalized Histogram of Time Between Bird Calls (< {tbins[1]} sec and >{tbins[0]} sec)")
                fig_time_diff, ax_time_diff = plt.subplots(figsize=(10, 5))
                for ch, diffs in time_diffs.items():
                    if len(diffs) > 0:
                        # Filter to only include time differences less than 2.5 sec
                        diffs_filtered = diffs[ (diffs < tbins[1]) & (diffs > tbins[0]) ]
                        if len(diffs_filtered) == 0:
                            continue
                        weights = np.ones_like(diffs_filtered) / len(diffs_filtered)
                        ax_time_diff.hist(diffs_filtered, bins=bins, alpha=0.5, weights=weights,
                                            label=f"{self.bird_name_lookup.get(filepath, {}).get(ch, ch)}")
                ax_time_diff.set_xlabel("Time between detected f0 (seconds)")
                ax_time_diff.set_ylabel("Normalized Count")
                ax_time_diff.legend()
                st.pyplot(fig_time_diff)

            with col2:
            # 🟢 Normalized Histogram: f0 values above cutoff
                st.markdown(f"###### Normalized Histogram of f0 Values Above Cutoff {self.cutoff} hz")
                fig_f0_hist, ax_f0_hist = plt.subplots(figsize=(10, 5))
                for ch, f0_vals in f0_values.items():
                    f0_above_cutoff = f0_vals[ (f0_vals > self.cutoff[0]) & (f0_vals <= self.cutoff[1]) ]
                    if len(f0_above_cutoff) > 0:
                        weights = np.ones_like(f0_above_cutoff) / len(f0_above_cutoff)  # Normalize
                        ax_f0_hist.hist(f0_above_cutoff, bins=50, alpha=0.5, weights=weights, label=f"{self.bird_name_lookup.get(filepath, {}).get(ch, ch)}")
                ax_f0_hist.set_xlabel("f0 Frequency (Hz)")
                ax_f0_hist.set_ylabel("Normalized Count")
                ax_f0_hist.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)  # Horizontal legend
                st.pyplot(fig_f0_hist)





         
def plot_line_chart(event, filtered_df, cdf, sdf):
    if event and event.selection.rows:
        selected_row_index = event.selection.rows[0]  
        selected_room = filtered_df.iloc[selected_row_index]["Room"]

        # Filter data for the selected room
        cdf_filtered = cdf[cdf["Room"] == selected_room].copy()
        sdf_filtered = sdf[sdf["Room"] == selected_room].copy()

        fig, ax = plt.subplots(figsize=(14, 5))  # Larger figure for readability

        # Convert time to hh:mm:ss for x-axis labels
        sdf_filtered["time_hms"] = sdf_filtered["time"].apply(seconds_to_hms)
        cdf_filtered["start_hms"] = cdf_filtered["start_time"].apply(seconds_to_hms)

        # Scatter plot for segment-level data
        ax.scatter(sdf_filtered["time"], sdf_filtered["Call Percentage"], 
                   color="blue", label="Segment Call %", alpha=0.6, marker="o", s=10)
        ax.scatter(sdf_filtered["time"], sdf_filtered["Dominance Score"], 
                   color="red", label="Segment Dominance %", alpha=0.6, marker="o", s=10)

        # Connect segment dots with lines for better readability
        ax.plot(sdf_filtered["time"], sdf_filtered["Call Percentage"], 
                color="blue", alpha=0.4, linewidth=1)
        ax.plot(sdf_filtered["time"], sdf_filtered["Dominance Score"], 
                color="red", alpha=0.4, linewidth=1)

        # Line segments for clip-level data (discontinuous)
        for _, row in cdf_filtered.iterrows():
            ax.plot([row["start_time"], row["end_time"]], 
                    [row["Call Percentage"], row["Call Percentage"]],  
                    color="blue", linewidth=3, marker="|")      
            ax.plot([row["start_time"], row["end_time"]], 
                    [row["Dominance Score"], row["Dominance Score"]],  
                    color="red", linewidth=3, marker="|")

            # Add **more visible** vertical dashed grey lines for clip boundaries
            ax.axvline(row["start_time"], color="grey", linestyle="dashed", linewidth=1.2, alpha=0.8)

            # Place readable clip labels **above the x-axis** 
            ax.text(row["start_time"], 102, row["start_hms"], 
                    rotation=90, fontsize=8, color="black", ha="right", va="top", bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Formatting
        ax.set_xlabel("Time (hh:mm:ss)")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 105)  # Allow space for clip labels at 102%
        ax.legend(fontsize=10, loc="upper left")  # Adjust legend for better spacing
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)  # Faint grid lines

        # **Set x-ticks every minute**
        min_time = int(sdf_filtered["time"].min())
        max_time = int(sdf_filtered["time"].max())
        tick_positions = np.arange(min_time, max_time, step=60)  # Every 60 seconds (1 minute)
        tick_labels = [seconds_to_hms(t) for t in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=9)

        st.pyplot(fig)




def get_combined_files(f0_filepath):

    print(f0_filepath)
    """Get corresponding combined files for each channel in the F0 file.
    Returns a list of (participant, combined_file) tuples in the same order as the f0 channels."""
    directory = os.path.dirname(f0_filepath)
    base_name = os.path.basename(f0_filepath)
    # Remove the _f0.json suffix first
    name_without_suffix = base_name.rsplit('_f0.json', 1)[0]
    # Split all parts by dash
    parts = name_without_suffix.split('-')
    if len(parts) < 2:
        return []
        
    timestamp = parts[0]
    # All remaining parts are participants
    participants = parts[1:]
    
    # Return list of tuples to maintain order
    combined_files = []
    for participant in participants:
        combined_file = os.path.join(directory, f"{timestamp}-{participant}.combined.json")
        if os.path.exists(combined_file):
            combined_files.append((participant, combined_file))
            print(f"Found combined file: {combined_file}")  # Debug print
        else:
            print(f"Warning: Combined file not found: {combined_file}")
    
    return combined_files

def process_combined_files(combined_files, cutoff):
    """Process combined files to generate subtitle events."""
    events = []
    
    # First pass: collect all segments with their positions
    all_segments = []
    for i, (participant, filepath) in enumerate(combined_files):
        bird_position = 'left' if i == 0 else 'right'
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for segment in data.get('segments', []):
            segment_type = segment.get('type')
            if segment_type in ['song', 'call', 'cage_noise']:
                # For call and cage_noise, verify they're in the right frequency range
                if segment_type in ['call', 'cage_noise']:
                    f0_value = segment.get('f0', 0)
                    if cutoff[0] <= f0_value <= cutoff[1]:
                        segment_type = 'call'
                    else:
                        segment_type = 'cage_noise'
                
                all_segments.append({
                    'onset': segment['onset_ms'] / 1000.0,
                    'offset': segment['offset_ms'] / 1000.0,
                    'type': segment_type,
                    'position': bird_position
                })
    
    # Sort segments by onset time
    all_segments.sort(key=lambda x: x['onset'])
    
    # Second pass: create events considering overlaps
    current_state = {'left': None, 'right': None}
    last_event_time = None
    
    def create_event(time, state):
        active_events = []
        for pos in ['left', 'right']:
            if state[pos]:
                active_events.append(f"{pos} {state[pos]}")
        return {'time': time, 'type': " and ".join(active_events) if active_events else None}
    
    # Process all segment boundaries
    time_points = []
    for seg in all_segments:
        # Add both onset and offset with the segment information
        time_points.append((seg['onset'], 'start', seg))
        time_points.append((seg['offset'], 'end', seg))
    
    # Sort by time value only
    time_points.sort(key=lambda x: x[0])
    
    # Process time points to generate events
    for time, event_type, segment in time_points:
        # Update state based on event type
        if event_type == 'start':
            current_state[segment['position']] = segment['type']
        else:  # end
            current_state[segment['position']] = None
        
        # Create event with current state
        event = create_event(time, current_state)
        
        # Only add event if it's different from the last one or if it's the first event
        if not events or events[-1]['type'] != event['type']:
            events.append(event)
    
    return events

def generate_subtitles(data, f0_filepath=None, format="vtt", cutoff=(1700,3000), include_cage=True):
    """Generate subtitles from either combined.json files or f0 data."""
    
    # First try to find and use combined files if f0_filepath is provided
    combined_files = get_combined_files(f0_filepath) if f0_filepath else []

    print(f"F0 filepath: {f0_filepath}")
    print(f"Found combined files: {combined_files}")
    
    if combined_files:
        # Process combined files
        events = process_combined_files(combined_files, cutoff)
    else:
        # Fall back to original f0-based logic
        f0_time_steps = data["f0_time_steps"]
        f0_values = {int(k): v for k, v in data["f0_values"].items()}
        events = []
        last_state = {'left': None, 'right': None}
        
        for i in range(len(f0_time_steps)):
            current_time = f0_time_steps[i]
            left_bird = f0_values[0][i]
            right_bird = f0_values[1][i]
            current_state = {'left': None, 'right': None}
            
            # Process left bird
            if cutoff[0] <= left_bird <= cutoff[1]:
                current_state['left'] = 'call'
            elif left_bird > 0 and include_cage:
                current_state['left'] = 'cage_noise'
                
            # Process right bird
            if cutoff[0] <= right_bird <= cutoff[1]:
                current_state['right'] = 'call'
            elif right_bird > 0 and include_cage:
                current_state['right'] = 'cage_noise'
            
            # Only create a new event if the state has changed
            if current_state != last_state:
                event_text = []
                for pos in ['left', 'right']:
                    if current_state[pos]:
                        event_text.append(f"{pos} {current_state[pos]}")
                events.append({
                    'time': current_time,
                    'type': " and ".join(event_text) if event_text else None
                })
                last_state = current_state.copy()
    
    # Filter out cage noise events if not included
    if not include_cage:
        events = [event for event in events if event['type'] is None or 'cage_noise' not in event['type']]
    
    # Sort events by time
    events.sort(key=lambda x: x['time'])
    
    # Generate subtitles
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
    
    # Process events into subtitle entries
    for i, event in enumerate(events):
        time = event['time']
        text = event['type']
        
        # If text changes, end the previous subtitle and start a new one
        if text != prev_text:
            # End previous subtitle if it exists
            if prev_text is not None and start_time is not None:
                subtitle_entry = (
                    f"{len(subtitles) + 1}\n"
                    f"{format_time(start_time, srt=True)} --> {format_time(time, srt=True)}\n"
                    f"{prev_text}\n"
                ) if format == "srt" else (
                    f"{format_time(start_time)} --> {format_time(time)}\n"
                    f"{prev_text}\n"
                )
                subtitles.append(subtitle_entry)
            
            # Start new subtitle if we have text
            if text is not None:
                start_time = time
            else:
                start_time = None
            prev_text = text
    
    # Handle last subtitle if needed
    if prev_text is not None and start_time is not None:
        end_time = events[-1]['time']
        subtitle_entry = (
            f"{len(subtitles) + 1}\n"
            f"{format_time(start_time, srt=True)} --> {format_time(end_time, srt=True)}\n"
            f"{prev_text}\n"
        ) if format == "srt" else (
            f"{format_time(start_time)} --> {format_time(end_time)}\n"
            f"{prev_text}\n"
        )
        subtitles.append(subtitle_entry)
    
    if format == "srt":
        return "\n".join(subtitles)
    else:
        return "WEBVTT\n\n" + "\n".join(subtitles)


def play_video(event, df, cutoff, include_cage):
    if event and event.selection.rows:
        selected_row_index = event.selection.rows[0]  # Get the index of the selected row
        selected_room = df.iloc[selected_row_index]["Room"]
        video_file = df.iloc[selected_row_index]["filepath"]

        start_time = df.iloc[selected_row_index].get("start_time", 0)  # Get start_time if available

        if pd.notna(video_file) and os.path.exists(video_file):
            fname = os.path.basename(video_file)

            srt_output = None
            f0_filepath = df.iloc[selected_row_index]["f0"]

            if pd.notna(f0_filepath):
                with open(f0_filepath, "r") as f:
                    srt_output = generate_subtitles(json.load(f), f0_filepath=f0_filepath, format="srt", cutoff=cutoff, include_cage=include_cage)
                    # print(srt_output)

            # Play video
            st.markdown(f"###### Room {selected_room},Filename {fname} ")
            col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is wider for better fit

            with col2: 
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

                wav_file = os.path.splitext(video_file)[0] + ".wav"# Generate corresponding WAV filename
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


class AudioAnalysis:
    def __init__(self, event, df, cutoff, clip_length=20):

        selected_row_index = event.selection.rows[0]  
        filepath = df.iloc[selected_row_index]["filepath"]
        f0 = df.iloc[selected_row_index]["f0"]

        data = load_json(f0)

        # Fix wav file extension replacement
        wav_file = re.sub(r"\.mp4$", ".wav", filepath)

        # Load audio
        self.audio_signal, self.sample_rate = librosa.load(wav_file, sr=None, mono=False)

        # Ensure multi-channel format
        if self.audio_signal.ndim == 1:
            self.audio_signal = np.expand_dims(self.audio_signal, axis=0)  # Convert mono to (1, N)

        # Set attributes
        self.start_time     = 0
        self.num_channels   = self.audio_signal.shape[0]
        self.total_duration = len(self.audio_signal[0]) / self.sample_rate
        self.clip_length    = min(clip_length, self.total_duration)  # Fix incorrect calculation
        self.end_time       = self.total_duration

        # Extract data from JSON
        self.f0_time_steps = np.array(data["f0_time_steps"])
        self.f0_values = {int(k): np.array(v) for k, v in data.get("f0_values", {}).items()}
        self.f0_values_mono = np.array(data["f0_values_mono"])

        # Compute time vectors
        self.time_audio = np.linspace(0, self.total_duration, len(self.audio_signal[0]))

        # Fix missing f0_cutoff initialization
        self.cutoff = cutoff

    def audio_plot_interface(self, clip_start_time = 0):

        # Convert seconds to time object
        def seconds_to_time(seconds):
            return time(int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60))

        # Convert time object back to total seconds
        def time_to_seconds(time_obj):
            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
 
        formatted_time = st.slider(
            "Select time (hh:mm:ss)",
            min_value=seconds_to_time( int(self.start_time) ),
            max_value=seconds_to_time( int(max(self.start_time, self.end_time - self.clip_length)) ) ,
            value=seconds_to_time( int(clip_start_time) ),
            step=timedelta(seconds=1),
            format="HH:mm:ss"
        )
        st.write(f"Selected Time: {formatted_time}")

        start_time = time_to_seconds(formatted_time)


        # Ensure valid start and end time
        start_time = max(start_time, self.start_time)
        end_time   = min(start_time + self.clip_length, self.end_time)

        # Define time window for plotting
        start_sample = int(start_time * self.sample_rate)
        end_sample   = int(end_time * self.sample_rate)

        # Filter data strictly within start_time and end_time
        mask_audio = (self.time_audio    >= start_time) & (self.time_audio    <= end_time)
        mask_f0    = (self.f0_time_steps >= start_time) & (self.f0_time_steps <= end_time)
        time_audio = self.time_audio[mask_audio]  # Apply mask

        # Determine unified y-axis limits for all signal waveforms
        signal_min = np.min(self.audio_signal[:, mask_audio]) if self.num_channels > 1 else np.min(self.audio_signal[mask_audio])
        signal_max = np.max(self.audio_signal[:, mask_audio]) if self.num_channels > 1 else np.max(self.audio_signal[mask_audio])

        # Create plots: One waveform & spectrogram for each channel + one for mono
        fig, axes = plt.subplots(self.num_channels * 2 + 2, 1, figsize=(14, self.num_channels * 4), sharex=True, constrained_layout=True)

        for i in range(self.num_channels):
            # Plot waveform
            axes[i * 2].plot(time_audio, self.audio_signal[i][mask_audio], color="b", alpha=0.7)
            axes[i * 2].set_ylabel(f"Ch {i} Signal")
            axes[i * 2].set_ylim(signal_min, signal_max)  # Standardize y-axis for all signals
            axes[i * 2].grid(True, linestyle='--', alpha=0.5)
            axes[i * 2].set_xlim(start_time, end_time)  # Ensure x-axis limits match selection


            # Overlay detected F0 values as black vertical markers on waveforms (if above cutoff)
            for t, f0 in zip(self.f0_time_steps[mask_f0], self.f0_values.get(i, [])[mask_f0]):
                if f0 > self.cutoff[0] and f0 < self.cutoff[1]:
                    axes[i * 2].axvspan(t, t + 0.1, color="black", alpha=0.3)  # 100ms width

            # Compute spectrogram and set correct time coordinates
            S = librosa.feature.melspectrogram(y=self.audio_signal[i][start_sample:end_sample], sr=self.sample_rate, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=self.sample_rate, x_axis='time',
                                    x_coords=np.linspace(start_time, end_time, S.shape[1]),
                                    y_axis='mel', ax=axes[i * 2 + 1], cmap='magma')
            axes[i * 2 + 1].set_ylabel(f"Ch {i} Spec")
            axes[i * 2 + 1].set_xlim(start_time, end_time)

            # Overlay detected F0 values on spectrogram (if above cutoff)
            for t, f0 in zip(self.f0_time_steps[mask_f0], self.f0_values.get(i, [])[mask_f0]):
                if f0 > self.cutoff[0] and f0 < self.cutoff[1]:
                    axes[i * 2 + 1].axvspan(t, t + 0.1, color="white", alpha=0.3)
                    axes[i * 2 + 1].scatter(t, f0, color='cyan', s=20, edgecolors='black')  # Mark exact F0 location

        # Compute and plot mono waveform
        mono_signal = np.mean(self.audio_signal, axis=0)
        axes[-2].plot(time_audio, mono_signal[mask_audio], color="g", alpha=0.7)
        axes[-2].set_ylabel("Mono Signal")
        axes[-2].set_ylim(signal_min, signal_max)  # Standardize y-axis
        axes[-2].grid(True, linestyle='--', alpha=0.5)
        axes[-2].set_xlim(start_time, end_time)

        # Compute and plot mono spectrogram
        S_mono = librosa.feature.melspectrogram(y=mono_signal[start_sample:end_sample], sr=self.sample_rate, n_mels=128, fmax=8000)
        S_mono_dB = librosa.power_to_db(S_mono, ref=np.max)
        librosa.display.specshow(S_mono_dB, sr=self.sample_rate, x_axis='time',
                                x_coords=np.linspace(start_time, end_time, S_mono.shape[1]),
                                y_axis='mel', ax=axes[-1], cmap='magma')
        axes[-1].set_ylabel("Mono Spec")
        axes[-1].set_xlim(start_time, end_time)

        # Overlay detected F0 values on mono spectrogram (if above cutoff)
        for t, f0 in zip(self.f0_time_steps[mask_f0], self.f0_values_mono[mask_f0]):
            if f0 > self.cutoff[0] and f0 < self.cutoff[1]:
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


import io
import os
import zipfile

def create_zip(file_paths):
    """Create a zip file containing all JSON and WAV files from the directories of the provided files."""
    # Build a list of tuples: (file_path, last_subdir, basename, size)
    all_files = []
    processed_dirs = set()  # Keep track of processed directories to avoid duplicates
    
    for file_path in file_paths:
        directory = os.path.dirname(file_path)
        if directory in processed_dirs:
            continue
            
        processed_dirs.add(directory)
        last_subdir = os.path.basename(directory)
        
        # Find all JSON and WAV files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(('.json', '.wav')):
                full_path = os.path.join(directory, filename)
                try:
                    size = os.path.getsize(full_path)
                except Exception:
                    size = 0
                all_files.append((full_path, last_subdir, filename, size))

    # Create the zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add each file under "data/last_subdirectory/filename"
        for file_path, last_subdir, basename, _ in all_files:
            arcname = os.path.join("data", last_subdir, basename)
            try:
                zip_file.write(file_path, arcname=arcname)
            except Exception as e:
                print(f"Error adding {file_path} to zip: {e}")
        
        # Write manifest file if there are files to include
        if all_files:
            manifest_txt = "\n".join(os.path.join("data", f[1], f[2]) for f in all_files)
            zip_file.writestr("all_data.txt", manifest_txt)

    zip_buffer.seek(0)
    return zip_buffer

def collect_song_segments(df):
    """Collect all song segments from the selected files."""
    segments = []
    
    for _, row in df.iterrows():
        if pd.isna(row['f0']):
            continue
            
        # Load the F0 file
        f0_file = row['f0']
        video_file = row['filepath']
        room = row['Room']
        
        # Get corresponding song files
        directory = os.path.dirname(f0_file)
        base_name = os.path.basename(f0_file)
        parts = base_name.split('_')[0].split('-')
        timestamp = parts[0]
        participants = parts[1:]
        
        # Load song files for each participant
        for participant in participants:
            song_file = os.path.join(directory, f"{timestamp}-{participant}.wav_results.json")
            if not os.path.exists(song_file):
                continue
                
            try:
                with open(song_file, 'r') as f:
                    song_data = json.load(f)
                
                if song_data.get('song_present', False):
                    for segment in song_data.get('segments', []):
                        start_time = segment['onset_ms'] / 1000.0  # Convert to seconds
                        end_time = segment['offset_ms'] / 1000.0
                        duration = end_time - start_time
                        
                        segments.append({
                            'Room': room,
                            'Bird': participant,
                            'Start': seconds_to_hms(start_time),
                            'Duration': seconds_to_hms(duration),
                            'filepath': video_file,
                            'start_time': start_time,
                            'end_time': end_time,
                            'f0': f0_file
                        })
            except Exception as e:
                print(f"Error processing {song_file}: {e}")
    
    # Convert to DataFrame and sort by start time
    if segments:
        segments_df = pd.DataFrame(segments)
        segments_df = segments_df.sort_values(['Room', 'start_time'])
        return segments_df
    return None