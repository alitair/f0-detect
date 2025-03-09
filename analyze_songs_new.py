import os
import json
from datetime import datetime
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_song_files(f0_filepath):
    """Get corresponding song files for each channel in the F0 file."""
    # Get the directory and base filename
    directory = os.path.dirname(f0_filepath)
    base_name = os.path.basename(f0_filepath)
    
    # Extract timestamp and participants from the filename
    # Example: 1725054582-USA5499-USA5497_f0.json -> timestamp = 1725054582, participants = [USA5499, USA5497]
    parts = base_name.split('_')[0].split('-')
    timestamp = parts[0]
    participants = parts[1:]
    
    song_files = {}
    for participant in participants:
        song_file = os.path.join(directory, f"{timestamp}-{participant}.wav_results.json")
        if os.path.exists(song_file):
            song_files[participant] = song_file
        else:
            print(f"Warning: Song file not found: {song_file}")
    
    if not song_files:
        print(f"Warning: No song files found for {f0_filepath}")
    
    return song_files

def is_song_at_time(song_data, time_sec):
    """Check if there is song at the given time."""
    if not song_data.get('song_present', False):
        return False
    
    time_ms = time_sec * 1000
    has_song = any(seg['onset_ms'] <= time_ms <= seg['offset_ms'] 
                  for seg in song_data.get('segments', []))
    
    return has_song

def classify_f0(f0_value):
    """Classify F0 value as call, cage noise, or none."""
    if f0_value == 0:
        return 'none'
    elif 1700 < f0_value < 3000:
        return 'call'
    else:
        return 'cage_noise'

def analyze_file(f0_filepath):
    """Analyze a single F0 file and its corresponding song files."""
    try:
        # Load F0 data
        f0_data = load_json(f0_filepath)
        
        # Get and load song files
        song_files = get_song_files(f0_filepath)
        if not song_files:
            print(f"Skipping {f0_filepath} - no song files found")
            return None
            
        song_data = {participant: load_json(filepath) 
                    for participant, filepath in song_files.items()}
        
        # Initialize segment analysis for each channel
        segment_stats = {}
        for participant in song_files.keys():
            segment_stats[participant] = []
            
            if song_data[participant].get('song_present', False):
                for segment in song_data[participant].get('segments', []):
                    # Convert ms to seconds
                    start_time = segment['onset_ms'] / 1000.0
                    end_time = segment['offset_ms'] / 1000.0
                    duration = end_time - start_time
                    
                    # Initialize counters for this segment
                    segment_info = {
                        'duration': duration,
                        'calls': 0,
                        'cage_noise': 0,
                        'total_points': 0
                    }
                    
                    # Find all F0 points within this segment
                    time_steps = f0_data['f0_time_steps']
                    f0_values = f0_data['f0_values']
                    participant_idx = list(song_files.keys()).index(participant)
                    
                    for i, time_step in enumerate(time_steps):
                        if start_time <= time_step <= end_time:
                            f0_value = float(f0_values[str(participant_idx)][i])
                            f0_class = classify_f0(f0_value)
                            
                            segment_info['total_points'] += 1
                            if f0_class == 'call':
                                segment_info['calls'] += 1
                            elif f0_class == 'cage_noise':
                                segment_info['cage_noise'] += 1
                    
                    # Calculate percentages
                    if segment_info['total_points'] > 0:
                        segment_info['call_percentage'] = (segment_info['calls'] / segment_info['total_points']) * 100
                        segment_info['cage_noise_percentage'] = (segment_info['cage_noise'] / segment_info['total_points']) * 100
                        segment_stats[participant].append(segment_info)
        
        return {
            'filename': os.path.basename(f0_filepath),
            'segment_stats': segment_stats
        }
        
    except Exception as e:
        print(f"Error processing {f0_filepath}: {e}")
        return None

def main():
    # Load list of F0 files from all_data.txt
    with open('all_data.txt', 'r') as f:
        f0_files = [line.strip() for line in f]
    
    # Process each file
    results = []
    for f0_file in f0_files:
        result = analyze_file(f0_file)
        if result:
            results.append(result)
    
    if not results:
        print("No valid files found for analysis.")
        return
    
    # Collect all segment statistics
    all_segments = []
    for result in results:
        for participant, segments in result['segment_stats'].items():
            all_segments.extend(segments)
    
    if not all_segments:
        print("No segments found for analysis.")
        return
    
    # Create duration bins (in seconds)
    duration_bins = [0, 0.5, 1, 2, 3, 4, 5, float('inf')]
    bin_labels = ['0-0.5s', '0.5-1s', '1-2s', '2-3s', '3-4s', '4-5s', '5s+']
    
    # Group segments by duration
    binned_segments = {label: [] for label in bin_labels}
    for segment in all_segments:
        duration = segment['duration']
        for i, (lower, upper) in enumerate(zip(duration_bins[:-1], duration_bins[1:])):
            if lower <= duration < upper:
                binned_segments[bin_labels[i]].append(segment)
                break
    
    # Calculate statistics for each bin
    call_means = []
    call_stds = []
    noise_means = []
    noise_stds = []
    counts = []
    
    for label in bin_labels:
        segments = binned_segments[label]
        if segments:
            call_percentages = [s['call_percentage'] for s in segments]
            noise_percentages = [s['cage_noise_percentage'] for s in segments]
            
            call_means.append(np.mean(call_percentages))
            call_stds.append(np.std(call_percentages))
            noise_means.append(np.mean(noise_percentages))
            noise_stds.append(np.std(noise_percentages))
            counts.append(len(segments))
        else:
            call_means.append(0)
            call_stds.append(0)
            noise_means.append(0)
            noise_stds.append(0)
            counts.append(0)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    x = np.arange(len(bin_labels))
    width = 0.35
    
    # Plot mean percentages
    rects1 = ax1.bar(x - width/2, call_means, width, label='Calls', color='#1f77b4')
    rects2 = ax1.bar(x + width/2, noise_means, width, label='Cage Noise', color='#2ca02c')
    
    # Add error bars
    ax1.errorbar(x - width/2, call_means, yerr=call_stds, fmt='none', color='black', capsize=5)
    ax1.errorbar(x + width/2, noise_means, yerr=noise_stds, fmt='none', color='black', capsize=5)
    
    ax1.set_ylabel('Percentage of Segment')
    ax1.set_title('Average Composition of Song Segments by Duration')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot segment counts
    ax2.bar(x, counts, width, color='#ff7f0e')
    ax2.set_ylabel('Number of Segments')
    ax2.set_xlabel('Segment Duration')
    ax2.set_title('Distribution of Segment Durations')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels)
    ax2.grid(True, alpha=0.3)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        ax2.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('segment_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total segments analyzed: {len(all_segments)}")
    print("\nBreakdown by duration:")
    for i, label in enumerate(bin_labels):
        if counts[i] > 0:
            print(f"\n{label} (n={counts[i]}):")
            print(f"  Calls:      {call_means[i]:.1f}% ± {call_stds[i]:.1f}%")
            print(f"  Cage Noise: {noise_means[i]:.1f}% ± {noise_stds[i]:.1f}%")

if __name__ == "__main__":
    main() 