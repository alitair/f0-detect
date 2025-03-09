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
        
        # Print debug info about song files
        for participant, data in song_data.items():
            print(f"\nSong file for {participant}:")
            print(f"  Song present: {data.get('song_present', False)}")
            print(f"  Number of segments: {len(data.get('segments', []))}")
            if data.get('segments'):
                print(f"  First segment: {data['segments'][0]}")
        
        # Initialize counters for each channel
        stats = {}
        participants = list(song_files.keys())  # Convert to list to avoid dictionary size changes
        for participant in participants:
            stats[participant] = {
                'song_with_calls': 0,
                'song_with_cage_noise': 0,
                'song_only': 0,
                'calls_only': 0,
                'cage_noise_only': 0,
                'no_sound': 0,
                'total_points': 0
            }
        
        # Print debug info about F0 data
        print(f"\nF0 data:")
        print(f"  Number of time steps: {len(f0_data['f0_time_steps'])}")
        print(f"  First time step: {f0_data['f0_time_steps'][0]}")
        print(f"  First F0 values: ", end="")
        for participant_idx in range(len(participants)):
            print(f"{float(f0_data['f0_values'][str(participant_idx)][0]):.2f} ", end="")
        print()
        
        # Analyze each time point
        time_steps = f0_data['f0_time_steps']
        f0_values = f0_data['f0_values']
        
        for i, time_step in enumerate(time_steps):
            for participant_idx, participant in enumerate(participants):
                # Get F0 value for this channel at this time
                f0_value = float(f0_values[str(participant_idx)][i])
                f0_class = classify_f0(f0_value)
                
                # Check if there's song at this time
                has_song = is_song_at_time(song_data[participant], time_step)
                
                # Update statistics
                stats[participant]['total_points'] += 1
                
                if has_song:
                    if f0_class == 'call':
                        stats[participant]['song_with_calls'] += 1
                    elif f0_class == 'cage_noise':
                        stats[participant]['song_with_cage_noise'] += 1
                    else:
                        stats[participant]['song_only'] += 1
                else:
                    if f0_class == 'call':
                        stats[participant]['calls_only'] += 1
                    elif f0_class == 'cage_noise':
                        stats[participant]['cage_noise_only'] += 1
                    else:
                        stats[participant]['no_sound'] += 1
        
        # Print debug info about stats
        print("\nStats for first file:")
        for participant in participants:
            print(f"\n{participant}:")
            for key, value in stats[participant].items():
                print(f"  {key}: {value}")
        
        # Calculate percentages
        for participant in participants:
            total = stats[participant]['total_points']
            if total > 0:
                for key in ['song_with_calls', 'song_with_cage_noise', 'song_only',
                           'calls_only', 'cage_noise_only', 'no_sound']:
                    stats[participant][f"{key}_pct"] = \
                        (stats[participant][key] / total) * 100
        
        return {
            'filename': os.path.basename(f0_filepath),
            'stats': stats,
            'duration_seconds': len(time_steps) * 0.1  # assuming 0.1s steps
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
    
    # Calculate overall statistics
    overall_stats = {
        'song_with_calls': 0,
        'song_with_cage_noise': 0,
        'song_only': 0,
        'calls_only': 0,
        'cage_noise_only': 0,
        'no_sound': 0,
        'total_points': 0
    }
    total_duration = 0
    
    # Accumulate statistics
    for result in results:
        total_duration += result['duration_seconds']
        for participant_stats in result['stats'].values():
            for key in overall_stats:
                if key in participant_stats:
                    overall_stats[key] += participant_stats[key]
    
    # Print summary
    print("\n=== Overall Statistics ===")
    print(f"Total files analyzed: {len(results)}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print("\nOverall time distribution:")
    for key in ['song_with_calls', 'song_with_cage_noise', 'song_only',
                'calls_only', 'cage_noise_only', 'no_sound']:
        percentage = (overall_stats[key] / overall_stats['total_points']) * 100 if overall_stats['total_points'] > 0 else 0
        print(f"{key:20s}: {percentage:6.2f}%")
    
    print("\n=== Individual File Statistics ===")
    for result in results:
        print(f"\nFile: {result['filename']}")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
        for participant, stats in result['stats'].items():
            print(f"\nParticipant: {participant}")
            for key in ['song_with_calls', 'song_with_cage_noise', 'song_only',
                       'calls_only', 'cage_noise_only', 'no_sound']:
                print(f"{key:20s}: {stats[f'{key}_pct']:6.2f}%")

    # Create stacked bar chart
    categories = ['song_with_calls', 'song_with_cage_noise', 'song_only',
                 'calls_only', 'cage_noise_only', 'no_sound']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd', (0.5, 0.5, 0.5, 0)]  # Last color is transparent gray
    
    # Sort results by timestamp
    results.sort(key=lambda x: int(x['filename'].split('-')[0]))
    
    # Collect data for plotting
    labels = []  # For x-axis labels
    data = []    # For the actual percentages
    file_boundaries = []  # To track where each file's channels start
    timestamps = []  # To store datetime objects
    current_pos = 0
    
    for result in results:
        file_boundaries.append(current_pos)
        timestamp = int(result['filename'].split('-')[0])
        dt = datetime.fromtimestamp(timestamp)
        date_str = dt.strftime('%Y-%m-%d')
        timestamps.append(dt)
        
        for participant in sorted(result['stats'].keys()):  # Sort participants for consistent ordering
            labels.append(f"{participant}\n({date_str})")
            participant_data = [result['stats'][participant][f"{cat}_pct"] for cat in categories]
            data.append(participant_data)
            current_pos += 1
    
    # Convert to numpy array for easier manipulation
    data = np.array(data)
    
    # Calculate number of bars per subplot
    total_bars = len(labels)
    bars_per_plot = total_bars // 4 + (1 if total_bars % 4 else 0)
    
    # Create the figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(20, 24))
    
    # Process each subplot
    for subplot_idx in range(4):
        ax = axs[subplot_idx]
        start_idx = subplot_idx * bars_per_plot
        end_idx = min((subplot_idx + 1) * bars_per_plot, total_bars)
        
        if start_idx >= total_bars:
            ax.set_visible(False)
            continue
            
        # Get date range for this subplot
        start_date = datetime.fromtimestamp(int(results[start_idx // 2]['filename'].split('-')[0]))
        end_date = datetime.fromtimestamp(int(results[min((end_idx - 1) // 2, len(results) - 1)]['filename'].split('-')[0]))
        ax.set_title(f'Calls from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
        
        # Plot bars for this subplot
        x = np.arange(end_idx - start_idx)  # Create evenly spaced x coordinates
        bottom = np.zeros(end_idx - start_idx)
        
        for i, cat in enumerate(categories):
            ax.bar(x, 
                  data[start_idx:end_idx, i], 
                  bottom=bottom, 
                  label=cat if subplot_idx == 0 else "", 
                  color=colors[i])
            bottom += data[start_idx:end_idx, i]
        
        # Add vertical lines between files
        for boundary in file_boundaries:
            if start_idx <= boundary < end_idx:
                relative_pos = boundary - start_idx
                ax.axvline(x=relative_pos - 0.5, color='black', linestyle='--', alpha=0.2)
        
        # Customize each subplot
        ax.set_ylim(0, 50)  # Set fixed y-axis range
        ax.set_xlim(-0.5, end_idx - start_idx - 0.5)  # Make bars stretch across plot
        ax.set_xticks(x)
        ax.set_xticklabels(labels[start_idx:end_idx], rotation=45, ha='right')
        ax.set_ylabel('Percentage')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Only show legend in the first subplot
        if subplot_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Distribution of Sound Categories by Channel and File', y=0.95, fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sound_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    main() 