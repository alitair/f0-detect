import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import glob

def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_song_files(f0_filepath):
    """Get corresponding song files for each channel in the F0 file.
    Returns a list of (participant, song_file) tuples in the same order as the f0 channels."""
    directory = os.path.dirname(f0_filepath)
    base_name = os.path.basename(f0_filepath)
    parts = base_name.split('_')[0].split('-')
    timestamp = parts[0]
    # Keep participants in order from filename
    participants = parts[1:]
    
    # Return list of tuples to maintain order
    song_files = []
    for participant in participants:
        song_file = os.path.join(directory, f"{timestamp}-{participant}.wav_results.json")
        if os.path.exists(song_file):
            song_files.append((participant, song_file))
        else:
            print(f"Warning: Song file not found: {song_file}")
    
    return song_files

def classify_point(f0_value, is_song):
    """Classify a point as song, call, cage noise, or no sound."""
    if is_song:
        return 'song'
    elif 1700 <= f0_value < 3000:
        return 'call'
    elif f0_value == 0:
        return 'no_sound'
    else:
        return 'cage_noise'

def analyze_file(f0_filepath):
    """Analyze a single F0 file and its corresponding song files."""
    try:
        # Load F0 data
        f0_data = load_json(f0_filepath)
        time_steps = f0_data['f0_time_steps']
        f0_values = {int(k): np.array(v) for k, v in f0_data['f0_values'].items()}
        
        # Get and load song files in correct order
        song_files = get_song_files(f0_filepath)
        if not song_files:
            print(f"Skipping {f0_filepath} - no song files found")
            return None
        
        # Initialize counters for each participant
        stats = {}
        for participant_idx, (participant, song_file) in enumerate(song_files):
            # Verify we have f0 data for this channel
            if participant_idx not in f0_values:
                print(f"Warning: Missing f0 data for channel {participant_idx} ({participant})")
                continue
                
            # Load song data
            song_data = load_json(song_file)
            
            total_points = len(time_steps)
            categories = {
                'song': 0,
                'call': 0,
                'cage_noise': 0,
                'no_sound': 0
            }
            
            # Get song segments for this participant
            song_segments = []
            if song_data.get('song_present', False):
                song_segments = song_data.get('segments', [])
            
            # Analyze each time point
            for i, time_step in enumerate(time_steps):
                f0_value = float(f0_values[participant_idx][i])
                
                # Check if this point is within a song segment
                time_ms = time_step * 1000
                is_song = any(seg['onset_ms'] <= time_ms <= seg['offset_ms'] 
                            for seg in song_segments)
                
                category = classify_point(f0_value, is_song)
                categories[category] += 1
            
            # Calculate percentages
            stats[participant] = {
                cat: (count / total_points) * 100 
                for cat, count in categories.items()
            }
            stats[participant]['duration'] = time_steps[-1] - time_steps[0]
            
            print(f"Processed {f0_filepath} channel {participant_idx} ({participant})")
        
        return {
            'filename': os.path.basename(f0_filepath),
            'stats': stats
        }
        
    except Exception as e:
        print(f"Error processing {f0_filepath}: {e}")
        return None

def find_f0_files(root_dir):
    """Find all F0 files in the root directory and its subdirectories."""
    pattern = os.path.join(root_dir, "**", "*-*-*_f0.json")
    return glob.glob(pattern, recursive=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze bird songs from a directory of F0 files.')
    parser.add_argument('root_dir', help='Root directory to search for F0 files')
    args = parser.parse_args()

    # Find all F0 files
    f0_files = find_f0_files(args.root_dir)
    if not f0_files:
        print(f"No F0 files found in {args.root_dir}")
        return

    print(f"Found {len(f0_files)} F0 files")
    
    # Process each file
    results = []
    for f0_file in f0_files:
        result = analyze_file(f0_file)
        if result:
            results.append(result)
    
    if not results:
        print("No valid files found for analysis.")
        return
    
    # Create plot
    categories = ['song', 'call', 'cage_noise', 'no_sound']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d3d3d3']
    
    # Sort results by timestamp
    for result in results:
        # Extract timestamp from filename
        timestamp = int(result['filename'].split('-')[0])
        result['timestamp'] = timestamp
        # Convert to readable date
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        result['date'] = date
        # Calculate average duration in minutes for the file
        durations = [stats['duration'] for stats in result['stats'].values()]
        avg_duration = sum(durations) / len(durations)
        result['duration_min'] = round(avg_duration / 60)  # Convert to minutes and round
    
    results.sort(key=lambda x: x['timestamp'])
    
    # Split results into 6 time periods
    n = len(results)
    chunk_size = (n + 5) // 6  # Round up division to ensure all data is included
    time_periods = [results[i:i + chunk_size] for i in range(0, n, chunk_size)]
    
    # Create the plot with 6 subplots
    fig, axes = plt.subplots(6, 1, figsize=(15, 30))
    
    # Create legend handles
    legend_handles = [
        plt.Rectangle((0,0),1,1, color=color, alpha=1.0 if cat != 'no_sound' else 0.3)
        for cat, color in zip(categories, colors)
    ]
    
    for period_idx, period_results in enumerate(time_periods):
        ax = axes[period_idx]
        
        # Calculate positions for bars
        bar_width = 0.8
        current_x = 0
        x_positions = []
        x_labels = []
        group_positions = []  # Store middle position of each file group
        group_labels = []    # Store labels for each file group
        
        # Plot stacked bars for each participant in this period
        for result_idx, result in enumerate(period_results):
            group_start_x = current_x
            
            for participant, stats in result['stats'].items():
                x_positions.append(current_x)
                x_labels.append("")  # Empty label for individual bars
                
                bottom = 0
                for cat, color in zip(categories, colors):
                    percentage = stats[cat]
                    if cat == 'no_sound':  # Make no_sound transparent
                        ax.bar(current_x, percentage, bar_width, bottom=bottom, color=color, alpha=0.3)
                    else:
                        ax.bar(current_x, percentage, bar_width, bottom=bottom, color=color)
                    bottom += percentage
                
                current_x += 1
            
            # Calculate middle position for this file group
            group_middle = (group_start_x + current_x - 1) / 2
            group_positions.append(group_middle)
            group_labels.append(f"{result['duration_min']} min\n{result['date']}")
            
            # Add vertical line between files (if not the last file)
            if result_idx < len(period_results) - 1:
                ax.axvline(x=current_x - 0.5, color='black', linestyle='--', alpha=0.3)
        
        # Set title for this period
        if period_results:
            start_date = period_results[0]['date']
            end_date = period_results[-1]['date']
            ax.set_title(f'Calls from {start_date} to {end_date}')
        
        # Customize subplot
        ax.set_ylabel('Percentage')
        ax.set_ylim(0, 50)  # Set y-axis limit to 50%
        ax.set_xticks(group_positions)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        if period_idx == 0:  # Only show legend on first subplot
            ax.legend(legend_handles, categories, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sound_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for result in results:
        print(f"\nFile: {result['filename']}")
        for participant, stats in result['stats'].items():
            print(f"\n{participant} (duration: {stats['duration']:.2f} seconds):")
            for cat in categories:
                print(f"  {cat:10}: {stats[cat]:.1f}%")

if __name__ == "__main__":
    main() 