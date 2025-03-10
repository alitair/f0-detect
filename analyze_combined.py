import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_combined_files(f0_filepath):
    """Get corresponding combined files for each participant in the F0 file.
    Returns a list of (participant, combined_file) tuples in the same order as the f0 channels."""
    directory = os.path.dirname(f0_filepath)
    base_name = os.path.basename(f0_filepath)
    parts = base_name.split('_')[0].split('-')
    timestamp = parts[0]
    # Keep participants in order from filename
    participants = parts[1:]
    
    # Return list of tuples to maintain order
    combined_files = []
    for participant in participants:
        combined_file = os.path.join(directory, f"{timestamp}-{participant}.combined.json")
        if os.path.exists(combined_file):
            combined_files.append((participant, combined_file))
        else:
            print(f"Warning: Combined file not found: {combined_file}")
    
    return combined_files

def analyze_file(f0_filepath):
    """Analyze combined files for a single recording."""
    try:
        # Get and load combined files in correct order
        combined_files = get_combined_files(f0_filepath)
        if not combined_files:
            print(f"Skipping {f0_filepath} - no combined files found")
            return None
        
        # Initialize counters for each participant
        stats = {}
        for participant, combined_file in combined_files:
            # Load combined data
            combined_data = load_json(combined_file)
            segments = combined_data.get('segments', [])
            
            # Calculate total duration from segments
            if segments:
                start_time = min(seg['onset_ms'] for seg in segments)
                end_time = max(seg['offset_ms'] for seg in segments)
                duration = (end_time - start_time) / 1000  # Convert to seconds
            else:
                # If no segments, use a default duration
                duration = 0
            
            # Initialize counters for each category
            total_time = {
                'song': 0,
                'call': 0,
                'cage_noise': 0,
                'no_sound': duration * 1000  # Initialize with total duration in ms
            }
            
            # Count time for each segment
            for segment in segments:
                segment_duration = segment['offset_ms'] - segment['onset_ms']
                total_time[segment['type']] += segment_duration
                total_time['no_sound'] -= segment_duration
            
            # Convert to percentages
            total_ms = sum(total_time.values())
            if total_ms > 0:  # Avoid division by zero
                stats[participant] = {
                    cat: (time_ms / total_ms) * 100 
                    for cat, time_ms in total_time.items()
                }
                stats[participant]['duration'] = duration
            
            print(f"Processed {combined_file}")
        
        return {
            'filename': os.path.basename(f0_filepath),
            'stats': stats
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
    
    # Split results into 4 time periods
    n = len(results)
    chunk_size = (n + 3) // 4  # Round up division to ensure all data is included
    time_periods = [results[i:i + chunk_size] for i in range(0, n, chunk_size)]
    
    # Create the plot with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    
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
    plt.savefig('sound_distribution_combined.png', bbox_inches='tight', dpi=300)
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