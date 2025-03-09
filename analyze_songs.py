import os
import json
from datetime import datetime
import glob
from collections import defaultdict

def analyze_json_file(filepath):
    """Analyze a single JSON file and return song statistics if song is present."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if not data.get('song_present', False):
            return None
            
        # Extract timestamp from filename
        filename = os.path.basename(filepath)
        timestamp = int(filename.split('-')[0])
        date = datetime.fromtimestamp(timestamp)
        
        # Calculate total song duration in seconds
        total_duration = 0
        for segment in data.get('segments', []):
            duration_ms = segment['offset_ms'] - segment['onset_ms']
            total_duration += duration_ms / 1000  # Convert to seconds
            
        return {
            'date': date,
            'num_segments': len(data.get('segments', [])),
            'total_duration': total_duration,
            'participant': filename.split('-')[1].split('.')[0],
            'timestamp': timestamp
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    # Find all JSON files matching the pattern
    json_files = glob.glob('data/**/*-*.wav_results.json', recursive=True)
    
    # Analyze each file
    songs_data = []
    for filepath in json_files:
        result = analyze_json_file(filepath)
        if result:
            songs_data.append(result)
    
    if not songs_data:
        print("No files with songs found.")
        return
    
    # Sort by date
    songs_data.sort(key=lambda x: x['timestamp'])
    
    # Calculate summary statistics
    total_files = len(songs_data)
    total_segments = sum(data['num_segments'] for data in songs_data)
    total_duration = sum(data['total_duration'] for data in songs_data)
    
    # Group by date for daily statistics
    daily_stats = defaultdict(lambda: {'files': 0, 'segments': 0, 'duration': 0})
    for data in songs_data:
        date_key = data['date'].strftime('%Y-%m-%d')
        daily_stats[date_key]['files'] += 1
        daily_stats[date_key]['segments'] += data['num_segments']
        daily_stats[date_key]['duration'] += data['total_duration']
    
    # Print summary
    print("\n=== Summary Statistics ===")
    print(f"Total files with songs: {total_files}")
    print(f"Total song segments: {total_segments}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    print("\n=== Daily Statistics ===")
    for date, stats in sorted(daily_stats.items()):
        print(f"\nDate: {date}")
        print(f"  Files with songs: {stats['files']}")
        print(f"  Total segments: {stats['segments']}")
        print(f"  Total duration: {stats['duration']:.2f} seconds ({stats['duration']/60:.2f} minutes)")
    
    print("\n=== Individual File Details ===")
    for data in songs_data:
        print(f"\nParticipant: {data['participant']}")
        print(f"Date: {data['date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Number of segments: {data['num_segments']}")
        print(f"Total duration: {data['total_duration']:.2f} seconds")

if __name__ == "__main__":
    main() 