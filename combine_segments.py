import os
import json
import numpy as np

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

def classify_point(f0_value):
    """Classify an F0 value as call or cage noise."""
    if 1700 <= f0_value < 3000:
        return 'call'
    elif f0_value > 0:  # Any non-zero value outside call range
        return 'cage_noise'
    return None

def find_segments(time_steps, values, segment_type):
    """Find continuous segments of a particular type in the f0 values.
    For single time step segments, the end time will be the next time step that is different.
    Will not start a new segment on the last time step.
    For call and cage_noise segments, includes the f0 value."""
    segments = []
    start_idx = None
    segment_values = []  # Store f0 values for current segment
    
    for i, (time, value) in enumerate(zip(time_steps, values)):
        classification = classify_point(value)
        
        # Don't start a new segment on the last time step
        if classification == segment_type and start_idx is None and i < len(values) - 1:
            start_idx = i
            segment_values = [value]
        elif classification == segment_type and start_idx is not None:
            segment_values.append(value)
        # End current segment
        elif classification != segment_type and start_idx is not None:
            # Convert time to milliseconds for consistency with song file format
            start_time_ms = int(time_steps[start_idx] * 1000)
            # For end time, use this time step since it's different
            end_time_ms = int(time_steps[i] * 1000)
            
            # Only add segments that have some duration
            if end_time_ms > start_time_ms:
                segment = {
                    "onset_ms": start_time_ms,
                    "offset_ms": end_time_ms,
                    "type": segment_type
                }
                # Add f0 for call and cage_noise segments
                if segment_type in ['call', 'cage_noise']:
                    segment["f0"] = float(np.mean(segment_values))
                segments.append(segment)
            start_idx = None
            segment_values = []
        # Handle case where segment extends to end of file
        elif i == len(values) - 1 and start_idx is not None:
            start_time_ms = int(time_steps[start_idx] * 1000)
            end_time_ms = int(time_steps[i] * 1000)
            
            if end_time_ms > start_time_ms:
                segment = {
                    "onset_ms": start_time_ms,
                    "offset_ms": end_time_ms,
                    "type": segment_type
                }
                # Add f0 for call and cage_noise segments
                if segment_type in ['call', 'cage_noise']:
                    segment_values.append(value)  # Include last value
                    segment["f0"] = float(np.mean(segment_values))
                segments.append(segment)
    
    return segments

def merge_overlapping_segments(segments):
    """Merge any overlapping segments of the same type."""
    if not segments:
        return segments
    
    # Sort segments by onset time
    segments.sort(key=lambda x: x['onset_ms'])
    
    merged = []
    current = segments[0]
    
    for next_seg in segments[1:]:
        if next_seg['onset_ms'] <= current['offset_ms'] and next_seg['type'] == current['type']:
            # Merge overlapping segments
            current['offset_ms'] = max(current['offset_ms'], next_seg['offset_ms'])
            # Update f0 if present in both segments
            if current['type'] in ['call', 'cage_noise'] and 'f0' in current and 'f0' in next_seg:
                # Weight averages by segment duration
                curr_duration = current['offset_ms'] - current['onset_ms']
                next_duration = next_seg['offset_ms'] - next_seg['onset_ms']
                total_duration = curr_duration + next_duration
                current['f0'] = (current['f0'] * curr_duration + next_seg['f0'] * next_duration) / total_duration
        else:
            merged.append(current)
            current = next_seg
    
    merged.append(current)
    return merged

def remove_overlaps(segments):
    """Remove overlaps between different types of segments, prioritizing songs > calls > cage noise."""
    if not segments:
        return segments
    
    # Sort segments by priority (song > call > cage_noise) and then by onset time
    priority = {'song': 0, 'call': 1, 'cage_noise': 2}
    segments.sort(key=lambda x: (priority[x['type']], x['onset_ms']))
    
    result = []
    for segment in segments:
        if not result:
            result.append(segment)
            continue
        
        # Check for overlap with existing segments
        overlap = False
        for existing in result:
            if segment['onset_ms'] < existing['offset_ms'] and segment['offset_ms'] > existing['onset_ms']:
                # Handle overlap based on priority
                if priority[segment['type']] > priority[existing['type']]:
                    # Current segment has lower priority, adjust it
                    if segment['onset_ms'] < existing['onset_ms']:
                        # Add segment before existing
                        new_seg = segment.copy()
                        new_seg['offset_ms'] = existing['onset_ms']
                        if new_seg['offset_ms'] > new_seg['onset_ms']:
                            result.append(new_seg)
                    if segment['offset_ms'] > existing['offset_ms']:
                        # Add segment after existing
                        new_seg = segment.copy()
                        new_seg['onset_ms'] = existing['offset_ms']
                        if new_seg['offset_ms'] > new_seg['onset_ms']:
                            result.append(new_seg)
                overlap = True
                break
        
        if not overlap:
            result.append(segment)
    
    return sorted(result, key=lambda x: x['onset_ms'])

def validate_and_fix_segments(segments):
    """Ensure segments are properly ordered, merged, and non-overlapping.
    Returns cleaned segments in chronological order.
    Trims or removes call/cage_noise segments that are within 10ms of a song segment.
    Preserves f0 values when splitting segments."""
    if not segments:
        return segments
        
    # First sort by onset time
    segments.sort(key=lambda x: x['onset_ms'])
    
    # First pass: identify and merge song segments
    song_segments = [s for s in segments if s['type'] == 'song']
    other_segments = [s for s in segments if s['type'] != 'song']
    
    # Merge overlapping song segments
    if song_segments:
        merged_songs = [song_segments[0]]
        for song in song_segments[1:]:
            prev = merged_songs[-1]
            if song['onset_ms'] <= prev['offset_ms']:
                prev['offset_ms'] = max(prev['offset_ms'], song['offset_ms'])
            else:
                merged_songs.append(song)
        song_segments = merged_songs
    
    # Second pass: trim or filter out call/cage_noise segments near songs
    filtered_segments = []
    for segment in other_segments:
        valid_parts = []
        current_part = segment.copy()
        
        for song in song_segments:
            # Add buffer around song
            song_start = song['onset_ms'] - 10
            song_end = song['offset_ms'] + 10
            
            # If segment ends before song buffer starts or starts after song buffer ends, keep as is
            if current_part['offset_ms'] <= song_start or current_part['onset_ms'] >= song_end:
                continue
                
            # If segment is completely within song buffer, skip it
            if current_part['onset_ms'] >= song_start and current_part['offset_ms'] <= song_end:
                current_part = None
                break
                
            # If segment overlaps song buffer start, trim end
            if current_part['onset_ms'] < song_start and current_part['offset_ms'] > song_start:
                current_part['offset_ms'] = song_start
                
            # If segment overlaps song buffer end, trim start
            elif current_part['onset_ms'] < song_end and current_part['offset_ms'] > song_end:
                # Save the part before song if it exists
                if current_part['onset_ms'] < song_start:
                    before_part = current_part.copy()
                    before_part['offset_ms'] = song_start
                    if before_part['offset_ms'] > before_part['onset_ms']:
                        valid_parts.append(before_part)
                
                # Continue with part after song
                current_part['onset_ms'] = song_end
        
        # Add any remaining valid part
        if current_part and current_part['offset_ms'] > current_part['onset_ms']:
            valid_parts.append(current_part)
            
        filtered_segments.extend(valid_parts)
    
    # Combine song segments with filtered segments and sort chronologically
    result = song_segments + filtered_segments
    result.sort(key=lambda x: (x['onset_ms'], -priority.get(x['type'], 3)))
    
    # Verify no zero-duration segments
    result = [seg for seg in result if seg['offset_ms'] > seg['onset_ms']]
    
    return result

# Add priority dictionary at module level
priority = {'song': 0, 'call': 1, 'cage_noise': 2}

def process_file(f0_filepath):
    """Process a single F0 file and its corresponding song files."""
    try:
        # Load F0 data
        f0_data = load_json(f0_filepath)
        time_steps = np.array(f0_data['f0_time_steps'])
        f0_values = {int(k): np.array(v) for k, v in f0_data['f0_values'].items()}
        
        # Get and load song files in correct order
        song_files = get_song_files(f0_filepath)
        if not song_files:
            return
        
        # Process each participant
        for participant_idx, (participant, song_file) in enumerate(song_files):
            # Verify we have f0 data for this channel
            if participant_idx not in f0_values:
                print(f"Warning: Missing f0 data for channel {participant_idx} ({participant})")
                continue
                
            # Load song data
            song_data = load_json(song_file)
            
            # Get song segments and add type field
            segments = []
            if song_data.get('song_present', False):
                for segment in song_data.get('segments', []):
                    segment['type'] = 'song'
                    segments.append(segment)
            
            # Find call segments using the correct channel
            call_segments = find_segments(time_steps, f0_values[participant_idx], 'call')
            segments.extend(call_segments)
            
            # Find cage noise segments using the correct channel
            cage_noise_segments = find_segments(time_steps, f0_values[participant_idx], 'cage_noise')
            segments.extend(cage_noise_segments)
            
            # Final validation to ensure proper ordering and no overlaps
            segments = validate_and_fix_segments(segments)
            
            # Create output data
            output_data = {
                'segments': segments,
                'song_present': any(seg['type'] == 'song' for seg in segments),
                'call_present': any(seg['type'] == 'call' for seg in segments),
                'cage_noise_present': any(seg['type'] == 'cage_noise' for seg in segments)
            }
            
            # Write output file
            output_filename = song_file.replace('.wav_results.json', '.combined.json')
            with open(output_filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Created {output_filename} for channel {participant_idx} ({participant})")
    except Exception as e:
        print(f"Error processing {f0_filepath}: {e}")

def main():
    # Load list of F0 files from all_data.txt
    with open('all_data.txt', 'r') as f:
        f0_files = [line.strip() for line in f]
    
    # Process each file
    for f0_file in f0_files:
        process_file(f0_file)

if __name__ == "__main__":
    main() 