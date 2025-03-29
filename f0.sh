#!/bin/bash

# Function to check if any _f0.json files exist in a directory
check_f0_exists() {
    local dir="$1"
    if find "$dir" -maxdepth 1 -name "*_f0.json" | grep -q .; then
        return 0  # Found _f0.json files
    else
        return 1  # No _f0.json files found
    fi
}

# Function to extract audio from an mp4 file and run f0.py
process_mp4() {
    local input_file="$1"
    local dir="$(dirname "$input_file")"
    
    # Check if any _f0.json files exist in the directory
    if check_f0_exists "$dir"; then
        return 0  # Skip processing if _f0.json files exist
    fi

    local base_name="${input_file%.mp4}"  # Remove .mp4 extension
    local output_file="${base_name}.wav"  # Change extension to .wav

    # Ensure input file exists
    if [ ! -f "$input_file" ]; then
        echo "Error: Input file $input_file does not exist"
        return 1
    fi

    # Extract all audio tracks from MP4 and save as WAV
    if ! ffmpeg -i "$input_file" -acodec pcm_s16le -ac 0 "$output_file" -y > /dev/null 2>&1; then
        echo "Error: Failed to extract audio from $input_file"
        return 1
    fi

    # Ensure WAV file was created and has size
    if [ ! -s "$output_file" ]; then
        echo "Error: Failed to create WAV file $output_file"
        return 1
    fi

    # Process the WAV file
    if ! python "$SCRIPT_DIR/f0.py" "$output_file"; then
        echo "Error: Failed to process WAV file $output_file"
        return 1
    fi
}

# Check if directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Ensure directory exists and is absolute
if [ ! -d "$1" ]; then
    echo "Error: Directory $1 does not exist"
    exit 1
fi

# Convert to absolute path
INPUT_DIR="$(cd "$1" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -f /home/alistairfraser/.bashrc ]; then
  source /home/alistairfraser/.bashrc
fi
# Source the Conda initialization script.
# Adjust the path below to the correct location for your Conda installation.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate birdnet
else
    echo "Conda initialization script not found!" >&2
    exit 1
fi

# Recursively find and process all MP4 files
find "$INPUT_DIR" -type f -name "*.mp4" | while read -r file; do
    process_mp4 "$file"
done

# Only process WAV files if no _f0.json files exist in the directory
if ! check_f0_exists "$INPUT_DIR"; then
    python "$SCRIPT_DIR/../tweety-net-detector/process_wav_files.py" "--root_dir" "$INPUT_DIR"
    python "$SCRIPT_DIR/combine_segments.py" "--root_dir" "$INPUT_DIR"
fi

exit 0
