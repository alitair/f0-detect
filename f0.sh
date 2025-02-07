#!/bin/bash

# Function to extract audio from an mp4 file and run f0.py
process_mp4() {
    local input_file="$1"
    local output_file="${input_file%.mp4}.wav"  # Change extension to .wav

    echo "Processing $input_file..."

    # Extract all audio tracks from MP4 and save as WAV
    ffmpeg -i "$input_file" -acodec pcm_s16le -ac 0 "$output_file" -y

    if [ $? -eq 0 ]; then
        echo "Successfully extracted audio to $output_file"
        
        # Run f0.py on the extracted wav file
        python f0.py "$output_file"
    else
        echo "Failed to extract audio from $input_file"
    fi
}

# Check if directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Recursively find and process all MP4 files
find "$1" -type f -name "*.mp4" | while read -r file; do
    process_mp4 "$file"
done

exit 0
