#!/bin/bash
if [ -f /home/alistairfraser/.bashrc ]; then
  source /home/alistairfraser/.bashrc
fi


# Function to extract audio from an mp4 file and run f0.py
process_mp4() {
    local input_file="$1"
    local base_name="${input_file%.mp4}"  # Remove .mp4 extension
    local json_file="${base_name}_f0.json"
    local output_file="${base_name}.wav"  # Change extension to .wav

    # Check if JSON file exists
    if [ -f "$json_file" ]; then
        echo "Skipping $input_file: $json_file already exists."
        return 0  # Skip processing
    fi

    echo "Processing $input_file..."

    # Extract all audio tracks from MP4 and save as WAV
    ffmpeg -i "$input_file" -acodec pcm_s16le -ac 0 "$output_file" -y > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "Successfully extracted audio to $output_file"
        # Run f0.py on the extracted wav file
        conda activate birdnet
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
