#!/bin/bash



# Function to extract audio from an mp4 file and run f0.py
process_mp4() {
    local input_file="$1"
    local base_name="${input_file%.mp4}"  # Remove .mp4 extension
    local json_file="${base_name}_f0.json"
    local output_file="${base_name}.wav"  # Change extension to .wav

    # Check if JSON file exists
    if [ -f "$json_file" ]; then
        # echo "Skipping $input_file: $json_file already exists."
        return 0  # Skip processing
    fi

    # echo "Processing $input_file..."

    # Extract all audio tracks from MP4 and save as WAV
    ffmpeg -i "$input_file" -acodec pcm_s16le -ac 0 "$output_file" -y > /dev/null 2>&1
    python "$SCRIPT_DIR/f0.py" "$output_file" 

}

# Check if directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

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
find "$1" -type f -name "*.mp4" | while read -r file; do
    process_mp4 "$file"
done

python "$SCRIPT_DIR/../tweety-net-detector/process_wav_files.py" "--root_dir" "$1"
python "$SCRIPT_DIR/combine_segments.py" "--root_dir" "$1"

exit 0
