#!/bin/bash
# This script syncs the "tweety" directory from the AWS S3 bucket "oregon.birdconv.mp4"
# to the local directory "/home/alistairfraser/data/buckets/orange.birdconv.mp4/tweety"
# Set script's working directory to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "SCRIPT_DIR: $SCRIPT_DIR"

if [ -f /home/alistairfraser/.bashrc ]; then
  source /home/alistairfraser/.bashrc
fi

# Define the base directory and S3 bucket details.
BASE_DIR="/home/alistairfraser/data/buckets"
BUCKET_NAME="oregon.birdconv.mp4"

# Compose the full destination directory path.
DEST_DIR="${BASE_DIR}/${BUCKET_NAME}"

# Create the destination directory if it doesn't exist.
mkdir -p "$DEST_DIR"

# Sync the S3 directory to the local directory.
aws s3 sync s3://${BUCKET_NAME} "$DEST_DIR"

# Check if the sync command was successful.
if [ $? -eq 0 ]; then
  echo "Sync successful!"
  "$SCRIPT_DIR/f0.sh" "$DEST_DIR"
else
  echo "Sync failed!" >&2
  exit 1
fi