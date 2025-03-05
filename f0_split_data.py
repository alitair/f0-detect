import os, random, itertools
from collections import defaultdict
import argparse
import numpy as np
import torch as t
from f0_model import ConversationDataset, load_model



def extract_participants(filename):
    """
    Given a filename of the form:
      data/{room}/{timestamp}-{participant1}-{participant2}_f0.json
    returns (participant1, participant2)
    """
    base = os.path.basename(filename)           # e.g. "1719941675-USA5494-USA5497_f0.json"
    base = base.split('_')[0]                     # "1719941675-USA5494-USA5497"
    parts = base.split('-')
    if len(parts) >= 3:
        return parts[1], parts[2]
    else:
        raise ValueError(f"Filename {filename} not in expected format.")

def get_file_sample_count(file_path, args):
    """
    Create a ConversationDataset with just the one file (and its flipped version)
    and return the total number of samples.
    """
    ds = ConversationDataset([file_path], args, inference_mode=False)
    return len(ds)

# --- Main splitting routine ---

def make_splits(args, seed=42):
    """
    Reads the file list, computes sample counts, and then starts with all files in training.
    Files are then re-assigned to validation or test only if both participants are represented in training
    (i.e. they appear in at least one other file in training).
    
    The target split proportions are approximately 80% training, 10% validation, and 10% test.
    """
    random.seed(seed)

    # 1. Read all file paths.
    with open(args.input, "r") as f:
        files = [line.strip() for line in f if line.strip()]

    # 2. Compute sample counts and extract participants.
    file_info = []  # Each element is a dict: {'file': ..., 'samples': ..., 'participants': (p1, p2)}
    for fp in files:
        try:
            p1, p2 = extract_participants(fp)
        except Exception as e:
            print(f"Skipping file {fp}: {e}")
            continue

        try:
            sample_count = get_file_sample_count(fp, args)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            continue

        if sample_count == 0:
            continue

        file_info.append({
            'file': fp,
            'samples': sample_count,
            'participants': (p1, p2)
        })

    if not file_info:
        print("No files with samples found!")
        return

    # 3. Compute overall totals and target quotas.
    total_samples = sum(info['samples'] for info in file_info)
    target_train = 0.8 * total_samples
    target_val   = 0.1 * total_samples
    target_test  = 0.1 * total_samples

    # 4. Start with all files in training.
    train_files = file_info.copy()
    test_files = []
    val_files = []

    # Helper: compute participant counts in a given list of files.
    def compute_training_counts(files_list):
        counts = defaultdict(int)
        for info in files_list:
            p1, p2 = info['participants']
            counts[p1] += 1
            counts[p2] += 1
        return counts

    training_counts = compute_training_counts(train_files)
    train_samples = sum(info['samples'] for info in train_files)
    test_samples = 0
    val_samples = 0

    # 5. In a second pass, try to move files from training to test/validation.
    # We iterate in random order to avoid bias.
    files_to_consider = train_files.copy()
    random.shuffle(files_to_consider)

    for info in files_to_consider:
        # Check eligibility: both participants must appear in at least one other file in training.
        p1, p2 = info['participants']
        if training_counts[p1] > 1 and training_counts[p2] > 1:
            # Eligible for moving to test or validation if quota remains.
            if test_samples < target_test:
                test_files.append(info)
                test_samples += info['samples']
            elif val_samples < target_val:
                val_files.append(info)
                val_samples += info['samples']
            else:
                continue  # Both quotas are filled; do not move.
            # Remove the file from training.
            train_files.remove(info)
            train_samples -= info['samples']
            training_counts[p1] -= 1
            training_counts[p2] -= 1
        # Stop early if training quota is reached.
        if train_samples <= target_train:
            break

    # 6. Print summary of sample counts.
    print("Total samples:", total_samples)
    print("Training samples:", train_samples, f"({train_samples/total_samples:.1%})")
    print("Validation samples:", val_samples, f"({val_samples/total_samples:.1%})")
    print("Test samples:", test_samples, f"({test_samples/total_samples:.1%})")

    # Prepare lists of file names for each split.
    train_list = [info['file'] for info in train_files]
    val_list   = [info['file'] for info in val_files]
    test_list  = [info['file'] for info in test_files]

    return train_list, val_list, test_list



def main():
    parser = argparse.ArgumentParser(description="Inference for SequenceModel")
    parser.add_argument("-i", "--input", type=str, default="all_data.txt", help="Path to the input file listing")
    parser.add_argument("-m", "--load_model", type=str, default="model.pth", help="Path to the trained model (default: model.pth)")
    parser.add_argument("--model_path", type=str, default="models/")
    args = parser.parse_args()
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model_file = os.path.join(args.model_path, args.load_model)
    
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found. Exiting.")
        return
    
    model = load_model(args, device, model_file)
    
    if args.input:
        splits = make_splits(args)
        if splits:
            train_list, val_list, test_list = splits
            # You can now save these lists or proceed as needed.
            with open("training.txt", "w") as f:
                for fp in train_list:
                    f.write(fp + "\n")
            with open("validation.txt", "w") as f:
                for fp in val_list:
                    f.write(fp + "\n")
            with open("test.txt", "w") as f:
                for fp in test_list:
                    f.write(fp + "\n")


if __name__ == "__main__":
    main()
