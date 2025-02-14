import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def extract_f0_values(filenames):
    all_f0_values = []
    for filename in filenames:
        data = load_json(filename)
        f0_values_dict = data["f0_values"]
        
        for key in sorted(f0_values_dict.keys(), key=int):
            f0_values = np.array(f0_values_dict[key])
            f0_values = f0_values[f0_values > 0]  # Ignore zero values
            all_f0_values.extend(f0_values)
    
    return np.array(all_f0_values)

def plot_combined_histogram(filenames):
    all_f0_values = extract_f0_values(filenames)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_f0_values, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title("Combined F0 Frequency Distribution")
    plt.xlabel("F0 Values")
    plt.ylabel("Frequency Count")
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_f0.py <filename1.json> <filename2.json> ...")
        sys.exit(1)
    
    filenames = sys.argv[1:]
    plot_combined_histogram(filenames)
