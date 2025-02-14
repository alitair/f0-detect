import json
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load JSON data from file
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

# Plot the f0_values against time_steps
def plot_f0_values(data):
    time_steps = np.array(data["f0_time_steps"])
    f0_values_dict = data["f0_values"]
    
    plt.figure(figsize=(10, 6))
    
    # Plot each f0_values track
    for key in sorted(f0_values_dict.keys(), key=int):
        f0_values = np.array(f0_values_dict[key])
        plt.plot(time_steps[:len(f0_values)], f0_values, label=f"Track {key}")
    
    plt.xlabel("Time Steps")
    plt.ylabel("F0 Values")
    plt.title("F0 Values Over Time")
    plt.legend()
    plt.grid(True)
    
    # Enable interactive zooming
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_f0.py <filename.json>")
        sys.exit(1)
    
    filename = sys.argv[1]
    data = load_json(filename)
    plot_f0_values(data)
