import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm

def main():
    parser = argparse.ArgumentParser(
        description="Load inference data from a JSON file and plot differences between ground truth and predictions."
    )
    parser.add_argument("input_file", help="Path to the input JSON file")
    args = parser.parse_args()
    
    # Load the JSON file provided via the command line
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    ground_truth = data.get('ground_truth', [])
    predictions  = data.get('prediction', [])

    for i in range(1, len(ground_truth)):
        gt_delta  = ground_truth[i][1] - ground_truth[i-1][1]
        if (  gt_delta < 0 )   :
            print("here gt_delta < 0", ground_truth[i][1], ground_truth[i-1][1], gt_delta)     
    
    if len(ground_truth) != len(predictions):
        print("Warning: Ground truth and predictions lists are not the same length!")
    
    # Create a figure with 2 rows and 2 columns of subplots
    fig, axs = plt.subplots(4, 2, figsize=(14, 10))
    fig.suptitle('Inference Data Analysis', fontsize=16)
    
    # Plot 1: (gt_cat - p_cat) vs. gt_time
 # ----------------- New Chart: Stacked Bar for GT=0 vs. GT=1 Over Intervals of 100 -----------------
    ax = axs[0, 0]

    interval = 50
    positions = []
    bar_widths = []
    pct_gt1_list = []
    pct_gt0_list = []

    for i in range(0, len(ground_truth), interval):
        gt_chunk = ground_truth[i:i+interval]
        if not gt_chunk:
            continue
        
        # Count how many ground-truth = 1 and ground-truth = 0 in this chunk
        total = len(gt_chunk)
        total_1 = sum(1 for gt in gt_chunk if gt[0] == 1)
        total_0 = sum(1 for gt in gt_chunk if gt[0] == 0)
        
        # Compute percentages
        pct_1 = (total_1 / total) * 100
        pct_0 = (total_0 / total) * 100
        
        # Determine the bar's position (avg time) and width (time span)
        first_time = gt_chunk[0][1]
        last_time = gt_chunk[-1][1]
        avg_time = (first_time + last_time) / 2
        positions.append(avg_time)
        bar_widths.append(last_time - first_time)
        
        # Store percentages
        pct_gt1_list.append(pct_1)
        pct_gt0_list.append(pct_0)

    # Plot stacked bars: first GT=0, then stack GT=1 on top
    ax.bar(
        positions,
        pct_gt0_list,
        width=bar_widths,
        label='GT=0',
        color='red',
        alpha=0.7
    )
    ax.bar(
        positions,
        pct_gt1_list,
        width=bar_widths,
        bottom=pct_gt0_list,  # Stacked on top of GT=0
        label='GT=1',
        color='green',
        alpha=0.7
    )

    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)  # Ensure y-axis goes from 0 to 100
    ax.set_title('Stacked Bar: GT=0 vs. GT=1 Over Intervals of 100')
    ax.legend()
    ax.grid(True)

    # Plot 3: (gt_f0 - p_f0) vs. gt_time for gt_cat==1 and gt_cat==0
    f0_diff_cat1, time_f0_cat1 = [], []
    f0_diff_cat0, time_f0_cat0 = [], []
    
    for gt, p in zip(ground_truth, predictions):
        gt_cat = gt[0]
        gt_time = gt[1]
        gt_f0 = gt[2]
        p_f0 = p[2]
        f0_diff = gt_f0 - p_f0
        if gt_cat == 1:
            f0_diff_cat1.append(f0_diff)
            time_f0_cat1.append(gt_time)
        elif gt_cat == 0:
            f0_diff_cat0.append(f0_diff)
            time_f0_cat0.append(gt_time)
    
    ax = axs[0,1]
    ax.scatter(time_f0_cat1, f0_diff_cat1, color='red', label='gt_cat 1')
    ax.scatter(time_f0_cat0, f0_diff_cat0, color='green', label='gt_cat 0')
    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('F0 Diff (gt_f0 - p_f0)')
    ax.set_title('F0 Diff vs. GT Time by Category')
    ax.legend()
    ax.grid(True)
    

    
 # ----------------- New Chart A: Bar Chart for Correct Prediction Percentage vs. GT Time -----------------
    ax = axs[1, 0]
    interval = 50
    bar_positions = []
    bar_widths = []
    correct_percentages = []

    # Lists for gt=1 and gt=0 accuracy lines
    correct_percentages_gt1 = []
    correct_percentages_gt0 = []

    for i in range(0, len(ground_truth), interval):
        gt_chunk = ground_truth[i:i+interval]
        pred_chunk = predictions[i:i+interval]
        if not gt_chunk:
            continue
        
        # Overall correctness in this chunk
        correct = sum(1 for gt, p in zip(gt_chunk, pred_chunk) if gt[0] == p[0])
        percent = (correct / len(gt_chunk)) * 100
        correct_percentages.append(percent)
        
        # Determine the bar's position (avg time) and width (time span)
        first_time = gt_chunk[0][1]
        last_time = gt_chunk[-1][1]
        avg_time = (first_time + last_time) / 2
        bar_positions.append(avg_time)
        bar_widths.append(last_time - first_time)
        
        # Compute correctness for gt=1
        # total_1 is how many ground truth = 1 in this interval
        total_1 = sum(1 for gt in gt_chunk if gt[0] == 1)
        # correct_1 is how many times we predicted correctly among those gt=1
        correct_1 = sum(1 for (gt, p) in zip(gt_chunk, pred_chunk) if gt[0] == 1 and p[0] == 1)
        if total_1 > 0:
            percent_1 = (correct_1 / total_1) * 100
        else:
            percent_1 = 0
        correct_percentages_gt1.append(percent_1)
        
        # Compute correctness for gt=0
        # total_0 is how many ground truth = 0 in this interval
        total_0 = sum(1 for gt in gt_chunk if gt[0] == 0)
        # correct_0 is how many times we predicted correctly among those gt=0
        correct_0 = sum(1 for (gt, p) in zip(gt_chunk, pred_chunk) if gt[0] == 0 and p[0] == 0)
        if total_0 > 0:
            percent_0 = (correct_0 / total_0) * 100
        else:
            percent_0 = 0
        correct_percentages_gt0.append(percent_0)

    # Plot overall correctness as a bar chart
    ax.bar(bar_positions, correct_percentages, width=bar_widths, align='center', color='tab:blue', alpha=0.7, label='Overall')

    # Overlay lines for gt=1 and gt=0 correctness
    ax.plot(bar_positions, correct_percentages_gt1, marker='o', color='red', label='gt=1')
    ax.plot(bar_positions, correct_percentages_gt0, marker='s', color='green', label='gt=0')

    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('Correct Prediction Percentage (%)')
    ax.set_title('Correct Prediction Percentage vs. GT Time (Bar Chart)')
    ax.set_ylim(0, 100)  # Y-axis from 0 to 100
    ax.legend()
    ax.grid(True)



    
    # Plot 4: gt_f0 vs. gt_time for gt_cat==1 and gt_cat==0
    gt_f0_cat1, time_only_cat1 = [], []
    gt_f0_cat0, time_only_cat0 = [], []
    
    for gt in ground_truth:
        gt_cat = gt[0]
        gt_time = gt[1]
        gt_f0 = gt[2]
        if gt_cat == 1:
            gt_f0_cat1.append(gt_f0)
            time_only_cat1.append(gt_time)
        elif gt_cat == 0:
            gt_f0_cat0.append(gt_f0)
            time_only_cat0.append(gt_time)
    
    ax = axs[1, 1]
    ax.scatter(time_only_cat1, gt_f0_cat1, color='red', label='gt_cat 1')
    ax.scatter(time_only_cat0, gt_f0_cat0, color='green', label='gt_cat 0')
    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('Ground Truth F0')
    ax.set_title('GT F0 vs. GT Time by Category')
    ax.legend()
    ax.grid(True)

    # ----------------- New Charts -----------------
       # Plot 2: (gt_time - p_time) vs. gt_time for gt_cat==1 and gt_cat==0
    gt_time_diff_cat1, gt_time_cat1 = [], []
    gt_time_diff_cat0, gt_time_cat0 = [], []
    
    for gt, p in zip(ground_truth, predictions):
        gt_cat = gt[0]
        gt_time = gt[1]
        p_time = p[1]
        time_diff = gt_time - p_time
        if gt_cat == 1:
            gt_time_diff_cat1.append(time_diff)
            gt_time_cat1.append(gt_time)
        elif gt_cat == 0:
            gt_time_diff_cat0.append(time_diff)
            gt_time_cat0.append(gt_time)
    
    ax = axs[2, 0]
    ax.scatter(gt_time_cat1, gt_time_diff_cat1, color='red', label='gt_cat 1')
    ax.scatter(gt_time_cat0, gt_time_diff_cat0, color='green', label='gt_cat 0')
    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('Time Diff (gt_time - p_time)')
    ax.set_title('Time Diff vs. GT Time by Category')
    ax.legend()
    ax.grid(True)
        
    # New Chart B: 2D Histogram for GT F0 vs. Predicted F0

    ax = axs[2, 1]
    gt_f0_vals   = [gt[2] for p,gt in zip(predictions, ground_truth) if gt[2] <= 3500 and p[2] <= 3500]
    pred_f0_vals = [p[2]  for p,gt in zip(predictions, ground_truth) if gt[2] <= 3500 and p[2] <= 3500]

    heatmap = ax.hist2d(gt_f0_vals, pred_f0_vals, bins=[20, 20], cmap='plasma')
        
    # Auto-detect scaling method
    max_bin = np.max(heatmap[0])
    if max_bin > 100:
        norm = LogNorm()
    elif max_bin > 10:
        norm = PowerNorm(gamma=0.5)
    else:
        norm = None
    heatmap = ax.hist2d(gt_f0_vals, pred_f0_vals, bins=[20, 20], cmap='plasma', norm=norm)

    fig.colorbar(heatmap[3], ax=ax)
    ax.set_xlabel('GT F0')
    ax.set_ylabel('Predicted F0')
    ax.set_title('2D Histogram: GT F0 vs. Predicted F0')
    
    # New Chart C: 2D Histogram for GT Time Difference vs. Predicted Time Difference
    ax = axs[3, 0]
    gt_time_deltas = []
    pred_time_deltas = []
    for i in range(1, len(ground_truth)):
        gt_delta  = ground_truth[i][1] - ground_truth[i-1][1]
        pred_delta = predictions[i][1] - predictions[i-1][1]
        if gt_delta <= 2.0 and pred_delta <= 2.0 and gt_delta >= 0.0:
            gt_time_deltas.append(gt_delta)
            pred_time_deltas.append(pred_delta)
        if gt_delta < 0.0:
            print("gt_delta <  0.0", ground_truth[i][1], ground_truth[i-1][1], gt_delta)
    heatmap = ax.hist2d(gt_time_deltas, pred_time_deltas, bins=[20, 20], cmap='plasma')
        
    # Auto-detect scaling method
    max_bin = np.max(heatmap[0])
    if max_bin > 100:
        norm = LogNorm()
    elif max_bin > 10:
        norm = PowerNorm(gamma=0.5)
    else:
        norm = None
    heatmap = ax.hist2d(gt_time_deltas, pred_time_deltas, bins=[20, 20], cmap='plasma', norm=norm)


    print("average gt_time_deltas", sum(gt_time_deltas)/len(gt_time_deltas))
    print("max gt_time_deltas", max(gt_time_deltas))
    print("min gt_time_deltas", min(gt_time_deltas))

    fig.colorbar(heatmap[3], ax=ax)
    ax.set_xlabel('GT Time Delta')
    ax.set_ylabel('Predicted Time Delta')
    ax.set_title('2D Histogram: GT Time Difference vs. Predicted Time Difference')

   

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()





if __name__ == '__main__':
    main()