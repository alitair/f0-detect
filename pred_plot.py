import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm
from f0_model import load_model, dim_time
import os

def main():
    parser = argparse.ArgumentParser( description="Load inference data from a JSON file and plot differences between ground truth and predictions." )
    parser.add_argument("-i", "--input", type=str, help="Path to inference json file")
    parser.add_argument("-m", "--load_model", type=str, default="model.pth",help="Path to the trained model (default: model.pth)")
    parser.add_argument("--model_path", type=str, default="models/")

    args = parser.parse_args()
    
   # Initialize model with architecture parameters matching training defaults.
    model_file = os.path.join(args.model_path, args.load_model)

    if args.load_model and os.path.exists(model_file):
        load_model(args, None , model_file)
    else:
        print(f"Model file {model_file} not found. Exiting.")
        return

    # Load the JSON file provided via the command line
    with open(args.input, 'r') as f:
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

    interval = len(ground_truth) // 10
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

    ax.bar(
        positions,
        pct_gt1_list,
        width=bar_widths,
        bottom=pct_gt0_list,  # Stacked on top of GT=0
        label='GT=1',
        color='red',
        alpha=0.7
    )

    ax.bar(
        positions,
        pct_gt0_list,
        width=bar_widths,
        label='GT=0',
        color='green',
        alpha=0.7
    )


    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('Vocalization Percentage (%)')
    ax.set_ylim(0, 100)  # Ensure y-axis goes from 0 to 100
    ax.set_title('Vocalization Percentage vs. Ground Truth Time')
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
    ax.set_ylabel('Ground Truth FO - Predicted FO')
    ax.set_title('F0 Diff vs. Ground Truth Time by Participant')
    ax.legend()
    ax.grid(True)
    

    
 # ----------------- New Chart A: Bar Chart for Correct Prediction Percentage vs. GT Time -----------------
    ax = axs[1, 0]
    interval = len(ground_truth) // 10
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
    ax.scatter(bar_positions, correct_percentages_gt1, marker='o', color='red', label='gt=1')
    ax.scatter(bar_positions, correct_percentages_gt0, marker='s', color='green', label='gt=0')

    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('Correct (%)')
    ax.set_title('Correct Prediction % vs. Ground Truth Time (Bar Chart)')
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
    ax.set_title('Ground Truth F0 vs. Time by Participant')
    ax.legend()
    ax.grid(True)

    # ----------------- New Charts -----------------
       # Plot 2: (gt_time - p_time) vs. gt_time for gt_cat==1 and gt_cat==0
    gt_time_diff_cat1, gt_time_cat1 = [], []
    gt_time_diff_cat0, gt_time_cat0 = [], []
    
    for gt in ground_truth:
        if gt[0] == 1:
            gt_time_diff_cat1.append(gt[3])
            gt_time_cat1.append(gt[1])
        elif gt[0] == 0:
            gt_time_diff_cat0.append(gt[3])
            gt_time_cat0.append(gt[1])
    
    ax = axs[2, 0]
    ax.scatter(gt_time_cat1, gt_time_diff_cat1, color='red', label='gt_cat 1')
    ax.scatter(gt_time_cat0, gt_time_diff_cat0, color='green', label='gt_cat 0')
    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('Ground Truth DT')
    ax.set_title('Ground Truth DT vs. Time by Participant')
    ax.legend()
    ax.grid(True)
        
    # New Chart B: 2D Histogram for GT F0 vs. Predicted F0
    ax = axs[2, 1]
    gt_f0_vals   = [ gt[2] for gt in ground_truth]
    pred_f0_vals = [ p[2]  for p  in predictions ]

    bins = [10,10]#[args.cutoff[2] , args.cutoff[2]]

    heatmap = ax.hist2d(pred_f0_vals, gt_f0_vals, bins=bins, cmap='plasma')
    max_bin = np.max(heatmap[0])
    if max_bin > 100:
        norm = LogNorm()
    elif max_bin > 10:
        norm = PowerNorm(gamma=0.5)
    else:
        norm = None
    heatmap = ax.hist2d(pred_f0_vals, gt_f0_vals,  bins=bins, cmap='plasma', norm=norm)

    fig.colorbar(heatmap[3], ax=ax)
    ax.set_ylabel('Ground Truth F0')
    ax.set_xlabel('Predicted F0')
    ax.set_title('Ground Truth F0 vs. Predicted F0')
    
    # New Chart C: 2D Histogram for GT Time Difference vs. Predicted Time Difference
    ax = axs[3, 0]
    gt_time_deltas   = []
    pred_time_deltas = []

    x_min, x_max = 0.0, 1.0 #args.time_cutoff
    # bin_edges = np.linspace(x_min, x_max, dim_time(args.time_cutoff)) 
    bin_edges = np.linspace(x_min, x_max, 10 ) 

    for i in range(0, len(ground_truth)):
        if ground_truth[i][3] < 2.0 :
            gt_time_deltas.append(  ground_truth[i][3] )
            pred_time_deltas.append( predictions[i][3] )

    heatmap = ax.hist2d(pred_time_deltas, gt_time_deltas, bins=[bin_edges, bin_edges], cmap='plasma')# bins=time_bins, 
    max_bin = np.max(heatmap[0])
    if max_bin > 100:
        norm = LogNorm()
    elif max_bin > 10:
        norm = PowerNorm(gamma=0.5)
    else:
        norm = None
    heatmap = ax.hist2d(pred_time_deltas, gt_time_deltas, bins=[bin_edges, bin_edges], cmap='plasma', norm=norm) #bins=time_bins, 


    fig.colorbar(heatmap[3], ax=ax)
    ax.set_ylabel('Ground Truth DT')
    ax.set_xlabel('Predicted DT')
    ax.set_title('Ground Truth DT vs. Predicted DT')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()





if __name__ == '__main__':
    main()