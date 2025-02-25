import json
import matplotlib.pyplot as plt
import argparse

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
    predictions = data.get('prediction', [])
    
    if len(ground_truth) != len(predictions):
        print("Warning: Ground truth and predictions lists are not the same length!")
    
    # Create a figure with 2 rows and 2 columns of subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Inference Data Analysis', fontsize=16)
    
    # Plot 1: (gt_cat - p_cat) vs. gt_time
    gt_time_list = []
    cat_diff = []
    for gt, p in zip(ground_truth, predictions):
        gt_cat = gt[0]
        p_cat = p[0]
        gt_time = gt[1]
        cat_diff.append(gt_cat - p_cat)
        gt_time_list.append(gt_time)
    
    ax = axs[0, 0]
    ax.scatter(gt_time_list, cat_diff, color='blue', label='(gt_cat - p_cat)')
    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('Category Difference')
    ax.set_title('Category Diff vs. GT Time')
    ax.legend()
    ax.grid(True)
    
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
    
    ax = axs[0, 1]
    ax.scatter(gt_time_cat1, gt_time_diff_cat1, color='red', label='gt_cat 1')
    ax.scatter(gt_time_cat0, gt_time_diff_cat0, color='green', label='gt_cat 0')
    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('Time Diff (gt_time - p_time)')
    ax.set_title('Time Diff vs. GT Time by Category')
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
    
    ax = axs[1, 0]
    ax.scatter(time_f0_cat1, f0_diff_cat1, color='red', label='gt_cat 1')
    ax.scatter(time_f0_cat0, f0_diff_cat0, color='green', label='gt_cat 0')
    ax.set_xlabel('Ground Truth Time')
    ax.set_ylabel('F0 Diff (gt_f0 - p_f0)')
    ax.set_title('F0 Diff vs. GT Time by Category')
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
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()