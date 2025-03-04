import argparse
import json
import os
import numpy as np
import torch as t
from f0_model import ConversationDataset, load_model, load_json, index_to_time, index_to_f0
import wandb

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
        print(f"results saved to {filepath}")

def perform_inference(args, model, device, input_file, dataset_type):
    output_file = f"{os.path.splitext(input_file)[0]}_p.json"
    dataset = ConversationDataset([input_file], args, inference_mode=True)
    
    ground_truth = []
    predictions = []

    model.eval()
    with t.no_grad():
        # For simplicity, run one sample at a time.
        for i, sample in enumerate(dataset):
            # Ensure tensors are on the correct device.
            sample = {k: v.to(device) for k, v in sample.items()}

            cst, cp, cdt, cf0, ctb = sample['start_time'], sample['participant'], sample['delta_time'], sample['f0'], sample['time_back']
            
 
            preds = model(cp[:-1].unsqueeze(0), cdt[:-1].unsqueeze(0), cf0[:-1].unsqueeze(0), ctb[:-1].unsqueeze(0))
            
            cst, cp, cdt, cf0, ctb = cst.item(), cp.tolist(), cdt.tolist(), cf0.tolist(), ctb.tolist()

            pp  = t.argmax(preds[0], dim=-1).squeeze(0).tolist()
            pdt = t.argmax(preds[1], dim=-1).squeeze(0).tolist()
            pf0 = t.argmax(preds[2], dim=-1).squeeze(0).tolist()


            gt = (cp[-1], cst               , index_to_f0( cf0[-1],args.cutoff), index_to_time( cdt[-1], args.time_cutoff )  )
            ip = (pp[-1], ctb[-2] + pdt[-1] , index_to_f0( pf0[-1],args.cutoff), index_to_time( pdt[-1], args.time_cutoff )  )
        
                
            ground_truth.append(gt)
            predictions.append(ip)     
    
    total_tokens = len(ground_truth)
    correct_p   = 0
    correct_dt  = 0
    correct_f0  = 0
    correct_all = 0

    n_correct = 0
    dt_diffs = []
    f0_diffs = []
    n_diffs  = []
    
    for gt, pred in zip(ground_truth, predictions):
        p_gt, t_gt, f0_gt, dt_gt = gt
        p_p , t_p , f0_p , dt_p  = pred

        n_correct = 0
        if p_gt == p_p:
            correct_p += 1
            n_correct += 1
        if dt_gt == dt_p:
            correct_dt += 1
            n_correct += 1
        if f0_gt == f0_p:
            correct_f0 += 1
            n_correct += 1
        if  n_correct == 3:
            correct_all += 1
        
        dt_diffs.append( abs(dt_gt - dt_p))
        f0_diffs.append( abs(f0_gt - f0_p))
        n_diffs.append( abs(n_correct)/3.0 )

    accuracy = {
        "participant": correct_p   / total_tokens if total_tokens > 0 else 0,
        "dt":          correct_dt  / total_tokens if total_tokens > 0 else 0,
        "f0":          correct_f0  / total_tokens if total_tokens > 0 else 0,
        "all_correct": correct_all / total_tokens if total_tokens > 0 else 0,
        "avg_correct": float(np.mean(n_diffs))    if total_tokens > 0 else 0
    }

    error_distribution = {
        "dt_error_mean": float(np.mean(dt_diffs)) if dt_diffs else 0,
        "dt_error_std" : float(np.std(dt_diffs) ) if dt_diffs else 0,
        "f0_error_mean": float(np.mean(f0_diffs)) if f0_diffs else 0,
        "f0_error_std" : float(np.std(f0_diffs) ) if f0_diffs else 0,
    }


    results = {
        "ground_truth": ground_truth, 
        "prediction": predictions,
        "dataset_type": dataset_type,
        "total_tokens": total_tokens,
        "accuracy": accuracy,
        "error_distribution": error_distribution,
        "args": vars(args)
    }

    save_json(results, output_file)    

    return {
        "input_file": input_file,
        "total_tokens": total_tokens,
        "accuracy": accuracy,
        "error_distribution": error_distribution, 
    }


def aggregate_results(args, dataset_type, file_results):

    total_tokens = sum(res["total_tokens"] for res in file_results)
    if total_tokens == 0:
        return None

    # Aggregate accuracy metrics:
    agg_accuracy = {}
    for metric in ["participant", "dt", "f0", "all_correct", "avg_correct"]:
        total_correct = sum(res["total_tokens"] * res["accuracy"][metric] for res in file_results)
        agg_accuracy[metric] = total_correct / total_tokens

    # For error distributions, use the weighted mean and variance formulas:
    # For time errors:
    sum_dt = sum(res["total_tokens"] * res["error_distribution"]["dt_error_mean"] for res in file_results)
    overall_dt_mean = sum_dt / total_tokens
    # Compute weighted variance: sum(n_i*(s_i^2 + m_i^2)) / total_tokens - overall_mean^2
    sum_dt_var = sum(res["total_tokens"] * (res["error_distribution"]["dt_error_std"]**2 +
                                               res["error_distribution"]["dt_error_mean"]**2)
                       for res in file_results)
    overall_dt_var = sum_dt_var / total_tokens - overall_dt_mean**2
    overall_dt_std = np.sqrt(overall_dt_var)

    # For f0 errors:
    sum_f0 = sum(res["total_tokens"] * res["error_distribution"]["f0_error_mean"] for res in file_results)
    overall_f0_mean = sum_f0 / total_tokens
    sum_f0_var = sum(res["total_tokens"] * (res["error_distribution"]["f0_error_std"]**2 +
                                              res["error_distribution"]["f0_error_mean"]**2)
                     for res in file_results)
    overall_f0_var = sum_f0_var / total_tokens - overall_f0_mean**2
    overall_f0_std = np.sqrt(overall_f0_var)

    agg_error_distribution = {
        "dt_error_mean": overall_dt_mean,
        "dt_error_std": overall_dt_std,
        "f0_error_mean": overall_f0_mean,
        "f0_error_std": overall_f0_std,
    }

    # If a WandB run ID is provided in args, resume that run and update the summary.
    if (getattr(args, "wandb_run_id", None) is not None and getattr(args, "wandb_project", None) is not None ) :

        run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="allow")
        run.log({
            f"{dataset_type}_total_tokens": total_tokens,
            **{f"{dataset_type}_accuracy_{metric}": value for metric, value in agg_accuracy.items()},
            **{f"{dataset_type}_error_{metric}": value for metric, value in agg_error_distribution.items()}
        })

    return {
        "file_results" : file_results,
        "total_tokens": total_tokens,
        "accuracy": agg_accuracy,
        "error_distribution": agg_error_distribution
    }


def process_files(args, model, device, file_path, dataset_type):
    if os.path.exists(file_path):
        inference_results = []
        with open(file_path, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]     
            for input_file in filenames:
                inference_results.append( perform_inference(args, model, device, input_file, dataset_type) )             
        return aggregate_results(args, dataset_type, inference_results)

def main():
    parser = argparse.ArgumentParser(description="Inference for SequenceModel")
    parser.add_argument("-i", "--input", type=str, help="Path to the input file for inference")
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
        perform_inference(args, model, device, args.input, "manual")
    else:
        for dataset_type, file_path in zip(["training", "validation", "test"], ["training.txt", "validation.txt", "test.txt"]):
            agg_results = process_files(args, model, device, file_path , dataset_type)
            output_file = f"{os.path.splitext(file_path)[0]}_results.json"
            save_json(agg_results, output_file) 

if __name__ == "__main__":
    main()
