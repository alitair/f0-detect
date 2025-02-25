import argparse
import json
import os
import numpy as np
import torch as t
from f0_model import ConversationDataset, load_model, load_json

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def perform_inference(args, model, device):
    """
    Load the input file into a ConversationDataset and run inference on every sample.
    Aggregates the ground truth and prediction tokens (each token is a tuple: (participant, time_step, f0)).
    """
    dataset = ConversationDataset([args.input], args.cutoff, args.context_length, args.prediction_length,args.num_continuous, inference_mode=True)
    
    inference_ground_truth = []
    inference_predictions = []
    
    cc = -1
    tt= -1
    ff = -1
    model.eval()
    with t.no_grad():
        # For simplicity, run one sample at a time.
        for i, sample in enumerate(dataset):
            # Ensure tensors are on the correct device.
            sample = {k: v.to(device) for k, v in sample.items()}
            # Add a batch dimension.
            
            context_cat   = sample['context_cat'].unsqueeze(0)
            context_time  = sample['context_time'].unsqueeze(0)
            context_f0    = sample['context_f0'].unsqueeze(0)

            # Run model forward pass.
            pred_cat_logits, pred_time_logits, pred_f0_logits = model(context_cat, context_time, context_f0)
            # For each token in the target sequence, choose the argmax as the predicted token.
            pred_cat  = t.argmax(pred_cat_logits , dim=-1).squeeze(0).tolist()
            pred_time = t.argmax(pred_time_logits, dim=-1).squeeze(0).tolist()
            pred_f0   = t.argmax(pred_f0_logits  , dim=-1).squeeze(0).tolist()

            pivot     = int(sample['pivot'].item())
            true_cat  = sample['target_cat'].tolist()
            true_time = sample['target_time'].tolist()
            true_f0   = sample['target_f0'].tolist()
            
            stop = 2 if true_time[0] == true_time[1] else 1
            for i in range(0,stop) :
                inference_ground_truth.append( (true_cat[i], (true_time[i] + pivot)/10, true_f0[i]*10) )
                inference_predictions.append(  (pred_cat[i], (pred_time[i] + pivot)/10, pred_f0[i]*10) )     
    
    return {"ground_truth": inference_ground_truth, "prediction": inference_predictions}


def greedy_predict(args, model, device, ic, horizon):

    model.eval()
    # Add batch dimension.


    pivot         = int(ic['pivot'].item())
    context_cat   = ic['context_cat' ].unsqueeze(0)  # shape: (1, context_length)
    context_time  = ic['context_time'].unsqueeze(0)  # shape: (1, context_length)
    context_f0    = ic['context_f0'  ].unsqueeze(0)  # shape: (1, context_length)
    
    predictions = []
    with t.no_grad():
        for _ in range(horizon):
            # Run the forward pass. The model concatenates context and target and returns predictions
            # only for the target part.
            pred_cat_logits, pred_time_logits, pred_f0_logits = model(context_cat, context_time, context_f0)

            # For each token in the target sequence, choose the argmax as the predicted token.
            pred_cat  = t.argmax(pred_cat_logits, dim=-1).squeeze(0).tolist()
            pred_time = t.argmax(pred_time_logits, dim=-1).squeeze(0).tolist()
            pred_f0   = t.argmax(pred_f0_logits, dim=-1).squeeze(0).tolist()
            
            stop = 2 if pred_time[0] == pred_time[1] else 1
            for i in range(0,stop) :
                predictions.append(  (pred_cat[i], (pred_time[i] + pivot)/10, pred_f0[i]*10) )     
            
            # Update the context window by dropping the oldest token and appending the new prediction.
            # Note: The time values here remain relative to the same pivot.
            context_cat   = t.cat([context_cat[:, 1:] , pred_cat ], dim=1)
            context_time  = t.cat([context_time[:, 1:], pred_time], dim=1)
            context_f0    = t.cat([context_f0[:, 1:]  , pred_f0  ], dim=1)

            context_time  = context_time - pred_time[1]
    
    return predictions



def analyze_results(args, results): 
    """
    Computes simple analysis statistics:
      - Total number of tokens
      - Accuracy per token attribute (participant, time_step, f0) and overall
      - Basic error statistics (mean and std of absolute differences for time and f0)
    """
    ground_truth = results["ground_truth"]
    prediction = results["prediction"]
    
    total_tokens = len(ground_truth)
    correct_participant = 0
    correct_time = 0
    correct_f0 = 0
    overall_correct = 0
    time_diffs = []
    f0_diffs = []
    
    for gt, pred in zip(ground_truth, prediction):
        p_gt, t_gt, f0_gt = gt
        p_pred, t_pred, f0_pred = pred
        if p_gt == p_pred:
            correct_participant += 1
        if t_gt == t_pred:
            correct_time += 1
        if f0_gt == f0_pred:
            correct_f0 += 1
        if gt == pred:
            overall_correct += 1
        
        time_diffs.append(abs(t_gt - t_pred))
        f0_diffs.append(abs(f0_gt - f0_pred))
    
    analysis = {
        "total_tokens": total_tokens,
        "args": vars(args),
        "accuracy": {
            "participant": correct_participant / total_tokens if total_tokens > 0 else 0,
            "time": correct_time / total_tokens if total_tokens > 0 else 0,
            "f0": correct_f0 / total_tokens if total_tokens > 0 else 0,
            "overall": overall_correct / total_tokens if total_tokens > 0 else 0
        },
        "error_distribution": {
            "time_error_mean": float(np.mean(time_diffs)) if time_diffs else 0,
            "time_error_std": float(np.std(time_diffs)) if time_diffs else 0,
            "f0_error_mean": float(np.mean(f0_diffs)) if f0_diffs else 0,
            "f0_error_std": float(np.std(f0_diffs)) if f0_diffs else 0
        }
    }
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Inference for SequenceModel")
    parser.add_argument("-i", "--input", type=str,
                        help="Path to the json file to be used to fill in the context window for inference")
    parser.add_argument("-m", "--load_model", type=str, default="model.pth",
                        help="Path to the trained model (default: model.pth)")
    parser.add_argument("--model_path", type=str, default="models/")
    # Remove -o and -a options.
    parser.add_argument("-io", "--io_results", type=str,
                        help="If provided, inference is not performed; this file is used as the results of inference to produce analysis")
    args = parser.parse_args()
    
    # Compute output file paths based on the input file.
    if args.input:
        input_dir = os.path.dirname(args.input)
        input_filename = os.path.basename(args.input)
        base, ext = os.path.splitext(input_filename)
        output_filepath = os.path.join(input_dir, f"{base}_p.json")
        analysis_filepath = os.path.join(input_dir, f"{base}_a.json")
    else:
        print("Error: Input file must be provided.")
        return
    
    # Set device.
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    
    # If io_results is provided, load that and perform analysis only.
    if args.io_results:
        if not os.path.exists(args.io_results):
            print(f"Error: The provided io_results file {args.io_results} does not exist.")
            return
        results = load_json(args.io_results)
        analysis = analyze_results(args, results)
        save_json(analysis, analysis_filepath)
        print(f"Analysis saved to {analysis_filepath}")
        return
    
    # Initialize model with architecture parameters matching training defaults.
    model_file = os.path.join(args.model_path, args.load_model)
    if args.load_model and os.path.exists(model_file):
        model = load_model(args, device, model_file)
    else:
        print(f"Model file {model_file} not found. Exiting.")
        return
    
    # Run inference.
    results = perform_inference(args, model, device)
    save_json(results, output_filepath)
    print(f"Inference results saved to {output_filepath}")
    
    # Compute analysis based on the inference results.
    analysis = analyze_results(args, results)
    save_json(analysis, analysis_filepath)
    print(f"Analysis saved to {analysis_filepath}")

if __name__ == "__main__":
    main()