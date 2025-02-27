import argparse
import os
import time
import math
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from f0_model import ConversationDataset, SequenceModel,load_model,save_model,dim_time
from tqdm import tqdm

# Optional wandb logging.
try:
    import wandb
except ImportError:
    wandb = None

def load_filepaths(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def prepare_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def compute_loss(batch, model, ce_loss, num_participants, num_continuous):
    # Clone target tensors before running the model to preserve the originals.
    context_cat = batch['context_cat'].detach().clone()
    context_time = batch['context_time'].detach().clone()
    context_f0 = batch['context_f0'].detach().clone()
    
    preds = model(context_cat, context_time, context_f0)
    
    loss_cat  = ce_loss(preds[0].view(-1, num_participants), batch['target_cat'].view(-1))
    loss_time = ce_loss(preds[1].view(-1, num_continuous  ), batch['target_time'].view(-1))
    loss_f0   = ce_loss(preds[2].view(-1, num_continuous  ), batch['target_f0'].view(-1))

    if t.isnan(loss_cat) or t.isnan(loss_time) or t.isnan(loss_f0):
        print( "context_cat",context_cat)
        print( "context_time",context_time)
        print( "context_f0",context_f0)
        print( "batch[target_cat]",batch['target_cat'].view(-1))
        print( "batch[target_time]",batch['target_time'].view(-1))
        print( "batch[target_f0]",batch['target_f0'].view(-1))
        print( "preds[0]",preds[0].view(-1, num_participants))
        print( "preds[1]",preds[1].view(-1, num_continuous))
        print( "preds[2]",preds[2].view(-1, num_continuous))
        raise ValueError("NaN loss encountered during training.")


    return loss_cat + loss_time + loss_f0


def evaluate_model(model, dataloader, ce_loss, device, num_participants, num_continuous, name="Validation"):
    model.eval()
    # model.train()
    total_loss, count = 0.0, 0
    with t.no_grad():
        for batch in dataloader:
            prepared = prepare_batch(batch, device)
            total_loss += compute_loss(prepared, model, ce_loss, num_participants, num_continuous).item()
            count += 1
    avg_loss = total_loss / count if count > 0 else float('inf')
    print(f"{name} Loss: {avg_loss:.4f}")
    return avg_loss

def profile_training(model, dataloader, optimizer, ce_loss, device, num_participants, num_continuous):
    model.train()
    batch = prepare_batch(next(iter(dataloader)), device)
    cc, ct, cf = batch['context_cat'], batch['context_time'], batch['context_f0']
    tc, tt, tf = batch['target_cat'], batch['target_time'], batch['target_f0']

    start = time.perf_counter()
    preds = model(cc, ct, cf)
    forward_time = (time.perf_counter() - start) * 1000

    loss = ce_loss(preds[0].view(-1, num_participants), tc.view(-1)) + \
           ce_loss(preds[1].view(-1, num_continuous), tt.view(-1)) + \
           ce_loss(preds[2].view(-1, num_continuous), tf.view(-1))

    start = time.perf_counter()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backward_time = (time.perf_counter() - start) * 1000

    mem = t.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Forward: {forward_time:.2f} ms, Backward: {backward_time:.2f} ms")
    if mem is not None:
        print(f"GPU Memory: {mem:.2f} MB")
    print(f"Total params: {total_params}, Trainable: {trainable_params}")
    return {"forward_time_ms": forward_time, "backward_time_ms": backward_time,
            "memory_allocated_mb": mem, "total_params": total_params,
            "trainable_params": trainable_params}

def main():

    def cutoff_tuple(s):
        parts = s.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Cutoff must be specified as 'lower,upper,divisions'")
        try:
            lower     = int(parts[0])
            upper     = int(parts[1])
            divisions = int(parts[2])
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid cutoff values")
        return (lower, upper, divisions)


    parser = argparse.ArgumentParser(description="Train SequenceModel")
    parser.add_argument("--training_filepath", type=str, default="training.txt")
    parser.add_argument("--validation_filepath", type=str, default="validation.txt")
    parser.add_argument("--test_filepaths", type=str, default="test.txt")
    parser.add_argument("--num_participants", type=int, default=2)
    parser.add_argument("--d_embedding", type=int, default=80)
    parser.add_argument("--transformer_layers", type=int, default=4)
    parser.add_argument("--transformer_nhead", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--cutoff"    , type=cutoff_tuple, default="1700,3000,26",  help="Cutoff in format: lower,upper,bins")
    parser.add_argument("--time_cutoff", type=float , default=3.0 ,  help="Upper time cutoff")
    parser.add_argument("--context_length", type=int, default=20)
    parser.add_argument("--prediction_length", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="models/")
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--save_model", type=str, default="model.pth")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Epochs to wait for improvement before early stopping")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_reporting_interval", type=int, default=10)
    args = parser.parse_args()

    if args.wandb and wandb is not None:
        wandb.init(project="sequence_model_training", config=vars(args))
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    print("Using device:", device)
    os.makedirs(args.model_path, exist_ok=True)


    # Initialize model using provided training arguments.
    if args.load_model and os.path.exists(os.path.join(args.model_path, args.load_model)):
        model = load_model(args,device, os.path.join(args.model_path, args.load_model))
    else:
        model = SequenceModel(args).to(device)

    # Load datasets.
    train_paths = load_filepaths(args.training_filepath)
    training_dataset = ConversationDataset(train_paths, args)
    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if os.path.exists(args.validation_filepath):
        val_paths = load_filepaths(args.validation_filepath)
        val_dataset = ConversationDataset(val_paths, args)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_loader = None
    if os.path.exists(args.test_filepaths):
        test_paths = load_filepaths(args.test_filepaths)
        test_dataset = ConversationDataset(test_paths, args)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)



    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    # Setup learning rate scheduler.
    # If validation set is provided, we use ReduceLROnPlateau; otherwise, StepLR.
    if val_loader:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Mixed precision setup.
    scaler = t.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Profile training step.
    prof_info = profile_training(model, train_loader, optimizer, ce_loss, device,
                                 args.num_participants, args.num_continuous)
    if args.wandb and wandb is not None:
        wandb.log({"profiling": prof_info})

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_checkpoint = os.path.join(args.model_path, "best_" + args.save_model)

    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False)
        for batch in pbar:
            batch = prepare_batch(batch, device)
            optimizer.zero_grad()
            if scaler is not None:
                with t.cuda.amp.autocast():
                    loss = compute_loss(batch, model, ce_loss, args.num_participants, args.num_continuous)
                scaler.scale(loss).backward()
                model.freeze_time_embeddings()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = compute_loss(batch, model, ce_loss, args.num_participants, args.num_continuous)
                loss.backward()
                model.freeze_time_embeddings()
                optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = epoch_loss / len(train_loader)
        # print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # Validate each epoch if validation data is available.
        if val_loader:
            val_loss = evaluate_model(model, val_loader, ce_loss, device,args.num_participants, args.num_continuous, name="Validation")
            scheduler.step(val_loss)
            # Checkpointing and early stopping.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_model( model, args, filepath=best_checkpoint)
                print(f"Checkpoint saved at epoch {epoch} with val_loss: {val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
            if args.wandb and wandb is not None:
                wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss})
        else:
            scheduler.step()
            if args.wandb and wandb is not None:
                wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    if args.wandb and wandb is not None:
        wandb.log({"total_training_time_sec": total_time, "device": device.type})
    
    # Load best model from checkpoint if validation was used.
    if val_loader and os.path.exists(best_checkpoint):
        model = load_model( args,device, best_checkpoint )
        print(f"Loaded best checkpoint from {best_checkpoint}")

    if test_loader:
        test_loss = evaluate_model(model, test_loader, ce_loss, device,  args.num_participants, args.num_continuous, name="Test")
        if args.wandb and wandb is not None:
            wandb.log({"test_loss": test_loss})

    save_model( model, args)        
    if args.wandb and wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
