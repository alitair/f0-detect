import argparse
import os
import time
import math
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from f0_model import ConversationDataset, SequenceModel, load_model, save_model, dim_time
from tqdm import tqdm
from dataclasses import dataclass

# Optional wandb logging.
try:
    import wandb
except ImportError:
    wandb = None

@dataclass
class LossStats:
    total: float = 0.0
    cat: float = 0.0
    time: float = 0.0
    f0: float = 0.0
    count: int = 0

    def update(self, tot, cat, tim, f0):
        self.total += tot
        self.cat   += cat
        self.time  += tim
        self.f0    += f0
        self.count += 1

    def averages(self):
        if self.count == 0:
            return {"total": 0, "cat": 0, "time": 0, "f0": 0}
        return {
            "total": self.total / self.count,
            "cat":   self.cat   / self.count,
            "time":  self.time  / self.count,
            "f0":    self.f0    / self.count
        }

def load_filepaths(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def prepare_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def compute_loss(batch, model, ce_loss, stats=None):
    # Extract full sequences.
    cp, cdt, cf0, ctb = batch['participant'], batch['delta_time'], batch['f0'], batch['time_back']
    
    # Use only context tokens (all tokens except the last one) as input.
    cp_in  = cp[:, :-1]
    cdt_in = cdt[:, :-1]
    cf0_in = cf0[:, :-1]
    ctb_in = ctb[:, :-1]
    
    # Run model forward pass with the context only.
    preds = model(cp_in, cdt_in, cf0_in, ctb_in)
    
    # Compute losses against targets (all tokens except the first one).
    loss_cat  = ce_loss(preds[0].reshape(-1, model.num_participants),  cp[:, 1:].reshape(-1))
    loss_time = ce_loss(preds[1].reshape(-1, model.num_times),        cdt[:, 1:].reshape(-1))
    loss_f0   = ce_loss(preds[2].reshape(-1, model.num_f0),           cf0[:, 1:].reshape(-1))
    total_loss = loss_cat + loss_time + loss_f0

    if stats is not None:
        stats.update(total_loss.item(), loss_cat.item(), loss_time.item(), loss_f0.item())
    return total_loss


def train_epoch(model, train_loader, ce_loss, optimizer, device, scaler=None, epoch=0, num_epochs=0):
    model.train()
    stats = LossStats()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
    for batch in pbar:
        batch = prepare_batch(batch, device)
        optimizer.zero_grad()
        if scaler is not None:
            with t.cuda.amp.autocast():
                tot = compute_loss(batch, model, ce_loss, stats)
            scaler.scale(tot).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            tot = compute_loss(batch, model, ce_loss, stats)
            tot.backward()
            optimizer.step()
        pbar.set_postfix(loss=f"{tot.item():.4f}")
    return stats.averages()

def evaluate(model, loader, ce_loss, device, name="Validation"):
    model.eval()
    stats = LossStats()
    with t.no_grad():
        for batch in loader:
            batch = prepare_batch(batch, device)
            _ = compute_loss(batch, model, ce_loss, stats)
    avg = stats.averages()
    print(f"{name} Loss: Total: {avg['total']:.4f}, Cat: {avg['cat']:.4f}, Time: {avg['time']:.4f}, f0: {avg['f0']:.4f}")
    return avg

def profile_training(model, dataloader, optimizer, ce_loss):
    model.train()
    batch = prepare_batch(next(iter(dataloader)), model.device)

    # NOTE this isn't a valid prediction, just a way to profile the forward and backward pass.

    cp, cdt, cf0, ctb = batch['participant'], batch['delta_time'], batch['f0'], batch['time_back']

    start = time.perf_counter()
    preds = model(cp, cdt, cf0, ctb)
    forward_time = (time.perf_counter() - start) * 1000

    loss = ce_loss(preds[0].view(-1, model.num_participants),  cp.reshape(-1)) + \
           ce_loss(preds[1].view(-1, model.num_times),        cdt.reshape(-1)) + \
           ce_loss(preds[2].view(-1, model.num_f0),           cf0.reshape(-1))

    start = time.perf_counter()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backward_time = (time.perf_counter() - start) * 1000

    mem = t.cuda.memory_allocated(model.device) / (1024 ** 2) if model.device.type == "cuda" else None
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
    parser.add_argument("--cutoff", type=cutoff_tuple, default="1700,3000,26", help="Cutoff in format: lower,upper,bins")
    parser.add_argument("--time_cutoff", type=float, default=3.0, help="Upper time cutoff")

    parser.add_argument("--d_participant", type=int, default=32) #64
    parser.add_argument("--d_time", type=int, default=32)#64
    parser.add_argument("--d_f0", type=int, default=32)#64
    parser.add_argument("--transformer_layers", type=int, default=6) #6
    parser.add_argument("--transformer_nhead", type=int, default=4)

    parser.add_argument("--context_length", type=int, default=100) #100

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=1)

    parser.add_argument("--model_path", type=str, default="models/")
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--save_model", type=str, default="model.pth")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--early_stopping_patience", type=int, default=30,help="Epochs to wait for improvement before early stopping")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_reporting_interval", type=int, default=10)
    args = parser.parse_args()

    if args.wandb and wandb is not None:
        args.wandb_project = "sequence_model_training"
        run = wandb.init(project=args.wandb_project, config=vars(args))
        args.wandb_run_id = run.id
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs(args.model_path, exist_ok=True)

    # Initialize model.
    if args.load_model and os.path.exists(os.path.join(args.model_path, args.load_model)):
        model = load_model(args, device, os.path.join(args.model_path, args.load_model))
    else:
        model = SequenceModel(args).to(device)

    # Load datasets.
    def load_or_create_dataset(file_path, cache_file, args):
        # if os.path.exists(cache_file):
        #     dataset = t.load(cache_file)
        # else:
        paths = load_filepaths(file_path)
        dataset = ConversationDataset(paths, args)
        # t.save(dataset, cache_file)
        return dataset

    # Training dataset.
    training_dataset = load_or_create_dataset(args.training_filepath, "training.pth", args)
    train_loader     = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    # Validation dataset.
    val_loader = None
    if os.path.exists(args.validation_filepath):
        val_dataset = load_or_create_dataset(args.validation_filepath, "validation.pth", args)
        val_loader  = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Test dataset.
    test_loader = None
    if os.path.exists(args.test_filepaths):
        test_dataset = load_or_create_dataset(args.test_filepaths, "test.pth", args)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    if val_loader:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    scaler = t.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Profile training step.
    prof_info = profile_training(model, train_loader, optimizer, ce_loss)
    if args.wandb and wandb is not None:
        wandb.log({"profiling": prof_info})

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_checkpoint = os.path.join(args.model_path, "best_" + args.save_model)

    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        train_avg = train_epoch(model, train_loader, ce_loss, optimizer, device, scaler, epoch, args.num_epochs)
        if val_loader:
            val_avg = evaluate(model, val_loader, ce_loss, device, name="Validation")
            scheduler.step(val_avg["total"])
            if val_avg["total"] < best_val_loss:
                best_val_loss = val_avg["total"]
                epochs_no_improve = 0
                save_model(model, args, filepath=best_checkpoint)
                print(f"Checkpoint saved at epoch {epoch} with val_loss: {val_avg['total']:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
            if args.wandb and wandb is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss_total": train_avg["total"],
                    "train_loss_cat": train_avg["cat"],
                    "train_loss_time": train_avg["time"],
                    "train_loss_f0": train_avg["f0"],
                    "val_loss_total": val_avg["total"],
                    "val_loss_cat": val_avg["cat"],
                    "val_loss_time": val_avg["time"],
                    "val_loss_f0": val_avg["f0"],
                })
        else:
            scheduler.step()
            if args.wandb and wandb is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss_total": train_avg["total"],
                    "train_loss_cat": train_avg["cat"],
                    "train_loss_time": train_avg["time"],
                    "train_loss_f0": train_avg["f0"],
                })

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    if args.wandb and wandb is not None:
        wandb.log({"total_training_time_sec": total_time, "device": device.type})
    
    if val_loader and os.path.exists(best_checkpoint):
        model = load_model(args, device, best_checkpoint)
        print(f"Loaded best checkpoint from {best_checkpoint}")

    if test_loader:
        test_avg = evaluate(model, test_loader, ce_loss, device, name="Test")
        if args.wandb and wandb is not None:
            wandb.log({
                "test_loss_total": test_avg["total"],
                "test_loss_cat": test_avg["cat"],
                "test_loss_time": test_avg["time"],
                "test_loss_f0": test_avg["f0"],
            })

    save_model(model, args)
    if args.wandb and wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()