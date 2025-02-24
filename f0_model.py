
import json, math
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset

def load_model(args,device,filepath):
    
    if filepath and os.path.exists(filepath):
        checkpoint = t.load(filepath, map_location=device)
        args.num_participants    = checkpoint["num_participants"]
        args.d_participant       = checkpoint["d_participant"]
        args.num_continuous      = checkpoint["num_continuous"]
        args.d_continuous        = checkpoint["d_continuous"]
        args.cutoff              = checkpoint["cutoff"]
        args.context_length      = checkpoint["context_length"]
        args.prediction_length   = checkpoint["prediction_length"]
        args.transformer_layers  = checkpoint["transformer_layers"]
        args.transformer_nhead   = checkpoint["transformer_nhead"]
        print(f"Loaded model and hyperparameters from {filepath}")
        model = SequenceModel(args.num_participants, args.d_participant, args.num_continuous,
                             args.d_continuous, transformer_layers=args.transformer_layers,
                             transformer_nhead=args.transformer_nhead).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
    else :
        return None
    
def save_model(model, args, filepath=None):
    if filepath is None:
        filepath = os.path.join(args.model_path, args.save_model)

    checkpoint = {
        "num_participants": args.num_participants,
        "d_participant": args.d_participant,
        "num_continuous": args.num_continuous,
        "d_continuous": args.d_continuous,
        "cutoff": args.cutoff,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "transformer_layers": args.transformer_layers,
        "transformer_nhead": args.transformer_nhead,
        "state_dict": model.state_dict()
    }
    t.save(checkpoint, filepath)
    print(f"model saved to {filepath}")

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def pad_window(tokens, target_length, pad_token=(-1, -1, -1), pad_left=True):
    if len(tokens) >= target_length:
        return tokens[-target_length:] if pad_left else tokens[:target_length]
    pads = [pad_token] * (target_length - len(tokens))
    return pads + tokens if pad_left else tokens + pads

class ConversationDataset(Dataset):
    def __init__(self, filepaths, cutoff, context_length, prediction_length):
        self.samples = []
        self.cutoff = cutoff
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        for fp in filepaths:
            self.load(fp, flip=False)
            self.load(fp, flip=True)
        

    def load(self, fp, flip=False):
        min_t, max_t, min_f0, max_f0 = float('inf'), float('-inf'), float('inf'), float('-inf')

        data = load_json(fp)
        time_steps = np.array(data["f0_time_steps"], dtype=float)
        f0_values = {int(k): np.array(v, dtype=float) for k, v in data.get("f0_values", {}).items()}
        tokens = []
        for participant, f0_seq in f0_values.items():
            valid = f0_seq >= self.cutoff
            if not valid.any():
                continue
            for t_step, f0 in zip(time_steps[valid], f0_seq[valid]):
                tokens.append((int(1-participant) if flip else int(participant),
                               int(t_step*10),
                               int(round(f0/10))))
        if not tokens: 
            return
        tokens.sort(key=lambda x: (x[1], x[0]))
        last_pivot = None
        for tok in tokens:
            pivot = tok[1]
            if last_pivot == pivot:
                continue
            last_pivot = pivot
            context_tokens = [t for t in tokens if t[1] <= pivot]
            prediction_tokens = [t for t in tokens if t[1] > pivot]
            if not context_tokens or not prediction_tokens:
                continue
            context_window = pad_window(context_tokens, self.context_length, pad_token=(-1,-1,-1), pad_left=True)
            prediction_window = pad_window(prediction_tokens, self.prediction_length, pad_token=(-1,-1,-1), pad_left=False)
            # Re-center time values relative to the pivot.
            sample = {
                'pivot'       : t.tensor(pivot, dtype=t.float16, device=self.device),
                'context_cat' : t.tensor([c[0] for c in context_window], dtype=t.long, device=self.device),
                'context_time': t.tensor([c[1] - pivot for c in context_window], dtype=t.long, device=self.device),
                'context_f0'  : t.tensor([c[2] for c in context_window], dtype=t.long, device=self.device),
                'target_cat'  : t.tensor([p[0] for p in prediction_window], dtype=t.long, device=self.device),
                'target_time' : t.tensor([ p[1] - pivot if p[1]>0  else -1 for p in prediction_window], dtype=t.long, device=self.device),
                'target_f0'   : t.tensor([p[2] for p in prediction_window], dtype=t.long, device=self.device)
            }
            min_t = min(sample['target_time'].min(), min_t)
            max_t = max(sample['target_time'].max(), max_t)
            min_f0 = min(sample['target_f0'].min(), min_f0)
            max_f0 = max(sample['target_f0'].max(), max_f0)
            self.samples.append(sample)
        print( f"min time={min_t}, max time={max_t}, min_f0={min_f0}, max_f0={max_f0}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
def sinusoidal_time_embedding(val, d_emb, scale_base=1000):
    i = t.arange(d_emb, dtype=t.float)
    angle = val / (scale_base ** (i / d_emb))
    pe = t.empty(d_emb)
    pe[0::2] = t.sin(angle[0::2])
    pe[1::2] = t.cos(angle[1::2])
    return pe

def sin_embeddings(num_emb, d_emb, scale_base=1000):
    embeddings = [sinusoidal_time_embedding(v, d_emb, scale_base) for v in range(num_emb)]
    return nn.Embedding.from_pretrained(t.stack(embeddings), freeze=True)

def sin_encoding(x, d_x, scale_base=1000):
    # x: [batch, seq_len]
    batch, seq_len = x.size()
    return t.stack([t.stack([sinusoidal_time_embedding(x[b,i].item(), d_x, scale_base) for i in range(seq_len)])
                    for b in range(batch)])

class SequenceModel(nn.Module):
    def __init__(self, num_categories, d_participant, num_continuous, d_continuous, transformer_layers=2, transformer_nhead=4):
        super().__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.d_participant = d_participant
        self.d_continuous = d_continuous
        self.category_embedding = nn.Embedding(num_categories, d_participant)
        self.continuous_embedding = sin_embeddings(num_continuous, d_continuous, scale_base=1000)
        self.d_total = d_participant + 2 * d_continuous
        self.pad_embedding = nn.Parameter(t.randn(self.d_total))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_total, nhead=transformer_nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def embed_tokens(self, cat_ids, times, f0_values):
        pad_mask = cat_ids < 0
        safe_cat = cat_ids.clone().clamp(min=0)
        cat_emb = self.category_embedding(safe_cat)
        time_emb = sin_encoding(times, d_x=self.d_continuous).to(cat_emb.device)
        f0_emb = sin_encoding(f0_values, d_x=self.d_continuous).to(cat_emb.device)
        x = t.cat([cat_emb, time_emb, f0_emb], dim=-1)
        pad_emb_exp = self.pad_embedding.unsqueeze(0).unsqueeze(0)
        return t.where(pad_mask.unsqueeze(-1), pad_emb_exp, x)

    def forward(self, context_cat, context_time, context_f0, target_cat, target_time, target_f0):
        context_emb = self.embed_tokens(context_cat, context_time, context_f0)
        target_emb = self.embed_tokens(target_cat, target_time, target_f0)
        x = t.cat([context_emb, target_emb], dim=1)
        combined_mask = t.cat([context_cat < 0, target_cat < 0], dim=1)
        x = self.transformer(x, src_key_padding_mask=combined_mask)
        target_out = x[:, context_emb.size(1):, :]
        pred_cat = target_out[:, :, :self.d_participant]
        pred_time = target_out[:, :, self.d_participant:self.d_participant+self.d_continuous]
        pred_f0 = target_out[:, :, self.d_participant+self.d_continuous:self.d_participant+2*self.d_continuous]
        pred_cat_logits = F.linear(pred_cat, self.category_embedding.weight)
        pred_time_logits = F.linear(pred_time, self.continuous_embedding.weight)
        pred_f0_logits = F.linear(pred_f0, self.continuous_embedding.weight)
        return pred_cat_logits, pred_time_logits, pred_f0_logits
