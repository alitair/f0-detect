
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
    def __init__(self, filepaths, cutoff, context_length, prediction_length, num_continuous, inference_mode=False):
        self.samples = []
        self.cutoff = cutoff
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.num_continuous = num_continuous
        for fp in filepaths:
            self.load(fp, flip=False)
            if not inference_mode :
                self.load(fp, flip=True)

    def sortTokens(self, fp, flip=False) :
        data = load_json(fp)
        time_steps = np.array(data["f0_time_steps"], dtype=float)
        f0_values = {int(k): np.array(v, dtype=float) for k, v in data.get("f0_values", {}).items()}

        if len( f0_values ) != 2:
            print(f"Warning: {fp} does not contain 2 participants. len(f0_values)={len(f0_values)}")
            return

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
            return []
        
        tokens.sort(key=lambda x: (x[1], x[0]))

        return tokens

    def load(self, fp, flip=False):

        tokens = self.sortTokens(fp, flip)
        if not tokens: 
            return

        samples = []
        i = self.context_length

        last_pivot = tokens[0][1]

        while ( i < len(tokens) - self.prediction_length - 1 ) :

            if (tokens[i+1][1] == tokens[i][1]):
                i = i + 1 

            pivot = tokens[i][1]

            if ( pivot - last_pivot > 5 ) :

                pos   = i + 1

                context_tokens    = tokens[pos - self.context_length : pos]
                prediction_tokens = tokens[pos : pos + self.prediction_length]

                # no need to pad windows at the moment (it isn't working well)
                # context_window    = pad_window(context_tokens, self.context_length, pad_left=True)
                # prediction_window = pad_window(prediction_tokens, self.prediction_length, pad_left=False)
        
                # Re-center time values relative to the pivot.
                sample = {
                    'pivot'       : t.tensor(pivot, dtype=t.float16, device=self.device),
                    'context_cat' : t.tensor([c[0] for c in context_tokens], dtype=t.long, device=self.device),
                    'context_time': t.tensor([c[1] - pivot for c in context_tokens], dtype=t.long, device=self.device),
                    'context_f0'  : t.tensor([c[2] for c in context_tokens], dtype=t.long, device=self.device),
                    'target_cat'  : t.tensor([p[0] for p in prediction_tokens], dtype=t.long, device=self.device),
                    'target_time' : t.tensor([p[1] - pivot if p[1] > pivot  else -1 for p in prediction_tokens], dtype=t.long, device=self.device),
                    'target_f0'   : t.tensor([p[2] for p in prediction_tokens], dtype=t.long, device=self.device)
                }

                max_t  = sample['target_time'].max().item()
                max_f0 = sample['target_f0'].max().item()
                if max_t >= self.num_continuous or max_f0 >= self.num_continuous:
                    print(f"Warning: {fp} time or f0 value exceeds continuous embedding range: max_t={max_t}, max_f0={max_f0}")
                    return;        
                else:
                    samples.append(sample)

                if ( prediction_tokens[-1][1] == prediction_tokens[-2][1] ):
                    i = i + self.prediction_length      
                else :
                    i = i + self.prediction_length - 1

                last_pivot = prediction_tokens[-2][1]
            else :
                i = i + 1


        print(f"Loaded {fp} with {len(samples)} samples")
        self.samples.extend(samples)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
def sinusoidal_time_embedding(val, d_emb, scale_base=10000):
    i = t.arange(d_emb, dtype=t.float)
    angle = val / (scale_base ** (i / d_emb))
    pe = t.empty(d_emb)
    pe[0::2] = t.sin(angle[0::2])
    pe[1::2] = t.cos(angle[1::2])
    return pe

def sin_embeddings(num_emb, d_emb, scale_base=10000):
    embeddings = [sinusoidal_time_embedding(v, d_emb, scale_base) for v in range(num_emb)]
    return nn.Embedding.from_pretrained(t.stack(embeddings), freeze=True)

def sin_encoding(x, d_x, scale_base=10000):
    # x: [batch, seq_len]
    batch, seq_len = x.size()
    return t.stack([t.stack([sinusoidal_time_embedding(x[b,i].item(), d_x, scale_base) for i in range(seq_len)])
                    for b in range(batch)])

class SequenceModel(nn.Module):
    def __init__(self, num_categories, d_participant, num_continuous, d_continuous, transformer_layers=2, transformer_nhead=4, prediction_length=2):
        super().__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.d_participant = d_participant
        self.d_continuous = d_continuous
        self.category_embedding = nn.Embedding(num_categories, d_participant)
        self.continuous_embedding = sin_embeddings(num_continuous, d_continuous, scale_base=1000)
        self.d_total = d_participant + 2 * d_continuous
        self.pad_embedding = nn.Parameter(t.randn(self.d_total))
        self.prediction_length = prediction_length
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_total, nhead=transformer_nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def embed_tokens(self, cat_ids, times, f0_values):
        # no padding. 
        # pad_mask = cat_ids < 0
        # safe_cat = cat_ids.clone().clamp(min=0)
        # cat_emb  = self.category_embedding(safe_cat)

        cat_emb  = self.category_embedding(cat_ids)
        time_emb = sin_encoding(times, d_x=self.d_continuous).to(cat_emb.device)
        f0_emb   = sin_encoding(f0_values, d_x=self.d_continuous).to(cat_emb.device)
        x = t.cat([cat_emb, time_emb, f0_emb], dim=-1)

        # no padding at the moment
        # pad_emb_exp = self.pad_embedding.unsqueeze(0).unsqueeze(0)
        # return t.where(pad_mask.unsqueeze(-1), pad_emb_exp, x)
        return x
    

    def forward(self, context_cat, context_time, context_f0):
        # Embed the context tokens.
        context_emb = self.embed_tokens(context_cat, context_time, context_f0)
        batch_size = context_emb.size(0)
        
        # Create dummy target embeddings using the pad embedding.
        # Expand so that we have (batch_size, prediction_length, d_total).
        dummy_target = self.pad_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, self.prediction_length, self.d_total)
        
        # Concatenate context and dummy target embeddings.
        x = t.cat([context_emb, dummy_target], dim=1)
        
        seq_len = x.size(1)
        # Create a causal mask: no token can attend to any token that comes later.
        causal_mask = t.triu(t.full((seq_len, seq_len), -1e9, device=x.device), diagonal=1)
        
        # Determine the index where target tokens start.
        target_start = context_emb.size(1)
        # Prevent any target token from attending to any other target token.
        causal_mask[target_start:, target_start:] = -1e9
        
        # Pass through the transformer.
        x = self.transformer(x, mask=causal_mask)
        
        # Extract predictions from the dummy target positions.
        pred_out = x[:, target_start:, :]
        pred_cat = pred_out[:, :, :self.d_participant]
        pred_time = pred_out[:, :, self.d_participant:self.d_participant+self.d_continuous]
        pred_f0 = pred_out[:, :, self.d_participant+self.d_continuous:self.d_participant+2*self.d_continuous]
        
        pred_cat_logits = F.linear(pred_cat, self.category_embedding.weight)
        pred_time_logits = F.linear(pred_time, self.continuous_embedding.weight)
        pred_f0_logits = F.linear(pred_f0, self.continuous_embedding.weight)
        
        return pred_cat_logits, pred_time_logits, pred_f0_logits

