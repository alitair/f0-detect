
import json, math
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
import copy
import itertools
import matplotlib.pyplot as plt


def time_to_int(t0) :
    return int(round( round(t0,1) * 10))

def time_to_index(t0, time_upper_bound):
    time_index = time_to_int(t0)
    if time_index >= 0 :
        return  min( time_index, time_to_int(time_upper_bound) ) 
    else :
        return time_index
    # return  min( max( time_to_int(t0) , 0 ), time_to_int(time_upper_bound) )        

def index_to_time(index, time_upper_bound):
    index = min( max( int(index), 0) , time_to_int(time_upper_bound) )
    return round(float(index)/10.0,1)

def time_indices(time_upper_bound) :
    return np.arange(0, time_to_index(time_upper_bound,time_upper_bound)  + 1, 1)

    # time_upper_bound = round( float(time_upper_bound),1)
    # return np.arange(0, time_upper_bound + 0.1, 0.1)

def dim_time(time_upper_bound):
    return time_to_int(time_upper_bound) + 1

def f0_to_index(f0 , cutoff ) :

    bins         = int(cutoff[2])
    freq_per_bin = float(cutoff[1] - cutoff[0]) / bins 

    f0_index = int( (f0 - cutoff[0])/freq_per_bin  ) 
    if ( f0_index < 0 or f0_index >= bins ) :
        return -1
    else :
        return int(f0_index) 

def index_to_f0(f0_index, cutoff) :
    bins         = int(cutoff[2])
    freq_per_bin = float(cutoff[1] - cutoff[0]) / bins

    f0 = cutoff[0] + f0_index * freq_per_bin + freq_per_bin/2.0

    if ( f0 < cutoff[0] or f0 >= cutoff[1] ) :
        return -1
    else :
        return round(f0)
    
def f0_indices(cutoff) :
    bins         = int(cutoff[2])
    return np.arange(0,bins,1)

    # freq_per_bin = float(cutoff[1] - cutoff[0]) / bins
    # return np.round( np.arange(cutoff[0], cutoff[1], freq_per_bin ) + freq_per_bin/2.0 ) 

def sin_embedding( val , d_emb, L , scale_base=10000 ):

    i = t.arange( d_emb//2, dtype=t.float)

    # ωₖ = 10000^(–2k/d₍model₎) * d₍model₎/L
    # original
    # wk = (scale_base ** ( -2*i / d_emb)) * d_emb / L
    wk = (scale_base ** ( - i / d_emb )) * d_emb / L

    angle =  wk * val

    pe = t.empty(d_emb)
    pe[0::2] = t.sin(angle)
    pe[1::2] = t.cos(angle)

    return pe

def sin_embeddings(values, d_emb, L,  scale_base=10000, freeze=True):
    embeddings = [sin_embedding(float(v), d_emb, L, scale_base=scale_base) for v in values]
    return nn.Embedding.from_pretrained(t.stack(embeddings), freeze=freeze)

def sin_encoding(x, d_x, L, scale_base=10000 ):
    # x: [batch, seq_len]
    batch, seq_len = x.size()
    return t.stack([t.stack([sin_embedding(x[b,i].item(), d_x, L, scale_base=scale_base) for i in range(seq_len)]) for b in range(batch)])



def main():
    d_emb_values = [16, 32, 64]
    L_values     = [10, 32, 100]
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    axs = axs.flatten()
    subplot_index = 0

    for d_emb in d_emb_values:
        for L in L_values:
            # Generate sample points from 0 to L.
            values = np.linspace(0, L, L+1)
            # Compute embedding for each value and stack the results.
            embeddings = t.stack([sin_embedding(t.tensor(val, dtype=t.float), d_emb, L) for val in values])
            # Convert embeddings to a numpy array for plotting.
            emb_np = embeddings.numpy()
            ax = axs[subplot_index]
            im = ax.imshow(emb_np, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f"d_emb = {d_emb}, L = {L}")
            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel("Sample Index (0 to L)")
            fig.colorbar(im, ax=ax)
            subplot_index += 1

    plt.tight_layout()
    plt.show()


def load_model(args,device,filepath):
    
    if filepath and os.path.exists(filepath):
        cp = t.load(filepath, map_location=device)
        args.num_participants    = cp["num_participants"]
        args.d_participant       = cp["d_participant"]
        args.d_time              = cp["d_time"]
        args.d_f0                = cp["d_f0"]
        args.transformer_layers  = cp["transformer_layers"]
        args.transformer_nhead   = cp["transformer_nhead"]
        args.cutoff              = cp["cutoff"]
        args.time_cutoff         = cp["time_cutoff"]
        args.context_length      = cp["context_length"]


        if "wandb_project" in cp:
            args.wandb_project = cp["wandb_project"] = args.wandb_project

        if "wandb_run_id" in cp:
            args.wandb_run_id = cp["wandb_run_id"]

        print(f"Loaded model and hyperparameters from {filepath}")
        model = SequenceModel(args).to(device)
        model.load_state_dict(cp["state_dict"])
        return model
    else :
        return None
    
def save_model(model, args, filepath=None):
    if filepath is None:
        filepath = os.path.join(args.model_path, args.save_model)

    checkpoint = {
        "num_participants"   : args.num_participants,
        "d_participant"      : args.d_participant,        
        "d_time"             : args.d_time,
        "d_f0"               : args.d_f0,
        "transformer_layers" : args.transformer_layers,
        "transformer_nhead"  : args.transformer_nhead,
        "cutoff"             : args.cutoff,
        'time_cutoff'        : args.time_cutoff,
        "context_length"     : args.context_length,
        "state_dict"         : model.state_dict()
    }


    if  args.wandb_project is not None :
        checkpoint["wandb_project"] = args.wandb_project

    if  args.wandb_run_id is not None :
        checkpoint["wandb_run_id"] = args.wandb_run_id

    t.save(checkpoint, filepath)
    print(f"model saved to {filepath}")

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


class ConversationDataset(Dataset):
    def __init__(self, filepaths, args, inference_mode=False):
        self.samples = []
        self.args = args
        self.cutoff = args.cutoff
        self.time_cutoff = args.time_cutoff
        self.context_length = args.context_length
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        for fp in filepaths:
            self.samples.append( self.load(fp) )
            if not inference_mode:
                self.samples.append( self.load(fp,flip=True) )

        self.samples = list(itertools.chain.from_iterable(self.samples))

    def sortTokens(self, fp , flip=False): 
        data = load_json(fp)
        time_steps = np.array(data["f0_time_steps"], dtype=float)
        f0_values = {int(k): np.array(v, dtype=float) for k, v in data.get("f0_values", {}).items()}
        if len( f0_values ) != 2 :
            print(f"Warning: {fp} does not contain 2 participants. len(f0_values)={len(f0_values)}")
            return

        def valid_f0(f0_seq):
            return (f0_seq >= self.cutoff[0]) & (f0_seq < self.cutoff[1])

        if flip :
            f0_values = {1: f0_values[0], 0: f0_values[1]}

        tokens = []
        for ts, f0_0, f0_1 in zip( time_steps, f0_values[0], f0_values[1] ) :
            if valid_f0(f0_0) :
                tokens.append( (0, ts, f0_0) )
            if valid_f0(f0_1) :
                tokens.append( (1, ts, f0_1) )

        return tokens

    def load(self, fp , flip=False):

        def print_sample(sample):
            for pos in range(self.context_length):
                print(" ".join(
                    f"{key}: {value[pos].item()}" if value.ndim > 0 else f"{key}: {value.item()}" for key, value in sample.items()
                ))
            print("-" * 40)

        tokens = self.sortTokens(fp,flip=flip)
        if not tokens: 
            return

        samples = []
        i = 1
        while ( i < len(tokens) - self.context_length ) :
            start_time = tokens[i][1]
            sample = {
                'start_time'  : t.tensor( start_time, dtype=t.float16, device=self.device),
                'participant' : t.tensor([ tokens[pos][0]                                                    for pos in range(i,i+self.context_length)], dtype=t.long, device=self.device),
                'delta_time'  : t.tensor([ time_to_index(tokens[pos][1] - tokens[pos-1][1],self.time_cutoff) for pos in range(i,i+self.context_length)], dtype=t.long, device=self.device),
                'f0'          : t.tensor([ f0_to_index(  tokens[pos][2], self.cutoff)                        for pos in range(i,i+self.context_length)], dtype=t.long, device=self.device),           
                'time_back'   : t.tensor([ tokens[pos][1] - start_time                                       for pos in range(i,i+self.context_length)], dtype=t.float, device=self.device),
            }
            samples.append(sample)
            # print_sample(sample)
            i = i + 1

        print(f"Loaded {fp} with {len(samples)} samples")
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    

class SequenceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.d_participant = args.d_participant
        self.d_time        = args.d_time
        self.d_f0          = args.d_f0
        self.d_total       = self.d_participant + self.d_time + self.d_f0

        self.longest_time  = 36000

        self.num_participants = args.num_participants
        self.num_times        = dim_time(args.time_cutoff)
        self.num_f0           = args.cutoff[2]

        print("Total Embedding size ", self.d_total)
        print(f"-- Embedding size for participants {self.d_participant} [ tokens={self.num_participants} ]")
        print(f"-- Embedding size for time         {self.d_time} [ tokens={self.num_times} ]")
        print(f"-- Embedding size for f0           {self.d_f0} [ tokens={self.num_f0} ]")

        self.ts = self.d_participant
        self.tf = self.ts + self.d_time
   
        self.participant_embeddings = nn.Embedding( args.num_participants , self.d_participant)
        self.time_embeddings        = nn.Embedding( self.num_times        , self.d_time)
        self.f0_embeddings          = nn.Embedding( self.num_f0           , self.d_f0)
        self.pad_embedding          = nn.Parameter(t.randn(self.d_total))

        self.context_length      = args.context_length

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_total, nhead=args.transformer_nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.transformer_layers)

    def embed_tokens(self, participant, delta_time , f0, time_back  ):

        participant_emb  = self.participant_embeddings(participant).to(self.device)
        time_emb         = self.time_embeddings(delta_time).to(self.device)
        f0_emb           = self.f0_embeddings(f0).to(self.device)

        x = t.cat([participant_emb, time_emb, f0_emb], dim=-1)
        pos_emb = sin_encoding(time_back, self.d_total, self.longest_time ).to(self.device)
        x = x + pos_emb

        if self.training:
            x[:, -1, :] = 0.0

        return  x 

    def forward(self, participant, delta_time , f0, time_back ):
        # Embed the context tokens.
        x = self.embed_tokens(participant, delta_time , f0, time_back )
  
        seq_len = x.size(1)
        causal_mask = t.triu(t.full((seq_len, seq_len), -1e9, device=x.device), diagonal=1)
        
        # Pass through the transformer.
        pred_out = self.transformer(x, mask=causal_mask)
        
        pred_cat  = pred_out[:, :,         :self.ts]
        pred_time = pred_out[:, :,  self.ts:self.tf]
        pred_f0   = pred_out[:, :,  self.tf:       ]
        
        pred_cat_logits  = F.linear(pred_cat , self.participant_embeddings.weight )
        pred_time_logits = F.linear(pred_time, self.time_embeddings.weight     )
        pred_f0_logits   = F.linear(pred_f0  , self.f0_embeddings.weight       ) 
        
        return pred_cat_logits, pred_time_logits, pred_f0_logits

if __name__ == "__main__":
    main()

