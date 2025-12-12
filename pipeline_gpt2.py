
import os
import sys
import math
import time
import argparse
import struct
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# -----------------------------------------------------------------------------
# Utils & Config
# -----------------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class GPT2Config:
    def __init__(self, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, block_size=1024, bias=True, padded_vocab_size=None):
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size if padded_vocab_size is not None else vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.bias = bias

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, NH, HS) -> (B, NH, T, HS)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU(approximate='tanh') # Matches C implementation
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PipelineGPT2(nn.Module):
    def __init__(self, config, checkpoint_path=None):
        super().__init__()
        self.config = config
        
        # Devices
        self.dev0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dev1 = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Model Components
        vocab_size_actual = config.padded_vocab_size
        C = config.n_embd
        
        # Stage 1
        self.wte = nn.Embedding(vocab_size_actual, C).to(self.dev0)
        self.wpe = nn.Embedding(config.block_size, C).to(self.dev0)
        
        self.split_layer = config.n_layer // 2
        self.blocks_part1 = nn.ModuleList([Block(config) for _ in range(self.split_layer)]).to(self.dev0)
        
        # Stage 2
        self.blocks_part2 = nn.ModuleList([Block(config) for _ in range(self.split_layer, config.n_layer)]).to(self.dev1)
        self.ln_f = LayerNorm(C, bias=config.bias).to(self.dev1)
        self.lm_head = nn.Linear(C, vocab_size_actual, bias=False).to(self.dev1)
        
        if checkpoint_path:
            self.load_from_binary(checkpoint_path)

    def load_from_binary(self, path):
        print(f"Loading model from {path}")
        if not os.path.exists(path):
            print(f"Error: checkpoint {path} not found")
            sys.exit(1)
            
        with open(path, 'rb') as f:
            header = f.read(256 * 4) # 256 ints
            header_ints = struct.unpack('256i', header)
            
            # Header checks matching train_gpt2.c
            magic = header_ints[0]
            if magic != 20240326:
                print("Bad magic model file")
                sys.exit(1)
            version = header_ints[1]
            if version != 3:
                print("Bad version in model file")
                sys.exit(1)
                
            # Config from header
            maxT = header_ints[2]
            V = header_ints[3]
            L = header_ints[4]
            NH = header_ints[5]
            C = header_ints[6]
            Vp = header_ints[7]
            
            print("[GPT-2]")
            print(f"max_seq_len: {maxT}")
            print(f"vocab_size: {V}")
            print(f"padded_vocab_size: {Vp}")
            print(f"num_layers: {L}")
            print(f"num_heads: {NH}")
            print(f"channels: {C}")
            
            # Re-verify config matches
            assert L == self.config.n_layer
            assert C == self.config.n_embd
            assert NH == self.config.n_head

            # Resize if needed
            if Vp != self.config.padded_vocab_size:
                print(f"Resize vocab_size from {self.config.padded_vocab_size} to {Vp}")
                self.config.padded_vocab_size = Vp
                self.config.vocab_size = V
                self.wte = nn.Embedding(Vp, C).to(self.dev0)
                self.lm_head = nn.Linear(C, Vp, bias=False).to(self.dev1)

            def read_tensor(shape, name):
                numel = int(np.prod(shape))
                bytes_to_read = numel * 4
                raw = f.read(bytes_to_read)
                arr = np.frombuffer(raw, dtype=np.float32)
                t = torch.from_numpy(arr.copy()).view(shape)
                # Debug print for weights
                if "weight" in name or "w" in name: 
                     print(f"Loaded {name} {tuple(shape)} norm: {t.norm().item():.4f}")
                return t

            # 1. wte
            wte_data = read_tensor((Vp, C), "wte")
            with torch.no_grad():
                self.wte.weight.copy_(wte_data.to(self.dev0))
                self.lm_head.weight.copy_(wte_data.to(self.dev1))

            # 2. wpe
            wpe_data = read_tensor((maxT, C), "wpe")
            with torch.no_grad():
                self.wpe.weight.copy_(wpe_data.to(self.dev0))

            # Read all big tensors first
            ln1w = read_tensor((L, C), "ln1w")
            ln1b = read_tensor((L, C), "ln1b")
            qkvw = read_tensor((L, 3*C, C), "qkvw")
            qkvb = read_tensor((L, 3*C), "qkvb")
            attprojw = read_tensor((L, C, C), "attprojw")
            attprojb = read_tensor((L, C), "attprojb")
            ln2w = read_tensor((L, C), "ln2w")
            ln2b = read_tensor((L, C), "ln2b")
            fcw  = read_tensor((L, 4*C, C), "fcw")
            fcb  = read_tensor((L, 4*C), "fcb")
            fcprojw = read_tensor((L, C, 4*C), "fcprojw")
            fcprojb = read_tensor((L, C), "fcprojb")
            
            # Assign to layers
            for i in range(L):
                dev = self.dev0 if i < self.split_layer else self.dev1
                blocks = self.blocks_part1 if i < self.split_layer else self.blocks_part2
                idx = i if i < self.split_layer else i - self.split_layer
                
                with torch.no_grad():
                     blocks[idx].ln_1.weight.copy_(ln1w[i].to(dev))
                     blocks[idx].ln_1.bias.copy_(ln1b[i].to(dev))

                     blocks[idx].attn.c_attn.weight.copy_(qkvw[i].to(dev))
                     blocks[idx].attn.c_attn.bias.copy_(qkvb[i].to(dev))
                     blocks[idx].attn.c_proj.weight.copy_(attprojw[i].to(dev))
                     blocks[idx].attn.c_proj.bias.copy_(attprojb[i].to(dev))
                     
                     blocks[idx].ln_2.weight.copy_(ln2w[i].to(dev))
                     blocks[idx].ln_2.bias.copy_(ln2b[i].to(dev))
                     
                     blocks[idx].mlp.c_fc.weight.copy_(fcw[i].to(dev))
                     blocks[idx].mlp.c_fc.bias.copy_(fcb[i].to(dev))
                     blocks[idx].mlp.c_proj.weight.copy_(fcprojw[i].to(dev))
                     blocks[idx].mlp.c_proj.bias.copy_(fcprojb[i].to(dev))

            # lnf
            lnfw = read_tensor((C,), "lnfw")
            lnfb = read_tensor((C,), "lnfb")
            with torch.no_grad():
                self.ln_f.weight.copy_(lnfw.to(self.dev1))
                self.ln_f.bias.copy_(lnfb.to(self.dev1))
            
            total_params = Vp*C + maxT*C + 2*L*C + L*3*C*C + L*3*C + L*C*C + L*C + 2*L*C + L*4*C*C + L*4*C + L*C*4*C + L*C + 2*C
            print(f"num_parameters: {total_params}")

    def forward(self, idx, targets=None):
        # --- Stage 1 (Dev 0) ---
        idx = idx.to(self.dev0)
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=self.dev0)

        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks_part1:
            x = block(x)

        # Move to Stage 2
        x = x.to(self.dev1)

        for block in self.blocks_part2:
            x = block(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            targets = targets.to(self.dev1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------

class DataLoader:
    def __init__(self, filename_pattern, B, T, process_rank=0, num_processes=1, shuffle=False):
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            print(f"Error: no files found matching the pattern: {filename_pattern}")
            sys.exit(1)
            
        self.shuffle = shuffle
        if self.shuffle:
            # Replicate C shuffling: shards first
            rng = torch.Generator()
            rng.manual_seed(42 + process_rank)
            perm = torch.randperm(len(self.files), generator=rng).tolist()
            self.files = [self.files[i] for i in perm]
            
        self.current_shard_idx = 0
        self.current_sample_idx = 0
        self.load_shard(0, process_rank)
        
    def load_shard(self, idx, seed_offset):
        filename = self.files[idx]
        with open(filename, "rb") as f:
            header = f.read(1024)
            header_ints = struct.unpack('256i', header)
            magic = header_ints[0]
            if magic != 20240520:
                print("Bad magic in the data file")
                sys.exit(1)
            self.ntok = header_ints[2]
            
            raw_data = f.read()
            self.tokens = np.frombuffer(raw_data, dtype=np.uint16)
            
        self.shard_num_samples = (len(self.tokens) - 1) // (self.B * self.T)
        
        if self.shuffle:
            # Intra-shard shuffle
            rng = torch.Generator()
            rng.manual_seed(42 + seed_offset + idx) 
            self.indices = torch.randperm(self.shard_num_samples, generator=rng).tolist()
        else:
            self.indices = list(range(self.shard_num_samples))
            
    def next_batch(self):
        if self.current_sample_idx >= len(self.indices):
             self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
             self.load_shard(self.current_shard_idx, 0)
             self.current_sample_idx = 0
             
        idx = self.indices[self.current_sample_idx]
        offset = idx * self.B * self.T
        
        chunk = self.tokens[offset : offset + self.B * self.T + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1]).view(self.B, self.T)
        y = torch.from_numpy(chunk[1:]).view(self.B, self.T)
        
        self.current_sample_idx += 1
        return x, y

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='dev/data/tinyshakespeare/tiny_shakespeare_train.bin')
    parser.add_argument('-j', default='dev/data/tinyshakespeare/tiny_shakespeare_val.bin')
    parser.add_argument('-o', default=None)
    parser.add_argument('-b', type=int, default=4)
    parser.add_argument('-t', type=int, default=1024)
    parser.add_argument('-l', type=float, default=3e-4) # 3e-4f
    parser.add_argument('-v', type=int, default=20)
    parser.add_argument('-m', type=int, default=20)
    parser.add_argument('-s', type=int, default=20)
    parser.add_argument('-g', type=int, default=64)
    args = parser.parse_args()

    # Print Params Table
    print("+-----------------------+----------------------------------------------------+")
    print("| Parameter             | Value                                              |")
    print("+-----------------------+----------------------------------------------------+")
    print(f"| train data pattern    | {args.i:<50} |")
    print(f"| val data pattern      | {args.j:<50} |")
    print(f"| output log file       | {str(args.o) if args.o else 'NULL':<50} |")
    print(f"| batch size B          | {args.b:<50} |")
    print(f"| sequence length T     | {args.t:<50} |")
    print(f"| learning rate         | {args.l:<50.6f} |")
    print(f"| val_loss_every        | {args.v:<50} |")
    print(f"| val_max_steps         | {args.m:<50} |")
    print(f"| sample_every          | {args.s:<50} |")
    print(f"| genT                  | {args.g:<50} |")
    print("+-----------------------+----------------------------------------------------+")

    # Device info
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"| device                | {dev_name:<50} |")
    print(f"| TF32                  | {'enabled' if torch.backends.cuda.matmul.allow_tf32 else 'disabled':<50} |")
    print("+-----------------------+----------------------------------------------------+")

    # Model
    config = GPT2Config(block_size=args.t)
    model = PipelineGPT2(config, checkpoint_path="gpt2_124M.bin")
    print("+-----------------------+----------------------------------------------------+")

    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # DataLoaders
    train_loader = DataLoader(args.i, args.b, args.t, shuffle=True)
    val_loader = DataLoader(args.j, args.b, args.t, shuffle=False)
    
    train_num_batches = len(train_loader.tokens) // (args.b * args.t) # Approx
    val_num_batches = len(val_loader.tokens) // (args.b * args.t)
    if val_num_batches > args.m: val_num_batches = args.m
    
    print(f"| train_num_batches     | {train_num_batches:<50} |")
    print(f"| val_num_batches       | {val_num_batches:<50} |")
    print("+-----------------------+----------------------------------------------------+")
    
    print(f"allocated {int(round(33422596 * 4 / (1024*1024)))} MiB for model parameters") 

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.l, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    
    total_sum_time = 0.0
    
    model.train()
    
    # Loop
    for step in range(train_num_batches + 1):
        last_step = (step == train_num_batches)
        
        # Validation
        if step % args.v == 0 or last_step:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                val_loader.current_shard_idx = 0
                val_loader.current_sample_idx = 0
                val_loader.load_shard(0, 0)
                
                for i in range(val_num_batches):
                    vx, vy = val_loader.next_batch()
                    _, vloss = model(vx, vy)
                    val_loss += vloss.item()
            val_loss /= val_num_batches
            print(f"val loss {val_loss:.6f}")
            model.train()
            
        # Sampling
        if (step > 0 and step % args.s == 0) or last_step:
             model.eval()
             print("generating:\n---")
             with torch.no_grad():
                 ctx = torch.tensor([[50256]], dtype=torch.long, device=model.dev0)
                 for t_gen in range(1, args.g):
                     logits, _ = model(ctx)
                     next_logits = logits[0, -1, :]
                     probs = F.softmax(next_logits, dim=-1)
                     # Random sample
                     idx = torch.multinomial(probs, 1).item()
                     # Decode
                     try:
                        print(enc.decode([idx]), end="", flush=True)
                     except:
                        print(f"{idx} ", end="", flush=True)
                        
                     ctx = torch.cat((ctx, torch.tensor([[idx]], device=model.dev0)), dim=1)
             print("\n---\n")
             model.train()
             
        if last_step:
            break
            
        # Training Step
        t0 = time.time()
        
        # Constant LR matches C implementation
        for pg in optimizer.param_groups:
            pg['lr'] = args.l 

        x, y = train_loader.next_batch()
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        total_sum_time += dt
        
        print(f"step {step+1:4d}/{train_num_batches}: train loss {loss.item():.6f} ({dt*1000:.6f} ms, {int(args.b * args.t / dt)} tok/s)")

    print(f"total average iteration time: {total_sum_time/train_num_batches*1000:.6f} ms")

if __name__ == '__main__':
    set_seed(1337)
    main()
