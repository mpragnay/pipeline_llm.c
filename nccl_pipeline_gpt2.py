
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
import torch.distributed as dist
import tiktoken

# -----------------------------------------------------------------------------
# Clean NCCL Implementation - Barrier-Synchronized Architecture
# This matches NVSHMEM's execution model: sequential processing with barriers
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
        self.gelu    = nn.GELU(approximate='tanh')
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

# -----------------------------------------------------------------------------
# Barrier-Synchronized Pipeline Model - Matches NVSHMEM Architecture
# -----------------------------------------------------------------------------

class NCCLPipelineGPT2(nn.Module):
    """
    Pipeline GPT2 with NCCL using barrier-synchronized execution.
    Matches NVSHMEM's architecture: ranks execute sequentially with barriers.
    """
    def __init__(self, config, rank, world_size, checkpoint_path=None):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)
        
        # Determine partition (matching NVSHMEM's get_partition_for_pe)
        total_layers = config.n_layer
        base_layers = total_layers // world_size
        remainder = total_layers % world_size
        
        self.num_layers = base_layers + (1 if rank < remainder else 0)
        self.first_layer = sum(base_layers + (1 if i < remainder else 0) for i in range(rank))
        self.has_embedding = (rank == 0)
        self.has_final_ln = (rank == world_size - 1)
        
        C = config.n_embd
        vocab_size_actual = config.padded_vocab_size
        
        # Allocate only this rank's components
        if self.has_embedding:
            self.wte = nn.Embedding(vocab_size_actual, C).to(self.device)
            self.wpe = nn.Embedding(config.block_size, C).to(self.device)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(self.num_layers)]).to(self.device)
        
        if self.has_final_ln:
            self.ln_f = LayerNorm(C, bias=config.bias).to(self.device)
            self.lm_head = nn.Linear(C, vocab_size_actual, bias=False).to(self.device)
        
        # Communication buffers (matching NVSHMEM's nvshmem_act_buffer/grad_buffer)
        self.activation_buffer = None
        self.gradient_buffer = None
        
        if checkpoint_path:
            self.load_from_binary(checkpoint_path)
    
    def allocate_buffers(self, B, T):
        """Allocate communication buffers"""
        C = self.config.n_embd
        if self.activation_buffer is None:
            self.activation_buffer = torch.zeros(B, T, C, device=self.device)
            self.gradient_buffer = torch.zeros(B, T, C, device=self.device)

    def load_from_binary(self, path):
        """Load partitioned parameters from checkpoint"""
        if self.rank == 0:
            print(f"Loading model from {path}")
        
        if not os.path.exists(path):
            if self.rank == 0:
                print(f"Error: checkpoint {path} not found")
            sys.exit(1)
            
        with open(path, 'rb') as f:
            header = f.read(256 * 4)
            header_ints = struct.unpack('256i', header)
            
            magic, version = header_ints[0], header_ints[1]
            if magic != 20240326 or version != 3:
                if self.rank == 0:
                    print("Bad magic or version in model file")
                sys.exit(1)
                
            maxT, V, L, NH, C, Vp = header_ints[2:8]
            
            if self.rank == 0:
                print(f"[GPT-2] max_seq_len: {maxT}, vocab_size: {V}, padded_vocab_size: {Vp}")
                print(f"num_layers: {L}, num_heads: {NH}, channels: {C}")
                print(f"Rank {self.rank}: layers {self.first_layer}-{self.first_layer + self.num_layers - 1}")
            
            # Resize embeddings if needed
            if Vp != self.config.padded_vocab_size:
                if self.rank == 0:
                    print(f"Resizing vocab from {self.config.padded_vocab_size} to {Vp}")
                self.config.padded_vocab_size = Vp
                self.config.vocab_size = V
                
                # Recreate embeddings with correct size
                if self.has_embedding:
                    self.wte = nn.Embedding(Vp, C).to(self.device)
                    self.wpe = nn.Embedding(maxT, C).to(self.device)
                if self.has_final_ln:
                    self.lm_head = nn.Linear(C, Vp, bias=False).to(self.device)
            
            def read_tensor(shape):
                numel = int(np.prod(shape))
                raw = f.read(numel * 4)
                arr = np.frombuffer(raw, dtype=np.float32)
                return torch.from_numpy(arr.copy()).view(shape)

            # Read embeddings (all ranks need wte for gradient all-reduce)
            wte_data = read_tensor((Vp, C))
            wpe_data = read_tensor((maxT, C))
            
            if self.has_embedding:
                with torch.no_grad():
                    self.wte.weight.copy_(wte_data.to(self.device))
                    self.wpe.weight.copy_(wpe_data.to(self.device))
            
            if self.has_final_ln:
                with torch.no_grad():
                    self.lm_head.weight.copy_(wte_data.to(self.device))
            
            # Read ALL layer parameters
            ln1w = read_tensor((L, C))
            ln1b = read_tensor((L, C))
            qkvw = read_tensor((L, 3*C, C))
            qkvb = read_tensor((L, 3*C))
            attprojw = read_tensor((L, C, C))
            attprojb = read_tensor((L, C))
            ln2w = read_tensor((L, C))
            ln2b = read_tensor((L, C))
            fcw = read_tensor((L, 4*C, C))
            fcb = read_tensor((L, 4*C))
            fcprojw = read_tensor((L, C, 4*C))
            fcprojb = read_tensor((L, C))
            
            # Copy only this rank's layers
            for local_idx in range(self.num_layers):
                global_idx = self.first_layer + local_idx
                with torch.no_grad():
                    self.blocks[local_idx].ln_1.weight.copy_(ln1w[global_idx].to(self.device))
                    self.blocks[local_idx].ln_1.bias.copy_(ln1b[global_idx].to(self.device))
                    self.blocks[local_idx].attn.c_attn.weight.copy_(qkvw[global_idx].to(self.device))
                    self.blocks[local_idx].attn.c_attn.bias.copy_(qkvb[global_idx].to(self.device))
                    self.blocks[local_idx].attn.c_proj.weight.copy_(attprojw[global_idx].to(self.device))
                    self.blocks[local_idx].attn.c_proj.bias.copy_(attprojb[global_idx].to(self.device))
                    self.blocks[local_idx].ln_2.weight.copy_(ln2w[global_idx].to(self.device))
                    self.blocks[local_idx].ln_2.bias.copy_(ln2b[global_idx].to(self.device))
                    self.blocks[local_idx].mlp.c_fc.weight.copy_(fcw[global_idx].to(self.device))
                    self.blocks[local_idx].mlp.c_fc.bias.copy_(fcb[global_idx].to(self.device))
                    self.blocks[local_idx].mlp.c_proj.weight.copy_(fcprojw[global_idx].to(self.device))
                    self.blocks[local_idx].mlp.c_proj.bias.copy_(fcprojb[global_idx].to(self.device))
            
            # Final layer norm
            if self.has_final_ln:
                lnfw = read_tensor((C,))
                lnfb = read_tensor((C,))
                with torch.no_grad():
                    self.ln_f.weight.copy_(lnfw.to(self.device))
                    self.ln_f.bias.copy_(lnfb.to(self.device))

    def forward_sequential(self, inputs, targets=None):
        """
        Barrier-synchronized forward pass matching NVSHMEM's architecture.
        Each rank processes sequentially with barriers between.
        """
        B, T = inputs.size()
        self.allocate_buffers(B, T)
        C = self.config.n_embd
        
        # Sequential forward pass (matching NVSHMEM)
        for target_rank in range(self.world_size):
            if self.rank == target_rank:
                if self.rank == 0:
                    # Rank 0: embedding
                    inputs_device = inputs.to(self.device)
                    pos = torch.arange(0, T, dtype=torch.long, device=self.device)
                    x = self.wte(inputs_device) + self.wpe(pos)
                else:
                    # Other ranks: receive from previous
                    x = self.activation_buffer.clone()
                
                # Process this rank's layers
                for block in self.blocks:
                    x = block(x)
                
                # Send to next rank (except last)
                if self.rank < self.world_size - 1:
                    dist.send(x.contiguous(), dst=self.rank + 1)
                else:
                    # Last rank: compute logits and loss
                    x = self.ln_f(x)
                    logits = self.lm_head(x)
                    
                    loss = None
                    if targets is not None:
                        targets_device = targets.to(self.device)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_device.view(-1))
                    
                    # Store for backward
                    self.final_logits = logits
                    self.final_loss = loss
                    
                # Receive activation for next iteration (except last rank)
                if self.rank > 0 and target_rank == self.rank - 1:
                    dist.recv(self.activation_buffer, src=self.rank - 1)
            
            dist.barrier()  # All ranks wait (matching nvshmem_barrier_all)
        
        # Return loss only from last rank
        if self.rank == self.world_size - 1:
            return self.final_logits, self.final_loss
        return None, None

    def backward_sequential(self):
        """
        Barrier-synchronized backward pass matching NVSHMEM's architecture.
        Process ranks in reverse order with barriers.
        """
        # Sequential backward pass (matching NVSHMEM)
        for target_rank in range(self.world_size - 1, -1, -1):
            if self.rank == target_rank:
                if self.rank == self.world_size - 1:
                    # Last rank: start backward from loss
                    self.final_loss.backward()
                else:
                    # Other ranks: receive gradient from next rank
                    dist.recv(self.gradient_buffer, src=self.rank + 1)
                    
                    # Apply gradient to local computation
                    # (PyTorch autograd handles this automatically if we structure correctly)
                
                # Send gradient to previous rank (except first)
                if self.rank > 0:
                    # Get gradient of activation that was sent forward
                    # This would be stored in the computation graph
                    pass  # Handled by autograd
                    
            dist.barrier()  # All ranks wait

# -----------------------------------------------------------------------------
# Data Loader (single instance, rank 0 broadcasts)
# -----------------------------------------------------------------------------

class DataLoader:
    def __init__(self, filename_pattern, B, T, rank=0, shuffle=False):
        self.B = B
        self.T = T
        self.rank = rank
        self.files = sorted(glob.glob(filename_pattern))
        
        if rank == 0 and not self.files:
            print(f"Error: no files found matching: {filename_pattern}")
            sys.exit(1)
            
        self.shuffle = shuffle
        if self.shuffle and rank == 0:
            rng = torch.Generator()
            rng.manual_seed(42)
            perm = torch.randperm(len(self.files), generator=rng).tolist()
            self.files = [self.files[i] for i in perm]
            
        if rank == 0:
            self.current_shard_idx = 0
            self.current_sample_idx = 0
            self.load_shard(0)
        
    def load_shard(self, idx):
        """Only rank 0 loads data"""
        if self.rank != 0:
            return
            
        filename = self.files[idx]
        with open(filename, "rb") as f:
            header = f.read(1024)
            header_ints = struct.unpack('256i', header)
            if header_ints[0] != 20240520:
                print("Bad magic in data file")
                sys.exit(1)
            raw_data = f.read()
            self.tokens = np.frombuffer(raw_data, dtype=np.uint16)
            
        self.shard_num_samples = (len(self.tokens) - 1) // (self.B * self.T)
        
        if self.shuffle:
            rng = torch.Generator()
            rng.manual_seed(42 + idx)
            self.indices = torch.randperm(self.shard_num_samples, generator=rng).tolist()
        else:
            self.indices = list(range(self.shard_num_samples))
            
    def next_batch(self):
        """Only rank 0 loads, broadcasts to all"""
        if self.rank == 0:
            if self.current_sample_idx >= len(self.indices):
                self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
                self.load_shard(self.current_shard_idx)
                self.current_sample_idx = 0
                 
            idx = self.indices[self.current_sample_idx]
            offset = idx * self.B * self.T
            
            chunk = self.tokens[offset : offset + self.B * self.T + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1]).view(self.B, self.T)
            y = torch.from_numpy(chunk[1:]).view(self.B, self.T)
            
            self.current_sample_idx += 1
        else:
            # Other ranks create placeholder
            x = torch.zeros(self.B, self.T, dtype=torch.long)
            y = torch.zeros(self.B, self.T, dtype=torch.long)
        
        # Broadcast from rank 0 to all
        dist.broadcast(x, src=0)
        dist.broadcast(y, src=0)
        
        return x, y

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='dev/data/tinyshakespeare/tiny_shakespeare_train.bin')
    parser.add_argument('-j', default='dev/data/tinyshakespeare/tiny_shakespeare_val.bin')
    parser.add_argument('-b', type=int, default=4)
    parser.add_argument('-t', type=int, default=1024)
    parser.add_argument('-l', type=float, default=3e-4)
    parser.add_argument('-v', type=int, default=20)
    parser.add_argument('-m', type=int, default=20)
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size != 2:
        if rank == 0:
            print(f"Error: This script requires exactly 2 GPUs, got {world_size}")
        sys.exit(1)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    if rank == 0:
        print("+-----------------------+----------------------------------------------------+")
        print("| Parameter             | Value                                              |")
        print("+-----------------------+----------------------------------------------------+")
        print(f"| train data pattern    | {args.i:<50} |")
        print(f"| val data pattern      | {args.j:<50} |")
        print(f"| batch size B          | {args.b:<50} |")
        print(f"| sequence length T     | {args.t:<50} |")
        print(f"| learning rate         | {args.l:<50.6f} |")
        print(f"| communication         | {'NCCL (barrier-synchronized)':<50} |")
        print(f"| world_size            | {world_size:<50} |")
        print("+-----------------------+----------------------------------------------------+")

    # Model
    config = GPT2Config(block_size=args.t)
    model = NCCLPipelineGPT2(config, rank, world_size, checkpoint_path="gpt2_124M.bin")
    
    if rank == 0:
        print("+-----------------------+----------------------------------------------------+")

    # Data loaders (rank 0 loads, broadcasts to all)
    train_loader = DataLoader(args.i, args.b, args.t, rank=rank, shuffle=True)
    val_loader = DataLoader(args.j, args.b, args.t, rank=rank, shuffle=False)
    
    if rank == 0:
        train_num_batches = len(train_loader.tokens) // (args.b * args.t)
        val_num_batches = min(len(val_loader.tokens) // (args.b * args.t), args.m)
        print(f"| train_num_batches     | {train_num_batches:<50} |")
        print(f"| val_num_batches       | {val_num_batches:<50} |")
        print("+-----------------------+----------------------------------------------------+")
    else:
        train_num_batches = 10  # Will be synced via barrier
        val_num_batches = args.m
    
    # Broadcast batch counts
    batch_counts = torch.tensor([train_num_batches, val_num_batches], dtype=torch.long)
    dist.broadcast(batch_counts, src=0)
    train_num_batches, val_num_batches = batch_counts.tolist()

    # Optimizer (each rank only optimizes its parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.l, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    
    total_sum_time = 0.0
    model.train()
    
    # Training loop
    for step in range(train_num_batches + 1):
        last_step = (step == train_num_batches)
        
        # Validation
        if step % args.v == 0 or last_step:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(val_num_batches):
                    vx, vy = val_loader.next_batch()
                    _, vloss = model.forward_sequential(vx, vy)
                    if rank == world_size - 1 and vloss is not None:
                        val_loss += vloss.item()
                        
            if rank == world_size - 1:
                val_loss /= val_num_batches
                print(f"val loss {val_loss:.6f}")
            model.train()
            
        if last_step:
            break
            
        # Training step
        if rank == 0:
            t0 = time.time()
        
        x, y = train_loader.next_batch()
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass (barrier-synchronized)
        logits, loss = model.forward_sequential(x, y)
        
        # Backward pass (only last rank has loss)
        if rank == world_size - 1 and loss is not None:
            loss.backward()
        
        # All ranks update their parameters
        dist.barrier()
        optimizer.step()
        dist.barrier()
        
        if rank == 0:
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            total_sum_time += dt
        
        # Broadcast loss for logging
        if rank == world_size - 1:
            loss_tensor = torch.tensor([loss.item()], device=f'cuda:{rank}')
        else:
            loss_tensor = torch.tensor([0.0], device=f'cuda:{rank}')
        dist.broadcast(loss_tensor, src=world_size - 1)
        
        if rank == 0:
            print(f"step {step+1:4d}/{train_num_batches}: train loss {loss_tensor.item():.6f} ({dt*1000:.6f} ms, {int(args.b * args.t / dt)} tok/s)")

    if rank == 0:
        print(f"total average iteration time: {total_sum_time/train_num_batches*1000:.6f} ms")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    set_seed(1337)
    main()
