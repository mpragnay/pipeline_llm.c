
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

# -----------------------------------------------------------------------------
# Custom Autograd Functions for NCCL Send/Recv with Gradient Flow
# -----------------------------------------------------------------------------

class NCCLSendFunction(torch.autograd.Function):
    """Send activation forward, receive gradient backward"""
    @staticmethod
    def forward(ctx, tensor, dst):
        ctx.dst = dst
        dist.send(tensor.contiguous(), dst=dst)
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        # Receive gradient from destination
        grad_input = torch.zeros_like(grad_output)
        dist.recv(grad_input, src=ctx.dst)
        return grad_input, None

class NCCLRecvFunction(torch.autograd.Function):
    """Receive activation forward, send gradient backward"""
    @staticmethod
    def forward(ctx, buffer, src):
        ctx.src = src
        ctx.shape = buffer.shape
        dist.recv(buffer, src=src)
        return buffer
    
    @staticmethod
    def backward(ctx, grad_output):
        # Send gradient back to source
        dist.send(grad_output.contiguous(), dst=ctx.src)
        return None, None

def nccl_send(tensor, dst):
    """Send tensor with gradient support"""
    return NCCLSendFunction.apply(tensor, dst)

def nccl_recv(buffer, src):
    """Receive tensor with gradient support"""
    return NCCLRecvFunction.apply(buffer, src)

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
        """Allocate communication buffers, reallocating if size changes"""
        C = self.config.n_embd
        # Reallocate if buffer doesn't exist or size changed
        if self.activation_buffer is None or self.activation_buffer.shape != (B, T, C):
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

    def forward_sequential(self, inputs, targets=None, verbose=False):
        """
        Barrier-synchronized forward pass matching NVSHMEM's architecture.
        Each rank processes sequentially with barriers between.
        """
        if verbose:
            print(f"[Rank {self.rank}] forward_sequential: START")
        B, T = inputs.size()
        self.allocate_buffers(B, T)
        C = self.config.n_embd
        
        # Process layers in sequence: rank 0, then rank 1
        # Rank 0 processes first
        if self.rank == 0:
            if verbose:
                print(f"[Rank {self.rank}] Computing embedding")
            inputs_device = inputs.to(self.device)
            pos = torch.arange(0, T, dtype=torch.long, device=self.device)
            x = self.wte(inputs_device) + self.wpe(pos)
            if verbose:
                print(f"[Rank {self.rank}] Embedding computed")
            
            if verbose:
                print(f"[Rank {self.rank}] Processing {len(self.blocks)} layers")
            for i, block in enumerate(self.blocks):
                x = block(x)
                if verbose and i % 2 == 0:
                    print(f"[Rank {self.rank}] Processed layer {i}")
            if verbose:
                print(f"[Rank {self.rank}] All layers processed")
            
            if verbose:
                print(f"[Rank {self.rank}] Sending to rank 1")
            x = nccl_send(x, dst=1)
            if verbose:
                print(f"[Rank {self.rank}] Sent to rank 1")
        
        # Rank 1 receives and processes
        elif self.rank == 1:
            if verbose:
                print(f"[Rank {self.rank}] Waiting to receive from rank 0")
            x = nccl_recv(self.activation_buffer, src=0)
            if verbose:
                print(f"[Rank {self.rank}] Received from rank 0")
            x = self.activation_buffer.clone()
            
            if verbose:
                print(f"[Rank {self.rank}] Processing {len(self.blocks)} layers")
            for i, block in enumerate(self.blocks):
                x = block(x)
                if verbose and i % 2 == 0:
                    print(f"[Rank {self.rank}] Processed layer {i}")
            if verbose:
                print(f"[Rank {self.rank}] All layers processed")
            
            # Last rank: compute logits and loss
            if verbose:
                print(f"[Rank {self.rank}] Computing logits and loss")
            x = self.ln_f(x)
            logits = self.lm_head(x)
            
            loss = None
            if targets is not None:
                targets_device = targets.to(self.device)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_device.view(-1))
                if verbose:
                    print(f"[Rank {self.rank}] Loss computed: {loss.item():.6f}")
            
            # Store for backward
            self.final_logits = logits
            self.final_loss = loss
        
        # Barrier at end to ensure both ranks finish before returning
        if verbose:
            print(f"[Rank {self.rank}] Waiting at final barrier")
        dist.barrier()
        if verbose:
            print(f"[Rank {self.rank}] forward_sequential: END")
        
        # Return loss only from last rank
        if self.rank == self.world_size - 1:
            return self.final_logits, self.final_loss
        return None, None
    
    def forward_inference(self, inputs):
        """
        Inference-only forward pass without autograd.
        Uses raw dist.send/recv instead of autograd functions.
        """
        print(f"[Rank {self.rank}] forward_inference: START", flush=True)
        B, T = inputs.size()
        print(f"[Rank {self.rank}] forward_inference: B={B}, T={T}", flush=True)
        self.allocate_buffers(B, T)
        print(f"[Rank {self.rank}] forward_inference: Buffers allocated", flush=True)
        C = self.config.n_embd
        
        # Rank 0 processes first
        if self.rank == 0:
            print(f"[Rank {self.rank}] forward_inference: Rank 0 processing", flush=True)
            inputs_device = inputs.to(self.device)
            print(f"[Rank {self.rank}] forward_inference: Inputs moved to device", flush=True)
            pos = torch.arange(0, T, dtype=torch.long, device=self.device)
            x = self.wte(inputs_device) + self.wpe(pos)
            print(f"[Rank {self.rank}] forward_inference: Embeddings computed", flush=True)
            
            print(f"[Rank {self.rank}] forward_inference: Processing {len(self.blocks)} blocks", flush=True)
            for i, block in enumerate(self.blocks):
                x = block(x)
                print(f"[Rank {self.rank}] forward_inference: Block {i} done", flush=True)
            print(f"[Rank {self.rank}] forward_inference: All blocks done", flush=True)
            
            # Send to rank 1 (no autograd tracking)
            print(f"[Rank {self.rank}] forward_inference: About to send to rank 1, x.shape={x.shape}", flush=True)
            print(f"[Rank {self.rank}] forward_inference: Calling dist.send...", flush=True)
            dist.send(x.contiguous(), dst=1)
            print(f"[Rank {self.rank}] forward_inference: dist.send RETURNED", flush=True)
        
        # Rank 1 receives and processes
        elif self.rank == 1:
            print(f"[Rank {self.rank}] forward_inference: Rank 1 about to receive", flush=True)
            print(f"[Rank {self.rank}] forward_inference: Calling dist.recv...", flush=True)
            dist.recv(self.activation_buffer, src=0)
            print(f"[Rank {self.rank}] forward_inference: dist.recv RETURNED", flush=True)
            x = self.activation_buffer.clone()
            print(f"[Rank {self.rank}] forward_inference: Cloned activation buffer", flush=True)
            
            print(f"[Rank {self.rank}] forward_inference: Processing {len(self.blocks)} blocks", flush=True)
            for i, block in enumerate(self.blocks):
                x = block(x)
                print(f"[Rank {self.rank}] forward_inference: Block {i} done", flush=True)
            print(f"[Rank {self.rank}] forward_inference: All blocks done", flush=True)
            
            # Last rank: compute logits
            print(f"[Rank {self.rank}] forward_inference: Computing logits", flush=True)
            x = self.ln_f(x)
            print(f"[Rank {self.rank}] forward_inference: ln_f done", flush=True)
            logits = self.lm_head(x)
            print(f"[Rank {self.rank}] forward_inference: lm_head done, logits.shape={logits.shape}", flush=True)
            
            # Store for return
            self.final_logits = logits
            print(f"[Rank {self.rank}] forward_inference: final_logits stored", flush=True)
        
        print(f"[Rank {self.rank}] forward_inference: About to return", flush=True)
        print(f"[Rank {self.rank}] forward_inference: END", flush=True)
        
        # Return logits only from last rank
        if self.rank == self.world_size - 1:
            print(f"[Rank {self.rank}] forward_inference: Returning logits", flush=True)
            return self.final_logits
        print(f"[Rank {self.rank}] forward_inference: Returning None", flush=True)
        return None

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
        # Get device for this rank
        device = torch.device(f'cuda:{self.rank}')
        
        if self.rank == 0:
            if self.current_sample_idx >= len(self.indices):
                self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
                self.load_shard(self.current_shard_idx)
                self.current_sample_idx = 0
                 
            idx = self.indices[self.current_sample_idx]
            offset = idx * self.B * self.T
            
            chunk = self.tokens[offset : offset + self.B * self.T + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1]).view(self.B, self.T).to(device)
            y = torch.from_numpy(chunk[1:]).view(self.B, self.T).to(device)
            
            self.current_sample_idx += 1
        else:
            # Other ranks create placeholder on GPU
            x = torch.zeros(self.B, self.T, dtype=torch.long, device=device)
            y = torch.zeros(self.B, self.T, dtype=torch.long, device=device)
        
        # Broadcast from rank 0 to all (NCCL requires GPU tensors)
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
    parser.add_argument('-s', type=int, default=20, help='sample generation every N steps')
    parser.add_argument('-g', type=int, default=64, help='generation length')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    verbose = args.verbose

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
        print(f"| val_loss_every        | {args.v:<50} |")
        print(f"| val_max_steps         | {args.m:<50} |")
        print(f"| sample_every          | {args.s:<50} |")
        print(f"| genT                  | {args.g:<50} |")
        print(f"| communication         | {'NCCL (barrier-synchronized)':<50} |")
        print(f"| world_size            | {world_size:<50} |")
        print("+-----------------------+----------------------------------------------------+")

    # Model
    config = GPT2Config(block_size=args.t)
    model = NCCLPipelineGPT2(config, rank, world_size, checkpoint_path="gpt2_124M.bin")
    
    if rank == 0:
        print("+-----------------------+----------------------------------------------------+")
    
    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")

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
    
    # Broadcast batch counts (NCCL requires GPU tensors)
    batch_counts = torch.tensor([train_num_batches, val_num_batches], dtype=torch.long, device=f'cuda:{rank}')
    dist.broadcast(batch_counts, src=0)
    train_num_batches, val_num_batches = batch_counts.cpu().tolist()

    # Optimizer (each rank only optimizes its parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.l, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    
    if verbose:
        print(f"[Rank {rank}] Starting training loop")
    total_sum_time = 0.0
    model.train()
    
    # Training loop
    for step in range(train_num_batches + 1):
        if verbose:
            print(f"[Rank {rank}] === Step {step}/{train_num_batches} ===")
        last_step = (step == train_num_batches)
        
        # Validation
        if step % args.v == 0 or last_step:
            if verbose:
                print(f"[Rank {rank}] Running validation")
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(val_num_batches):
                    if verbose:
                        print(f"[Rank {rank}] Val batch {i}/{val_num_batches}")
                    vx, vy = val_loader.next_batch()
                    _, vloss = model.forward_sequential(vx, vy, verbose=verbose)
                    if rank == world_size - 1 and vloss is not None:
                        val_loss += vloss.item()
                        
            if rank == world_size - 1:
                val_loss /= val_num_batches
                print(f"val loss {val_loss:.6f}")
            model.train()
        
        # Sampling (matching NVSHMEM's approach)
        if (step > 0 and step % args.s == 0) or last_step:
            model.eval()
            if rank == 0:
                print("generating:\n---")
            
            print(f"[Rank {rank}] Starting generation", flush=True)
            
            with torch.no_grad():
                # Initialize generation tokens with BOS token (50256)
                gen_tokens = torch.full((1, args.g), 50256, dtype=torch.long, device=f'cuda:{rank}')
                print(f"[Rank {rank}] Initialized gen_tokens", flush=True)
                
                for t in range(1, args.g):
                    print(f"\n[Rank {rank}] ========== GENERATION STEP {t}/{args.g} ==========\n", flush=True)
                    
                    # Both ranks do forward pass with current context
                    # Context is gen_tokens[:, :t]
                    ctx = gen_tokens[:, :t]
                    print(f"[Rank {rank}] Step {t}: ctx = gen_tokens[:, :{t}], shape {ctx.shape}", flush=True)
                    print(f"[Rank {rank}] Step {t}: Calling forward_inference...", flush=True)
                    logits = model.forward_inference(ctx)
                    print(f"[Rank {rank}] Step {t}: forward_inference RETURNED, logits={type(logits)}", flush=True)
                    
                    # Barrier to ensure both ranks ready before sampling/broadcast
                    print(f"[Rank {rank}] Step {t}: About to call dist.barrier()...", flush=True)
                    try:
                        dist.barrier()
                        print(f"[Rank {rank}] Step {t}: dist.barrier() RETURNED SUCCESSFULLY", flush=True)
                    except Exception as e:
                        print(f"[Rank {rank}] Step {t}: dist.barrier() EXCEPTION: {e}", flush=True)
                        raise
                    
                    # Only rank 1 (last) has logits and samples
                    if rank == world_size - 1:
                        # Get logits for last position
                        print(f"[Rank {rank}] Step {t}: Sampling from logits, shape={logits.shape}", flush=True)
                        next_logits = logits[0, -1, :]  # Shape: [vocab_size]
                        print(f"[Rank {rank}] Step {t}: next_logits extracted, shape={next_logits.shape}", flush=True)
                        
                        # Move to CPU to avoid CUDA sync issues with argmax
                        print(f"[Rank {rank}] Step {t}: Moving to CPU...", flush=True)
                        next_logits_cpu = next_logits.cpu()
                        print(f"[Rank {rank}] Step {t}: Moved to CPU", flush=True)
                        
                        # Use greedy sampling (argmax)
                        print(f"[Rank {rank}] Step {t}: Calling argmax...", flush=True)
                        next_token_cpu = torch.argmax(next_logits_cpu, dim=-1, keepdim=True)  # Shape: [1]
                        print(f"[Rank {rank}] Step {t}: argmax done", flush=True)
                        next_token = next_token_cpu.to(model.device)
                        print(f"[Rank {rank}] Step {t}: Sampled token={next_token.item()}", flush=True)
                        
                        # Update local gen_tokens
                        gen_tokens[0, t] = next_token.item()
                        print(f"[Rank {rank}] Step {t}: Updated gen_tokens[0, {t}] = {next_token.item()}", flush=True)
                    else:
                        # Other ranks prepare to receive
                        print(f"[Rank {rank}] Step {t}: Creating placeholder for broadcast", flush=True)
                        next_token = torch.zeros(1, dtype=torch.long, device=model.device)
                    
                    # Rank 1 broadcasts token to all ranks
                    print(f"[Rank {rank}] Step {t}: Preparing broadcast...", flush=True)
                    token_to_broadcast = torch.zeros(1, dtype=torch.long, device=model.device)
                    if rank == world_size - 1:
                        token_to_broadcast[0] = next_token.item()
                        print(f"[Rank {rank}] Step {t}: Set token_to_broadcast[0] = {next_token.item()}", flush=True)
                    print(f"[Rank {rank}] Step {t}: Calling dist.broadcast...", flush=True)
                    dist.broadcast(token_to_broadcast, src=world_size - 1)
                    print(f"[Rank {rank}] Step {t}: dist.broadcast RETURNED", flush=True)
                    
                    # Non-last ranks update gen_tokens
                    if rank != world_size - 1:
                        gen_tokens[0, t] = token_to_broadcast.item()
                        print(f"[Rank {rank}] Step {t}: Updated gen_tokens[{t}] = {token_to_broadcast.item()}", flush=True)
                    
                    # Rank 0 prints the token
                    if rank == 0:
                        token_id = token_to_broadcast.item()
                        try:
                            print(enc.decode([token_id]), end="", flush=True)
                        except:
                            print(f"{token_id} ", end="", flush=True)

            
            print(f"[Rank {rank}] Generation complete", flush=True)
            if rank == 0:
                print("\n---\n")
            model.train()
            
        if last_step:
            break
            
        # Training step
        if verbose:
            print(f"[Rank {rank}] Training step {step}")
        if rank == 0:
            t0 = time.time()
        
        if verbose:
            print(f"[Rank {rank}] Loading batch")
        x, y = train_loader.next_batch()
        if verbose:
            print(f"[Rank {rank}] Batch loaded, zeroing gradients")
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass (barrier-synchronized)
        if verbose:
            print(f"[Rank {rank}] Starting forward pass")
        logits, loss = model.forward_sequential(x, y, verbose=verbose)
        if verbose:
            print(f"[Rank {rank}] Forward pass complete")
        
        # Backward pass (only last rank has loss)
        if rank == world_size - 1 and loss is not None:
            if verbose:
                print(f"[Rank {rank}] Starting backward pass")
            loss.backward()
            if verbose:
                print(f"[Rank {rank}] Backward pass complete")
        
        # All ranks update their parameters
        if verbose:
            print(f"[Rank {rank}] Waiting at barrier before optimizer.step()")
        dist.barrier()
        if verbose:
            print(f"[Rank {rank}] Passed barrier, calling optimizer.step()")
        optimizer.step()
        if verbose:
            print(f"[Rank {rank}] optimizer.step() complete")
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
