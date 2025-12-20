#!/usr/bin/env python3
"""
Direct comparison plots for NCCL vs NVSHMEM at batch_size=16.
Shows both throughput and latency across different microbatch sizes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

def load_nccl_data():
    """Load and process NCCL microbatching data."""
    df = pd.read_csv('NCCL_Microbatching.csv', skiprows=1)
    df.columns = ['batch_size', 'microbatch_size', 'tokens_per_second', 'avg_iteration_time_ms']
    df = df.dropna()
    return df

def load_nvshmem_data():
    """Load and process NVSHMEM microbatching data."""
    with open('NVSHMEM_Microbatching.csv', 'r') as f:
        lines = f.readlines()
    
    batch_sizes = [2, 4, 8, 16, 32]
    microbatch_sizes = [1, 2, 4, 8, 16, 32]
    
    data = []
    
    for i, mb_size in enumerate(microbatch_sizes):
        row_idx = i + 2
        if row_idx >= len(lines):
            break
        parts = lines[row_idx].strip().split(',')
        
        response_times = parts[1:6]
        if len(parts) > 8:
            throughputs = parts[9:14]
        else:
            continue
            
        for j, batch_size in enumerate(batch_sizes):
            try:
                if j < len(throughputs) and throughputs[j].strip():
                    tok_sec = float(throughputs[j])
                    if j < len(response_times) and response_times[j].strip():
                        resp_time = float(response_times[j])
                    else:
                        resp_time = np.nan
                    
                    data.append({
                        'batch_size': batch_size,
                        'microbatch_size': mb_size,
                        'tokens_per_second': tok_sec,
                        'avg_iteration_time_ms': resp_time
                    })
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(data)
    return df

def create_batch16_comparison():
    """
    Create side-by-side comparison plots for batch_size=16.
    Left: Throughput, Right: Latency
    """
    # Load data
    nccl_df = load_nccl_data()
    nvshmem_df = load_nvshmem_data()
    
    # Filter for batch_size=16
    nccl_b16 = nccl_df[nccl_df['batch_size'] == 16].sort_values('microbatch_size')
    nvshmem_b16 = nvshmem_df[nvshmem_df['batch_size'] == 16].sort_values('microbatch_size')
    
    print(f"NCCL batch=16 data points: {len(nccl_b16)}")
    print(f"NVSHMEM batch=16 data points: {len(nvshmem_b16)}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors for NCCL and NVSHMEM
    nccl_color = '#E74C3C'  # Red
    nvshmem_color = '#3498DB'  # Blue
    
    # === LEFT PLOT: Throughput ===
    ax1.plot(nccl_b16['microbatch_size'], nccl_b16['tokens_per_second'],
            marker='o', linewidth=3, markersize=10, label='NCCL',
            color=nccl_color, linestyle='-', markeredgecolor='white', markeredgewidth=1.5)
    
    ax1.plot(nvshmem_b16['microbatch_size'], nvshmem_b16['tokens_per_second'],
            marker='s', linewidth=3, markersize=10, label='NVSHMEM',
            color=nvshmem_color, linestyle='-', markeredgecolor='white', markeredgewidth=1.5)
    
    ax1.set_xlabel('Microbatch Size', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Throughput (Tokens/sec)', fontweight='bold', fontsize=12)
    ax1.set_title('Throughput Comparison (Batch Size = 16)', fontweight='bold', fontsize=13, pad=15)
    ax1.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_xscale('log', base=2)
    
    # Add value annotations for key points
    for idx, row in nccl_b16.iterrows():
        if row['microbatch_size'] in [1, 16]:  # Annotate extremes
            ax1.annotate(f"{row['tokens_per_second']:.0f}",
                        xy=(row['microbatch_size'], row['tokens_per_second']),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8, color=nccl_color, fontweight='bold')
    
    for idx, row in nvshmem_b16.iterrows():
        if row['microbatch_size'] in [1, 16]:
            ax1.annotate(f"{row['tokens_per_second']:.0f}",
                        xy=(row['microbatch_size'], row['tokens_per_second']),
                        xytext=(0, -12), textcoords='offset points',
                        ha='center', fontsize=8, color=nvshmem_color, fontweight='bold')
    
    # === RIGHT PLOT: Latency ===
    # Filter out NaN values for latency
    nccl_b16_latency = nccl_b16.dropna(subset=['avg_iteration_time_ms'])
    nvshmem_b16_latency = nvshmem_b16.dropna(subset=['avg_iteration_time_ms'])
    
    ax2.plot(nccl_b16_latency['microbatch_size'], nccl_b16_latency['avg_iteration_time_ms'],
            marker='o', linewidth=3, markersize=10, label='NCCL',
            color=nccl_color, linestyle='-', markeredgecolor='white', markeredgewidth=1.5)
    
    ax2.plot(nvshmem_b16_latency['microbatch_size'], nvshmem_b16_latency['avg_iteration_time_ms'],
            marker='s', linewidth=3, markersize=10, label='NVSHMEM',
            color=nvshmem_color, linestyle='-', markeredgecolor='white', markeredgewidth=1.5)
    
    ax2.set_xlabel('Microbatch Size', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
    ax2.set_title('Latency Comparison (Batch Size = 16)', fontweight='bold', fontsize=13, pad=15)
    ax2.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_xscale('log', base=2)
    
    # Add value annotations for key points
    for idx, row in nccl_b16_latency.iterrows():
        if row['microbatch_size'] in [1, 16]:
            ax2.annotate(f"{row['avg_iteration_time_ms']:.0f}",
                        xy=(row['microbatch_size'], row['avg_iteration_time_ms']),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8, color=nccl_color, fontweight='bold')
    
    for idx, row in nvshmem_b16_latency.iterrows():
        if row['microbatch_size'] in [1, 16]:
            ax2.annotate(f"{row['avg_iteration_time_ms']:.0f}",
                        xy=(row['microbatch_size'], row['avg_iteration_time_ms']),
                        xytext=(0, -12), textcoords='offset points',
                        ha='center', fontsize=8, color=nvshmem_color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plot_batch16_nccl_vs_nvshmem.png', bbox_inches='tight', dpi=300)
    print("\n✓ Created plot_batch16_nccl_vs_nvshmem.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY: Batch Size = 16 Comparison")
    print("="*60)
    
    print("\nThroughput (Tokens/sec):")
    print(f"  NCCL    - Min: {nccl_b16['tokens_per_second'].min():.0f}, Max: {nccl_b16['tokens_per_second'].max():.0f}")
    print(f"  NVSHMEM - Min: {nvshmem_b16['tokens_per_second'].min():.0f}, Max: {nvshmem_b16['tokens_per_second'].max():.0f}")
    print(f"  NVSHMEM Advantage: {((nvshmem_b16['tokens_per_second'].mean() / nccl_b16['tokens_per_second'].mean() - 1) * 100):.1f}% average")
    
    if len(nccl_b16_latency) > 0 and len(nvshmem_b16_latency) > 0:
        print("\nLatency (ms):")
        print(f"  NCCL    - Min: {nccl_b16_latency['avg_iteration_time_ms'].min():.0f}, Max: {nccl_b16_latency['avg_iteration_time_ms'].max():.0f}")
        print(f"  NVSHMEM - Min: {nvshmem_b16_latency['avg_iteration_time_ms'].min():.0f}, Max: {nvshmem_b16_latency['avg_iteration_time_ms'].max():.0f}")
    
    print("="*60)
    
    plt.close()

if __name__ == "__main__":
    print("Generating NCCL vs NVSHMEM comparison plot for Batch Size = 16...")
    print()
    create_batch16_comparison()
    print("\nPlot generation complete!")
