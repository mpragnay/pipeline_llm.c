#!/usr/bin/env python3
"""
Visualization script for NCCL and NVSHMEM microbatching performance analysis.
Generates three publication-ready plots:
1. Heatmap comparison of NCCL vs NVSHMEM throughput
2. Line plots showing microbatch size impact on throughput
6. Scatter plot showing throughput vs latency trade-off
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_nccl_data():
    """Load and process NCCL microbatching data."""
    df = pd.read_csv('NCCL_Microbatching.csv', skiprows=1)
    df.columns = ['batch_size', 'microbatch_size', 'tokens_per_second', 'avg_iteration_time_ms']
    # Remove rows with missing data
    df = df.dropna()
    return df

def load_nvshmem_data():
    """Load and process NVSHMEM microbatching data."""
    # Read the file and process the complex format
    with open('NVSHMEM_Microbatching.csv', 'r') as f:
        lines = f.readlines()
    
    # Parse response time data (rows 3-8, columns indexed by batch size)
    batch_sizes = [2, 4, 8, 16, 32]
    microbatch_sizes = [1, 2, 4, 8, 16, 32]
    
    data = []
    
    # Parse throughput data (Tok/sec) - starts at line 3, column 9
    for i, mb_size in enumerate(microbatch_sizes):
        row_idx = i + 2  # Data starts at line 3 (index 2)
        if row_idx >= len(lines):
            break
        parts = lines[row_idx].strip().split(',')
        
        # Response times are in columns 1-5
        response_times = parts[1:6]
        # Throughput is in columns 8-12
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

def create_heatmap_comparison(nccl_df, nvshmem_df):
    """
    Plot 1: Side-by-side heatmaps comparing NCCL and NVSHMEM throughput.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get all unique batch and microbatch sizes
    batch_sizes = sorted(pd.concat([nccl_df['batch_size'], nvshmem_df['batch_size']]).unique())
    microbatch_sizes = sorted(pd.concat([nccl_df['microbatch_size'], nvshmem_df['microbatch_size']]).unique())
    
    # Create pivot tables for heatmaps
    nccl_pivot = nccl_df.pivot_table(
        values='tokens_per_second',
        index='microbatch_size',
        columns='batch_size',
        aggfunc='mean'
    )
    
    nvshmem_pivot = nvshmem_df.pivot_table(
        values='tokens_per_second',
        index='microbatch_size',
        columns='batch_size',
        aggfunc='mean'
    )
    
    # Reindex to ensure all sizes are included
    nccl_pivot = nccl_pivot.reindex(index=microbatch_sizes, columns=batch_sizes)
    nvshmem_pivot = nvshmem_pivot.reindex(index=microbatch_sizes, columns=batch_sizes)
    
    # Use the same color scale for both
    vmin = min(nccl_pivot.min().min(), nvshmem_pivot.min().min())
    vmax = max(nccl_pivot.max().max(), nvshmem_pivot.max().max())
    
    # NCCL heatmap
    sns.heatmap(nccl_pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Tokens/sec'}, ax=ax1,
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='gray')
    ax1.set_title('NCCL Throughput', fontweight='bold', pad=10)
    ax1.set_xlabel('Batch Size', fontweight='bold')
    ax1.set_ylabel('Microbatch Size', fontweight='bold')
    ax1.invert_yaxis()
    
    # NVSHMEM heatmap
    sns.heatmap(nvshmem_pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Tokens/sec'}, ax=ax2,
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='gray')
    ax2.set_title('NVSHMEM Throughput', fontweight='bold', pad=10)
    ax2.set_xlabel('Batch Size', fontweight='bold')
    ax2.set_ylabel('Microbatch Size', fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('plot1_heatmap_comparison.png', bbox_inches='tight')
    print("✓ Created plot1_heatmap_comparison.png")
    plt.close()

def create_microbatch_impact_plot(nccl_df, nvshmem_df):
    """
    Plot 2: Line plots showing the impact of microbatch size on throughput.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    
    # NCCL subplot
    batch_sizes = sorted(nccl_df['batch_size'].unique())
    for i, batch_size in enumerate(batch_sizes):
        subset = nccl_df[nccl_df['batch_size'] == batch_size].sort_values('microbatch_size')
        ax1.plot(subset['microbatch_size'], subset['tokens_per_second'],
                marker='o', linewidth=2, markersize=7, label=f'Batch={batch_size}',
                color=colors[i])
    
    ax1.set_xlabel('Microbatch Size', fontweight='bold')
    ax1.set_ylabel('Throughput (Tokens/sec)', fontweight='bold')
    ax1.set_title('NCCL: Microbatch Size Impact', fontweight='bold', pad=10)
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log', base=2)
    
    # NVSHMEM subplot
    batch_sizes = sorted(nvshmem_df['batch_size'].unique())
    for i, batch_size in enumerate(batch_sizes):
        subset = nvshmem_df[nvshmem_df['batch_size'] == batch_size].sort_values('microbatch_size')
        ax2.plot(subset['microbatch_size'], subset['tokens_per_second'],
                marker='s', linewidth=2, markersize=7, label=f'Batch={batch_size}',
                color=colors[i])
    
    ax2.set_xlabel('Microbatch Size', fontweight='bold')
    ax2.set_ylabel('Throughput (Tokens/sec)', fontweight='bold')
    ax2.set_title('NVSHMEM: Microbatch Size Impact', fontweight='bold', pad=10)
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('plot2_microbatch_impact.png', bbox_inches='tight')
    print("✓ Created plot2_microbatch_impact.png")
    plt.close()

def create_latency_heatmap_comparison(nccl_df, nvshmem_df):
    """
    Plot 1B: Side-by-side heatmaps comparing NCCL and NVSHMEM latency.
    Lower latency is better, so use reversed color scheme.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get all unique batch and microbatch sizes
    batch_sizes = sorted(pd.concat([nccl_df['batch_size'], nvshmem_df['batch_size']]).unique())
    microbatch_sizes = sorted(pd.concat([nccl_df['microbatch_size'], nvshmem_df['microbatch_size']]).unique())
    
    # Create pivot tables for heatmaps
    nccl_pivot = nccl_df.pivot_table(
        values='avg_iteration_time_ms',
        index='microbatch_size',
        columns='batch_size',
        aggfunc='mean'
    )
    
    nvshmem_pivot = nvshmem_df.pivot_table(
        values='avg_iteration_time_ms',
        index='microbatch_size',
        columns='batch_size',
        aggfunc='mean'
    )
    
    # Reindex to ensure all sizes are included
    nccl_pivot = nccl_pivot.reindex(index=microbatch_sizes, columns=batch_sizes)
    nvshmem_pivot = nvshmem_pivot.reindex(index=microbatch_sizes, columns=batch_sizes)
    
    # Use the same color scale for both
    vmin = min(nccl_pivot.min().min(), nvshmem_pivot.min().min())
    vmax = max(nccl_pivot.max().max(), nvshmem_pivot.max().max())
    
    # NCCL heatmap - reversed colormap (lower is better)
    sns.heatmap(nccl_pivot, annot=True, fmt='.0f', cmap='YlGnBu_r',
                cbar_kws={'label': 'Latency (ms)'}, ax=ax1,
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='gray')
    ax1.set_title('NCCL Latency', fontweight='bold', pad=10)
    ax1.set_xlabel('Batch Size', fontweight='bold')
    ax1.set_ylabel('Microbatch Size', fontweight='bold')
    ax1.invert_yaxis()
    
    # NVSHMEM heatmap
    sns.heatmap(nvshmem_pivot, annot=True, fmt='.0f', cmap='YlGnBu_r',
                cbar_kws={'label': 'Latency (ms)'}, ax=ax2,
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='gray')
    ax2.set_title('NVSHMEM Latency', fontweight='bold', pad=10)
    ax2.set_xlabel('Batch Size', fontweight='bold')
    ax2.set_ylabel('Microbatch Size', fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('plot1b_latency_heatmap_comparison.png', bbox_inches='tight')
    print("✓ Created plot1b_latency_heatmap_comparison.png")
    plt.close()

def create_latency_impact_plot(nccl_df, nvshmem_df):
    """
    Plot 2B: Line plots showing the impact of microbatch size on latency.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    
    # NCCL subplot
    batch_sizes = sorted(nccl_df['batch_size'].unique())
    for i, batch_size in enumerate(batch_sizes):
        subset = nccl_df[nccl_df['batch_size'] == batch_size].sort_values('microbatch_size')
        # Only plot if we have latency data
        subset_with_latency = subset.dropna(subset=['avg_iteration_time_ms'])
        if len(subset_with_latency) > 0:
            ax1.plot(subset_with_latency['microbatch_size'], 
                    subset_with_latency['avg_iteration_time_ms'],
                    marker='o', linewidth=2, markersize=7, label=f'Batch={batch_size}',
                    color=colors[i])
    
    ax1.set_xlabel('Microbatch Size', fontweight='bold')
    ax1.set_ylabel('Latency (ms)', fontweight='bold')
    ax1.set_title('NCCL: Microbatch Size Impact on Latency', fontweight='bold', pad=10)
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log', base=2)
    
    # NVSHMEM subplot
    batch_sizes = sorted(nvshmem_df['batch_size'].unique())
    for i, batch_size in enumerate(batch_sizes):
        subset = nvshmem_df[nvshmem_df['batch_size'] == batch_size].sort_values('microbatch_size')
        # Only plot if we have latency data
        subset_with_latency = subset.dropna(subset=['avg_iteration_time_ms'])
        if len(subset_with_latency) > 0:
            ax2.plot(subset_with_latency['microbatch_size'], 
                    subset_with_latency['avg_iteration_time_ms'],
                    marker='s', linewidth=2, markersize=7, label=f'Batch={batch_size}',
                    color=colors[i])
    
    ax2.set_xlabel('Microbatch Size', fontweight='bold')
    ax2.set_ylabel('Latency (ms)', fontweight='bold')
    ax2.set_title('NVSHMEM: Microbatch Size Impact on Latency', fontweight='bold', pad=10)
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('plot2b_latency_impact.png', bbox_inches='tight')
    print("✓ Created plot2b_latency_impact.png")
    plt.close()

def create_tradeoff_scatter(nccl_df, nvshmem_df):

    """
    Plot 6: Scatter plot showing throughput vs latency trade-off.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Prepare data
    nccl_clean = nccl_df.dropna(subset=['avg_iteration_time_ms', 'tokens_per_second'])
    nvshmem_clean = nvshmem_df.dropna(subset=['avg_iteration_time_ms', 'tokens_per_second'])
    
    # Normalize batch sizes for point sizing
    all_batch_sizes = pd.concat([nccl_clean['batch_size'], nvshmem_clean['batch_size']]).unique()
    min_batch = all_batch_sizes.min()
    max_batch = all_batch_sizes.max()
    
    def get_point_size(batch_size):
        # Scale between 100 and 500
        return 100 + 400 * (batch_size - min_batch) / (max_batch - min_batch)
    
    # Plot NCCL points
    for batch_size in sorted(nccl_clean['batch_size'].unique()):
        subset = nccl_clean[nccl_clean['batch_size'] == batch_size]
        ax.scatter(subset['avg_iteration_time_ms'], subset['tokens_per_second'],
                  s=get_point_size(batch_size), alpha=0.6, marker='o',
                  label=f'NCCL B={batch_size}', edgecolors='black', linewidth=1)
    
    # Plot NVSHMEM points
    for batch_size in sorted(nvshmem_clean['batch_size'].unique()):
        subset = nvshmem_clean[nvshmem_clean['batch_size'] == batch_size]
        ax.scatter(subset['avg_iteration_time_ms'], subset['tokens_per_second'],
                  s=get_point_size(batch_size), alpha=0.6, marker='^',
                  label=f'NVSHMEM B={batch_size}', edgecolors='black', linewidth=1)
    
    # Find and highlight best configurations (Pareto frontier)
    all_data = pd.concat([
        nccl_clean[['avg_iteration_time_ms', 'tokens_per_second']],
        nvshmem_clean[['avg_iteration_time_ms', 'tokens_per_second']]
    ], ignore_index=True)

    
    # Sort by latency
    sorted_data = all_data.sort_values('avg_iteration_time_ms')
    pareto_points = []
    max_throughput = -np.inf
    
    for _, row in sorted_data.iterrows():
        if row['tokens_per_second'] > max_throughput:
            pareto_points.append(row)
            max_throughput = row['tokens_per_second']
    
    if pareto_points:
        pareto_df = pd.DataFrame(pareto_points).sort_values('avg_iteration_time_ms')
        ax.plot(pareto_df['avg_iteration_time_ms'], pareto_df['tokens_per_second'],
               'r--', linewidth=2, alpha=0.5, label='Pareto Frontier', zorder=1)
    
    ax.set_xlabel('Average Iteration Time (ms)', fontweight='bold')
    ax.set_ylabel('Throughput (Tokens/sec)', fontweight='bold')
    ax.set_title('Throughput vs Latency Trade-off\n(Point size ∝ Batch Size)',
                fontweight='bold', pad=10)
    ax.legend(loc='best', frameon=True, shadow=True, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations for best points
    best_throughput_idx = all_data['tokens_per_second'].idxmax()
    best_latency_idx = all_data['avg_iteration_time_ms'].idxmin()
    
    best_throughput_time = float(all_data.loc[best_throughput_idx, 'avg_iteration_time_ms'])
    best_throughput_val = float(all_data.loc[best_throughput_idx, 'tokens_per_second'])
    best_latency_time = float(all_data.loc[best_latency_idx, 'avg_iteration_time_ms'])
    best_latency_val = float(all_data.loc[best_latency_idx, 'tokens_per_second'])
    
    ax.annotate('Best Throughput',
               xy=(best_throughput_time, best_throughput_val),
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.annotate('Best Latency',
               xy=(best_latency_time, best_latency_val),
               xytext=(10, -20), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('plot6_tradeoff_scatter.png', bbox_inches='tight')
    print("✓ Created plot6_tradeoff_scatter.png")
    plt.close()

def main():
    """Main execution function."""
    print("Loading data...")
    nccl_df = load_nccl_data()
    nvshmem_df = load_nvshmem_data()
    
    print(f"NCCL data points: {len(nccl_df)}")
    print(f"NVSHMEM data points: {len(nvshmem_df)}")
    print()
    
    print("Generating visualizations...")
    create_heatmap_comparison(nccl_df, nvshmem_df)
    create_latency_heatmap_comparison(nccl_df, nvshmem_df)
    create_microbatch_impact_plot(nccl_df, nvshmem_df)
    create_latency_impact_plot(nccl_df, nvshmem_df)
    create_tradeoff_scatter(nccl_df, nvshmem_df)
    
    print("\n" + "="*50)
    print("All plots generated successfully!")
    print("="*50)
    print("\nGenerated files:")
    print("  - plot1_heatmap_comparison.png (Throughput)")
    print("  - plot1b_latency_heatmap_comparison.png (Latency)")
    print("  - plot2_microbatch_impact.png (Throughput)")
    print("  - plot2b_latency_impact.png (Latency)")
    print("  - plot6_tradeoff_scatter.png")

if __name__ == "__main__":
    main()
