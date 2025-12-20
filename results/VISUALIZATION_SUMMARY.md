# Visualization Summary

This document describes all the generated plots for NCCL vs NVSHMEM microbatching performance analysis.

## Generated Plots

### **Throughput Visualizations**

#### 1. `plot1_heatmap_comparison.png` - Throughput Heatmaps

- **Purpose**: Side-by-side comparison of NCCL and NVSHMEM throughput performance
- **Metrics**: Tokens per second (higher is better)
- **X-axis**: Batch Size (2, 4, 8, 16, 32)
- **Y-axis**: Microbatch Size (1, 2, 4, 8, 16, 32)
- **Color**: YlOrRd (Yellow-Orange-Red), darker = higher throughput
- **Key Insights**:
  - Quickly identify optimal batch/microbatch combinations
  - Compare performance patterns between NCCL and NVSHMEM
  - NVSHMEM generally shows better throughput across configurations

#### 2. `plot2_microbatch_impact.png` - Throughput vs Microbatch Size

- **Purpose**: Show how microbatch size affects throughput for different batch sizes
- **Metrics**: Tokens per second
- **X-axis**: Microbatch Size (log scale, base 2)
- **Y-axis**: Throughput (Tokens/sec)
- **Lines**: Different colors for each batch size configuration
- **Key Insights**:
  - Smaller microbatch sizes generally improve throughput
  - Larger batch sizes achieve higher absolute throughput
  - Diminishing returns as microbatch size decreases

---

### **Latency Visualizations**

#### 3. `plot1b_latency_heatmap_comparison.png` - Latency Heatmaps

- **Purpose**: Side-by-side comparison of NCCL and NVSHMEM latency
- **Metrics**: Average iteration time in milliseconds (lower is better)
- **X-axis**: Batch Size (2, 4, 8, 16, 32)
- **Y-axis**: Microbatch Size (1, 2, 4, 8, 16, 32)
- **Color**: YlGnBu_r (Yellow-Green-Blue reversed), darker = lower latency (better)
- **Key Insights**:
  - Identify lowest latency configurations
  - Compare latency characteristics between backends
  - Smaller batch sizes generally have lower latency
  - NCCL shows competitive latency, especially at smaller configurations

#### 4. `plot2b_latency_impact.png` - Latency vs Microbatch Size

- **Purpose**: Show how microbatch size affects latency for different batch sizes
- **Metrics**: Average iteration time (ms)
- **X-axis**: Microbatch Size (log scale, base 2)
- **Y-axis**: Latency (ms)
- **Lines**: Different colors for each batch size configuration
- **Key Insights**:
  - Smaller microbatch sizes can reduce latency
  - Batch size has the largest impact on latency
  - Trade-offs between throughput optimization and latency minimization

---

### **Combined Analysis**

#### 5. `plot6_tradeoff_scatter.png` - Throughput vs Latency Trade-off

- **Purpose**: Visualize the throughput-latency frontier across all configurations
- **X-axis**: Average Iteration Time (ms) - latency
- **Y-axis**: Throughput (Tokens/sec)
- **Point markers**:
  - Circles = NCCL
  - Triangles = NVSHMEM
  - Size ∝ Batch Size
- **Special features**:
  - **Red dashed line**: Pareto frontier (optimal configurations)
  - **Yellow annotation**: Best throughput configuration
  - **Green annotation**: Best latency configuration
- **Key Insights**:
  - Visualize the fundamental trade-off between speed and latency
  - Identify Pareto-optimal configurations for different use cases
  - NVSHMEM configurations dominate the Pareto frontier
  - Helps select the right configuration based on whether you optimize for throughput or latency

---

## How to Regenerate

To regenerate all plots, run:

```bash
cd results
source venv/bin/activate
python create_plots.py
```

## Data Sources

- **NCCL**: `NCCL_Microbatching.csv` (18 data points)
- **NVSHMEM**: `NVSHMEM_Microbatching.csv` (20 data points)

## Plot Quality

- All plots are generated at **300 DPI** for publication quality
- Use serif fonts for professional appearance
- Consistent color schemes for easy comparison
- Grid lines and legends for readability

## Key Findings Summary

### Throughput

- **Best overall throughput**: NVSHMEM with batch_size=32, microbatch_size varies
- **Optimal for NCCL**: Smaller microbatch sizes with larger batch sizes
- **Optimal for NVSHMEM**: Consistently strong across configurations

### Latency

- **Best latency**: Smaller batch sizes (2-4) with optimized microbatch sizes
- **NCCL advantage**: Competitive latency at small configurations
- **NVSHMEM advantage**: Better latency at larger batch sizes

### Trade-offs

- **High throughput required**: Use batch_size=32 with NVSHMEM (expect ~1800ms latency)
- **Low latency required**: Use batch_size=2-4 with smaller microbatch sizes (~135-250ms)
- **Balanced**: batch_size=8-16 configurations on the Pareto frontier

---

## Citation

If using these visualizations in a publication, ensure you:

1. Mention the microbatching configurations tested
2. Specify the hardware used (GPU model, interconnect)
3. Note NCCL and NVSHMEM versions
4. Include batch size and microbatch size ranges
