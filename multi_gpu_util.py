#!/usr/bin/env python3
"""
Multi-GPU Utilization Controller
Automatically detects all available GPUs and maintains utilization at target %
"""

import time
import torch
import threading
import argparse
import signal
import sys
import os
import select
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown

class GPUWorker:
    def __init__(self, gpu_id, target_util=50, check_interval=0.5, adjustment_rate=0.05):
        self.gpu_id = gpu_id
        self.target_util = target_util
        self.check_interval = check_interval
        self.adjustment_rate = adjustment_rate
        self.device = torch.device(f"cuda:{gpu_id}")
        
        self.running = False
        self.workload_size = 500  # Initial matrix size
        
        # Threads
        self.workload_thread = None
        self.monitor_thread = None
        
        # NVML Handle
        self.handle = nvmlDeviceGetHandleByIndex(self.gpu_id)
        self.name = torch.cuda.get_device_name(self.gpu_id)
        print(f"[GPU {self.gpu_id}] Initialized: {self.name}")

    def get_gpu_utilization(self):
        """Get current GPU utilization percentage"""
        try:
            util = nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu
        except Exception:
            return 0

    def _workload_loop(self):
        """The actual heavy lifting loop running in a separate thread"""
        while self.running:
            try:
                # Create matrices on specific GPU
                size = self.workload_size
                # We use torch.randn to ensure math intensity
                matrix_a = torch.randn(size, size, device=self.device)
                matrix_b = torch.randn(size, size, device=self.device)
                
                # Compute
                _ = torch.matmul(matrix_a, matrix_b)
                
                # Sync ensures the GPU actually finishes the work before loop repeats
                torch.cuda.synchronize(self.device)
            except Exception as e:
                print(f"[GPU {self.gpu_id}] Workload Error: {e}")
                time.sleep(1)

    def _monitor_loop(self):
        """Monitors and adjusts workload size"""
        while self.running:
            current_util = self.get_gpu_utilization()
            
            # --- Logic to Adjust Workload ---
            diff = self.target_util - current_util
            adjustment = 1.0 + (diff * self.adjustment_rate / 100.0)
            adjustment = max(0.8, min(1.2, adjustment)) # Clamp adjustment swing
            
            new_size = int(self.workload_size * adjustment)
            # Clamp matrix size (min 100, max 4000 to prevent OOM)
            new_size = max(100, min(4000, new_size))
            
            self.workload_size = new_size
            
            # Print status (Formatted to align columns)
            print(f"[GPU {self.gpu_id}] Util: {current_util:>3}% | Target: {self.target_util}% | Matrix Size: {self.workload_size}")
            
            time.sleep(self.check_interval)

    def start(self):
        """Start both the workload and monitor threads"""
        self.running = True
        
        # Start Workload Thread
        self.workload_thread = threading.Thread(target=self._workload_loop, daemon=True)
        self.workload_thread.start()
        
        # Start Monitor Thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Signals threads to stop and waits for them"""
        self.running = False
        if self.workload_thread and self.workload_thread.is_alive():
            self.workload_thread.join(timeout=1.0)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        # Clean up memory
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        print(f"[GPU {self.gpu_id}] Stopped.")

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Load Controller')
    parser.add_argument('--target', type=int, default=50, help='Target utilization %')
    args = parser.parse_args()

    # 1. Initialize NVML
    try:
        nvmlInit()
    except Exception as e:
        print(f"Error initializing NVML: {e}")
        sys.exit(1)

    # 2. Detect GPUs
    if not torch.cuda.is_available():
        print("No CUDA devices found.")
        sys.exit(1)
        
    device_count = torch.cuda.device_count()
    print(f"\nFound {device_count} GPU(s). Initializing controllers...\n")

    # 3. Create Workers
    workers = []
    for i in range(device_count):
        worker = GPUWorker(gpu_id=i, target_util=args.target)
        workers.append(worker)

    # 4. Start All
    print("Starting workloads...")
    for worker in workers:
        worker.start()

    print("\n---------------------------------------------------------")
    print(f"Running on {device_count} GPUs. Target Utilization: {args.target}%")
    print("Press 'q' + Enter to quit, or Ctrl+C")
    print("---------------------------------------------------------\n")

    # 5. Main Supervisor Loop
    running = True
    try:
        while running:
            # Check for 'q' input without blocking the main thread entirely
            if sys.stdin in select.select([sys.stdin], [], [], 1.0)[0]:
                line = sys.stdin.readline().strip()
                if line.lower() == 'q':
                    running = False
    except KeyboardInterrupt:
        print("\nCtrl+C detected.")
    finally:
        print("\nShutting down all GPUs...")
        for worker in workers:
            worker.stop()
        nvmlShutdown()
        print("Done.")

if __name__ == "__main__":
    main()
