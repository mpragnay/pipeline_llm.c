#!/usr/bin/env python3
"""
Add verbose flag wrapping to nccl_pipeline_gpt2.cu
This script wraps debug print statements with if (verbose) checks
"""

import re
import sys

def add_verbose_wrapping(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        
        # Check if this is a print statement to wrap
        should_wrap = False
        
        # Patterns to wrap
        if any(pattern in line for pattern in [
            '[DEBUG]',
            'check_tensor_debug(',
            '] Starting forward',
            '] Executing forward',
            '] Sending activations',
            '] Waiting to receive',
            '] NCCL',  
            '] Executing backward',
            '] Gradient norm:',
            '] Sending gradients',
            '] Backward pass DONE',
            '] Forward pass DONE',
            '] Accum grad norm',
        ]):
            should_wrap = True
        
        # Don't wrap loss output or critical messages
        if 'Loss:' in line or 'Training completed' in line or 'Parameters synchronized' in line:
            should_wrap = False
            
        if should_wrap and 'printf(' in line and 'if (verbose)' not in lines[max(0, i-1)]:
            # Add verbose check
            modified_lines.append(f"{indent}if (verbose) {{\n")
            modified_lines.append(line)
            
            # Check if statement continues on next line
            if not line.rstrip().endswith(';'):
                i += 1
                while i < len(lines) and not lines[i].rstrip().endswith(';'):
                    modified_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    modified_lines.append(lines[i])
            
            modified_lines.append(f"{indent}}}\n")
        else:
            modified_lines.append(line)
        
        i += 1
    
    # Write back
    with open(filename +'.verbose', 'w') as f:
        f.writelines(modified_lines)
    
    print(f"Modified file written to {filename}.verbose")
    print("Review changes and replace original if correct")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 add_verbose.py nccl_pipeline_gpt2.cu")
        sys.exit(1)
    
    add_verbose_wrapping(sys.argv[1])
