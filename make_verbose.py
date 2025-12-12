#!/usr/bin/env python3
"""
Make verbose logging conditional in nccl_pipeline_gpt2.cu
Wraps all [DEBUG] prints and verbose tensor checks with if (verbose) conditionals
"""

import re
import sys

def make_verbose_conditional(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Update stage_accumulate_gradients signature to accept verbose parameter
    content = content.replace(
        'void stage_accumulate_gradients(PipelineStage *stage) {',
        'void stage_accumulate_gradients(PipelineStage *stage, int verbose) {'
    )
    
    # 2. Wrap the debug prints in stage_accumulate_gradients with verbose check
    old_section = '''  // DEBUG: Check accumulated gradients
  printf("[Stage %d] [DEBUG CHECKPOINT 8] After gradient accumulation\\n",
         stage->pipe_config.stage_id);
  float acc_grad_norm = compute_gradient_norm(stage->grads_accumulated_memory,
                                              stage->num_parameters);
  printf("[Stage %d] Accumulated gradient norm: %.6e\\n",
         stage->pipe_config.stage_id, acc_grad_norm);
  check_tensor_debug("grads_accumulated", stage->grads_accumulated_memory,
                     stage->num_parameters, stage->pipe_config.stage_id, false);'''
    
    new_section = '''  // DEBUG: Check accumulated gradients
  if (verbose) {
    printf("[Stage %d] [DEBUG CHECKPOINT 8] After gradient accumulation\\n",
           stage->pipe_config.stage_id);
    float acc_grad_norm = compute_gradient_norm(stage->grads_accumulated_memory,
                                                stage->num_parameters);
    printf("[Stage %d] Accumulated gradient norm: %.6e\\n",
           stage->pipe_config.stage_id, acc_grad_norm);
    check_tensor_debug("grads_accumulated", stage->grads_accumulated_memory,
                       stage->num_parameters, stage->pipe_config.stage_id, false, verbose);
  }'''
    
    content = content.replace(old_section, new_section)
    
    # 3. Update all calls to stage_accumulate_gradients to pass verbose
    content = re.sub(
        r'stage_accumulate_gradients\(&stage\);',
        'stage_accumulate_gradients(&stage, verbose);',
        content
    )
    
    # 4. Update all check_tensor_debug calls to pass verbose parameter
    # This is complex, so we'll do it with a regex that finds the function calls
    # For now, let's just update the ones in main loop
    
    # 5. Wrap all [DEBUG] printf statements with if (verbose) checks
    # Pattern: if (rank == 0)\n    printf("[DEBUG]
    patterns = [
        (r'(  if \(rank == 0\))\n    (printf\("\[DEBUG\])', r'\1 && verbose)\n    \2'),
        (r'(    if \(rank == 0\))\n      (printf\("\[DEBUG\])', r'\1 && verbose)\n      \2'),
        (r'(      if \(rank == 0\))\n        (printf\("\[DEBUG\])', r'\1 && verbose)\n        \2'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # 6. Wrap standalone printf statements that contain [Stage %d] and stage operations
    # These need individual if (verbose) wrapping
    stage_print_patterns = [
        r'(    )(printf\("\[Stage %d\] Starting forward pass)',
        r'(      )(printf\("\[Stage %d\] Executing forward \(First Stage\))',
        r'(        )(printf\("\[Stage %d\] Sending activations to Stage)',
        r'(        )(printf\("\[Stage %d\] NCCL Send initiated)',
        r'(        )(printf\("\[Stage %d\] Waiting to receive from Stage)',
        r'(        )(printf\("\[Stage %d\] NCCL Recv complete)',
        r'(      )(printf\("\[Stage %d\] Executing forward\.\.\.)',
        r'(    )(printf\("\[Stage %d\] Forward pass DONE)',
        r'(      )(printf\("\[Stage %d\] Executing backward pass \(Last Stage\))',
        r'(        )(printf\("\[Stage %d\] Sending gradients to Stage)',
        r'(        )(printf\("\[Stage %d\] NCCL gradient send initiated)',
        r'(        )(printf\("\[Stage %d\] Waiting to receive gradients from Stage)',
        r'(        )(printf\("\[Stage %d\] NCCL gradient recv complete)',
        r'(      )(printf\("\[Stage %d\] Executing backward pass \(First Stage\))',
        r'(    )(printf\("\[Stage %d\] Backward pass DONE)',
    ]
    
    for pattern in stage_print_patterns:
        content = re.sub(pattern, r'\1if (verbose)\n\1  \2', content)
    
    # 7. Wrap gradient norm prints
    content = re.sub(
        r'(      )(printf\("\[Stage %d\] Gradient norm:)',
        r'\1if (verbose)\n\1  \2',
        content
    )
    
    # 8. Wrap Accum grad norm print
    content = re.sub(
        r'(    )(printf\("\[Stage %d\] Accum grad norm:)',
        r'\1if (verbose)\n\1  \2',
        content
    )
    
    # Write the modified content
    with open(filename, 'w') as f:
        f.write(content)
    
    # Count changes
    if content != original_content:
        print(f"✓ Modified {filename}")
        print(f"  - Added verbose parameter to stage_accumulate_gradients")
        print(f"  - Wrapped [DEBUG] prints with verbose checks")
        print(f"  - Wrapped stage operation prints with verbose checks")
    else:
        print(f"✗ No changes made to {filename}")

if __name__ == '__main__':
    filename = 'nccl_pipeline_gpt2.cu'
    make_verbose_conditional(filename)
    print("\nDone! Review the changes and compile to test.")
