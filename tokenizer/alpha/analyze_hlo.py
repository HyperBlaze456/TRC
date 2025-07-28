"""
Analyze XLA HLO dumps to identify memory-intensive operations.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict


def parse_hlo_text(file_path: str) -> Dict[str, List[Tuple[str, int]]]:
    """Parse HLO text file and extract operation sizes."""
    operations = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Pattern to match operations with shapes
        # Example: %convolution.123 = f32[32,512,1000]{2,1,0} convolution(...)
        pattern = r'%(\w+\.\d+)\s*=\s*(\w+)\[([\d,]+)\].*?\s+(\w+)\('
        
        for match in re.finditer(pattern, content):
            op_name = match.group(1)
            dtype = match.group(2)
            shape_str = match.group(3)
            op_type = match.group(4)
            
            # Calculate size
            shape = [int(x) for x in shape_str.split(',')]
            dtype_size = 4 if dtype in ['f32', 's32', 'u32'] else 2  # bytes
            size_bytes = dtype_size
            for dim in shape:
                size_bytes *= dim
            
            if op_type not in operations:
                operations[op_type] = []
            operations[op_type].append((op_name, size_bytes))
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return operations


def analyze_hlo_dumps(dump_dir: str = "/tmp/xla_dump"):
    """Analyze all HLO dumps in the directory."""
    dump_path = Path(dump_dir)
    
    if not dump_path.exists():
        print(f"HLO dump directory {dump_dir} does not exist.")
        print("Run the debug script first to generate dumps.")
        return
    
    print(f"\nAnalyzing HLO dumps in {dump_dir}")
    print("="*80)
    
    # Find all HLO text files
    hlo_files = list(dump_path.glob("*.hlo.txt")) + list(dump_path.glob("*.txt"))
    
    if not hlo_files:
        print("No HLO files found.")
        return
    
    total_memory_by_op = {}
    large_operations = []
    
    for hlo_file in hlo_files:
        print(f"\nAnalyzing {hlo_file.name}...")
        operations = parse_hlo_text(str(hlo_file))
        
        # Aggregate by operation type
        for op_type, op_list in operations.items():
            if op_type not in total_memory_by_op:
                total_memory_by_op[op_type] = 0
            
            for op_name, size_bytes in op_list:
                total_memory_by_op[op_type] += size_bytes
                
                # Track large operations (> 100MB)
                if size_bytes > 100 * 1024 * 1024:
                    large_operations.append((hlo_file.name, op_type, op_name, size_bytes))
    
    # Print summary
    print("\n" + "="*80)
    print("MEMORY USAGE BY OPERATION TYPE")
    print("="*80)
    
    sorted_ops = sorted(total_memory_by_op.items(), key=lambda x: x[1], reverse=True)
    for op_type, total_bytes in sorted_ops[:20]:
        print(f"{op_type:30s}: {total_bytes / 1e9:10.2f} GB")
    
    print("\n" + "="*80)
    print("LARGE INDIVIDUAL OPERATIONS (>100MB)")
    print("="*80)
    
    sorted_large = sorted(large_operations, key=lambda x: x[3], reverse=True)
    for file_name, op_type, op_name, size_bytes in sorted_large[:20]:
        print(f"{file_name}: {op_type} {op_name}: {size_bytes / 1e9:.2f} GB")


def find_memory_peaks(dump_dir: str = "/tmp/xla_dump"):
    """Look for memory allocation patterns in HLO."""
    dump_path = Path(dump_dir)
    
    if not dump_path.exists():
        return
    
    print("\n" + "="*80)
    print("SEARCHING FOR MEMORY ALLOCATION PATTERNS")
    print("="*80)
    
    # Look for specific patterns that often cause memory issues
    patterns_to_check = [
        (r'reshape.*\[([\d,]+)\]', "Reshape operations"),
        (r'broadcast.*\[([\d,]+)\]', "Broadcast operations"),
        (r'dot.*\[([\d,]+)\]', "Dot products"),
        (r'convolution.*\[([\d,]+)\]', "Convolutions"),
        (r'reduce.*\[([\d,]+)\]', "Reductions"),
        (r'transpose.*\[([\d,]+)\]', "Transposes"),
        (r'concatenate.*\[([\d,]+)\]', "Concatenations"),
        (r'slice.*\[([\d,]+)\]', "Slices"),
    ]
    
    for hlo_file in dump_path.glob("*.txt"):
        try:
            with open(hlo_file, 'r') as f:
                content = f.read()
            
            print(f"\nIn {hlo_file.name}:")
            
            for pattern, desc in patterns_to_check:
                matches = re.findall(pattern, content)
                if matches:
                    total_elements = 0
                    max_shape = None
                    max_elements = 0
                    
                    for shape_str in matches:
                        shape = [int(x) for x in shape_str.split(',')]
                        elements = 1
                        for dim in shape:
                            elements *= dim
                        total_elements += elements
                        
                        if elements > max_elements:
                            max_elements = elements
                            max_shape = shape
                    
                    if max_elements > 1_000_000:  # Only show if > 1M elements
                        print(f"  {desc}: {len(matches)} ops, largest shape: {max_shape}, "
                              f"max elements: {max_elements / 1e6:.1f}M")
        
        except Exception as e:
            print(f"Error processing {hlo_file}: {e}")


if __name__ == "__main__":
    # Run analysis
    analyze_hlo_dumps()
    find_memory_peaks()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Run the debug script: python tokenizer/alpha/train_debug.py")
    print("2. When it fails, check /tmp/xla_dump for HLO files")
    print("3. Run this script again to analyze the dumps")
    print("4. Look for operations with unusually large memory requirements")