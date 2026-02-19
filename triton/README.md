# Triton-Adapter-Opt Tool

Triton Ascend Adapter optimizer driver for MLIR-based IR transformations.

## Overview

This tool provides command-line access to all Ascend-specific MLIR passes, similar to `mlir-opt` but specialized for the Triton-to-Ascend compilation pipeline.

## Building

```bash
# From project root
mkdir build && cd build
cmake .. -G Ninja
ninja triton-adapter-opt

# Binary will be at:
# - build/bin/triton-adapter-opt (if built from root)
# - build/third_party/ascend/bin/triton-adapter-opt (if built from ascend subdirectory)
```

## Basic Usage

### Print Help
```bash
triton-adapter-opt --help
```

### Basic IR Transformations

Convert TTIR to Linalg:
```bash
triton-adapter-opt input.mlir --triton-to-linalg -o output.mlir
```

Run full compilation pipeline:
```bash
triton-adapter-opt input.mlir \
  --triton-to-structure \
  --discrete-mask-access-conversion \
  --triton-to-annotation \
  --triton-to-unstructure \
  --triton-to-hivm \
  --triton-to-hfusion \
  --triton-to-llvm \
  --triton-to-linalg \
  -o output.mlir
```

### Common Passes

| Pass | Description |
|------|-------------|
| `--triton-to-structure` | Linearize pointer and mask expressions |
| `--discrete-mask-access-conversion` | Convert discrete mask accesses to continuous+select |
| `--triton-to-annotation` | Convert compile hints to backend annotations |
| `--triton-to-unstructure` | Handle non-contiguous memory accesses |
| `--triton-to-hivm` | Convert to HIVM dialect |
| `--triton-to-hfusion` | Hardware fusion optimizations |
| `--triton-to-llvm` | Convert to LLVM dialect |
| `--triton-to-linalg` | Convert Triton to Linalg structured ops |

### Pass Parameters

Many passes accept parameters:

```bash
# Triton to Linalg with specific options
triton-adapter-opt input.mlir \
  --triton-to-linalg="named-ops=true enable-nd2nz-on-vector=false" \
  -o output.mlir

# Multiple parameters
triton-adapter-opt input.mlir \
  --triton-to-structure="enable-mask-fallback-conversion=true optimize-dynamic-offset=true" \
  --triton-to-linalg="compile-on-910-95=false" \
  -o output.mlir
```

## Debugging Workflow

### Step-by-Step Pass Execution

Instead of running the full pipeline at once, you can run passes individually to debug issues:

```bash
# Step 1: Structure the Triton IR
triton-adapter-opt kernel.ttir.mlir \
  --triton-to-structure \
  -o step1-structure.mlir

# Step 2: Convert mask accesses
triton-adapter-opt step1-structure.mlir \
  --discrete-mask-access-conversion \
  -o step2-mask.mlir

# Step 3: Add annotations
triton-adapter-opt step2-mask.mlir \
  --triton-to-annotation \
  -o step3-annotation.mlir

# And so on...
```

### Dump IR After Each Pass

```bash
triton-adapter-opt input.mlir \
  --triton-to-structure \
  --print-ir-after-all \
  --print-ir-module-scope \
  2>&1 | tee debug.log
```

### Verify IR

```bash
# Validate IR structure
triton-adapter-opt input.mlir --verify-diagnostics

# Check for malformed IR
triton-adapter-opt input.mlir --verify-each=true
```

## Advanced Usage

### Using Pass Pipeline

```bash
triton-adapter-opt input.mlir \
  --pass-pipeline='builtin.module(
    triton-to-structure,
    discrete-mask-access-conversion,
    triton-to-linalg
  )' \
  -o output.mlir
```

### Combine with Standard MLIR Passes

```bash
triton-adapter-opt input.mlir \
  --triton-to-linalg \
  --linalg-tile-and-fuse=tile-sizes=32,32 \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  -o output.ll.mlir
```

### Compare IR Before and After

```bash
# Generate baseline
triton-adapter-opt input.mlir -o baseline.mlir

# Generate optimized
triton-adapter-opt input.mlir \
  --triton-to-linalg="enable-nd2nz-on-vector=true" \
  -o optimized.mlir

# Compare
diff -u baseline.mlir optimized.mlir
```

## Testing

### Basic Test

```bash
# Create a simple test IR
cat > test.mlir << EOF
module {
  tt.func @simple_kernel(%arg0: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    %1 = tt.load %arg0[%0] : f32
    tt.return
  }
}
EOF

# Run pipeline
triton-adapter-opt test.mlir \
  --triton-to-linalg \
  -o test_output.mlir

# Check output
cat test_output.mlir
```

### Using Python Integration

```python
#!/usr/bin/env python3

from pathlib import Path
from triton._C.libtriton import ir, ascend
from triton.backends.ascend.backend.utils import _get_triton_adapter_opt_path
import subprocess
import tempfile

def test_pass(pipeline):
    """Test a pass pipeline"""
    test_mlir = '''
module {
  tt.func @test_kernel(%arg0: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    %1 = tt.load %arg0[%0] : f32
    tt.return
  }
}
'''

    triton_adapter_opt = _get_triton_adapter_opt_path()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.mlir"
        output_path = Path(tmpdir) / "output.mlir"

        input_path.write_text(test_mlir)

        cmd = [triton_adapter_opt, str(input_path)] + pipeline + ["-o", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Test passed")
            print("Output:", output_path.read_text())
        else:
            print("✗ Test failed:", result.stderr)

# Test different pipelines
test_pass(["--triton-to-structure"])
test_pass(["--triton-to-linalg"])
test_pass(["--triton-to-structure", "--triton-to-linalg"])
```

## Troubleshooting

### Error: Unknown Pass

```bash
# List all available passes
triton-adapter-opt --help | grep -A 100 "Available Passes"
```

### Error: Dialect Not Registered

Ensure the tool was built with all ascend dialects:
```bash
triton-adapter-opt input.mlir \
  --load-dialect-lib=third_party/ascend/lib/libTritonAscendDialect.so \
  --triton-to-linalg
```

### Debug Pass Execution

```bash
# Syntax debug
triton-adapter-opt input.mlir \
  --mlir-print-op-generic \
  --print-ir-before-all \
  --print-ir-after-all

# Control flow debug
triton-adapter-opt input.mlir \
  --mlir-print-stacktrace-on-diagnostic
```

## Performance Tips

### Process Large Files

For large IR files, use stdin/stdout:
```bash
triton-adapter-opt < large_input.mlir > output.mlir
```

### Parallel Batch Processing

```bash
# Process multiple files in parallel
find . -name "*.mlir" -print0 | xargs -0 -P4 -I{} \
  sh -c 'triton-adapter-opt {} -o $(dirname {})/$(basename {} .mlir).out.mlir'
```

## See Also

- [MLIR Documentation](https://mlir.llvm.org/)
- [Triton MLIR Passes](../../docs/en/triton_ir_passes.md)
- [Ascend Backend Architecture](../../docs/en/ascend_backend_arch.md)
- [IR Debugging Guide](../../docs/en/ir_debugging.md)

## License

Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
