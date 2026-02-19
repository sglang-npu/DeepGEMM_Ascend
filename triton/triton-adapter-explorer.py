#!/usr/bin/env python3
"""
Triton Adapter Explorer - Manual IR Transformation Tool

This is a Python-based tool for debugging and exploring Triton to Ascend IR
transformations. While not as fast as the C++ version, it provides immediate
access to all passes for development and debugging.
"""

import argparse
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

try:
    from triton._C.libtriton import ir, ascend
except ImportError:
    print("Error: Could not import triton modules.")
    print("Please build Triton first or run from the Python environment where Triton is installed.")
    sys.exit(1)


class TritonAdapterExplorer:
    """Python-based IR transformation tool for Triton Ascend compilation pipeline."""

    def __init__(self):
        """Initialize the explorer with MLIR context and dialects."""
        try:
            self.context = ir.context()
            ir.load_dialects(self.context)
            ascend.load_dialects(self.context)
            print("✓ MLIR context and dialects loaded", file=sys.stderr)
        except Exception as e:
            print(f"✗ Error loading dialects: {e}", file=sys.stderr)
            raise

    def load_mlir(self, path: str):
        """Load MLIR file from the given path."""
        try:
            print(f"✓ Loading MLIR file: {path}", file=sys.stderr)
            return ir.parse_mlir_module(path, self.context)
        except Exception as e:
            print(f"✗ Error loading MLIR file: {e}", file=sys.stderr)
            raise

    def print_module(self, module, stage_name=""):
        """Print the IR with optional stage header."""
        if stage_name:
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"IR After: {stage_name}", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
        print(str(module))

    def run_pass(self, module, pass_name: str, **options):
        """Run a single Pass on the module."""
        print(f"\n[Pass] Running: {pass_name}", file=sys.stderr)
        try:
            pm = ir.pass_manager(self.context)
            pm.enable_debug()

            if pass_name == "triton-to-structure":
                enable_mask_fallback_conversion = options.get("enable_mask_fallback_conversion", True)
                optimize_dynamic_offset = options.get("optimize_dynamic_offset", True)
                ascend.passes.ttir.add_triton_to_structure(
                    pm, enable_mask_fallback_conversion, optimize_dynamic_offset
                )
            elif pass_name == "discrete-mask-access-conversion":
                compile_on_910_95 = options.get("compile_on_910_95", False)
                force_simt_template = options.get("force_simt_template", False)
                ascend.passes.ttir.add_discrete_mask_access_conversion(
                    pm, compile_on_910_95, force_simt_template
                )
            elif pass_name == "triton-to-annotation":
                ascend.passes.ttir.add_triton_to_annotation(pm)
            elif pass_name == "triton-to-unstructure":
                compile_on_910_95 = options.get("compile_on_910_95", False)
                force_simt_template = options.get("force_simt_template", False)
                ascend.passes.ttir.add_triton_to_unstructure(
                    pm, compile_on_910_95, force_simt_template
                )
            elif pass_name == "triton-to-hivm":
                ascend.passes.ttir.add_triton_to_hivm(pm)
            elif pass_name == "triton-to-hfusion":
                ascend.passes.ttir.add_triton_to_hfusion(pm)
            elif pass_name == "triton-to-llvm":
                ascend.passes.ttir.add_triton_to_llvm(pm)
            elif pass_name == "triton-to-linalg":
                global_kernel = options.get("global_kernel", False)
                named_ops = options.get("named_ops", False)
                enable_nd2nz_on_vector = options.get("enable_nd2nz_on_vector", False)
                enable_select_analysis = options.get("enable_select_analysis", False)
                compile_on_910_95 = options.get("compile_on_910_95", False)
                ascend.passes.ttir.add_triton_to_linalg(
                    pm, global_kernel, named_ops, enable_nd2nz_on_vector,
                    enable_select_analysis, compile_on_910_95
                )
            elif pass_name == "bubble-up-operation":
                ascend.passes.ttir.add_bubble_up_operation(pm)
            else:
                raise ValueError(f"Unknown pass: {pass_name}")

            pm.run(module)
            print(f"  ✓ Pass completed", file=sys.stderr)
            return module
        except Exception as e:
            print(f"  ✗ Pass failed: {e}", file=sys.stderr)
            raise

    def run_pipeline(self, module, pipeline: list, dump_after_each=False):
        """Run a sequence of passes on the module."""
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"Starting Pipeline: {len(pipeline)} passes", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)

        for i, pass_config in enumerate(pipeline, 1):
            pass_name = pass_config["name"]
            options = pass_config.get("options", {})

            self.run_pass(module, pass_name, **options)

            if dump_after_each:
                self.print_module(module, f"Step {i}: {pass_name}")

        return module


def create_full_pipeline(compile_on_910_95=False, force_simt_template=False):
    """Create the standard Triton to Ascend compilation pipeline."""
    return [
        {
            "name": "triton-to-structure",
            "options": {
                "enable_mask_fallback_conversion": True,
                "optimize_dynamic_offset": True
            }
        },
        {
            "name": "discrete-mask-access-conversion",
            "options": {
                "compile_on_910_95": compile_on_910_95,
                "force_simt_template": force_simt_template
            }
        },
        {
            "name": "triton-to-annotation",
            "options": {}
        },
        {
            "name": "triton-to-unstructure",
            "options": {
                "compile_on_910_95": compile_on_910_95,
                "force_simt_template": force_simt_template
            }
        },
        {
            "name": "triton-to-hivm",
            "options": {}
        },
        {
            "name": "triton-to-hfusion",
            "options": {}
        },
        {
            "name": "triton-to-llvm",
            "options": {}
        },
        {
            "name": "triton-to-linalg",
            "options": {
                "global_kernel": False,
                "named_ops": False,
                "enable_nd2nz_on_vector": False,
                "enable_select_analysis": False,
                "compile_on_910_95": compile_on_910_95
            }
        }
    ]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Triton Adapter Explorer - Manual IR Transformation Tool",
        epilog="""
Examples:
  # Load and print IR
  python triton-adapter-explorer.py input.mlir

  # Run single pass
  python triton-adapter-explorer.py input.mlir -p triton-to-structure

  # Run multiple passes
  python triton-adapter-explorer.py input.mlir \
    -p triton-to-structure \
    -p triton-to-linalg

  # Run full pipeline with step-by-step output
  python triton-adapter-explorer.py input.mlir \
    --full-pipeline \
    --dump-after-each
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input", help="输入 MLIR 文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径 (默认: stdout)", default="-")

    parser.add_argument("-p", "--pass", dest="passes", action="append",
                       help="要运行的 Pass (可多次使用)",
                       choices=["triton-to-structure",
                               "triton-to-annotation",
                               "triton-to-linalg",
                               "triton-to-unstructure",
                               "triton-to-hivm",
                               "triton-to-hfusion",
                               "triton-to-llvm",
                               "discrete-mask-access-conversion",
                               "bubble-up-operation",
                               "full-pipeline"])

    parser.add_argument("--compile-on-910-95", action="store_true",
                       help="是否为 910_95 编译")
    parser.add_argument("--force-simt-template", action="store_true",
                       help="强制使用 SIMT 模板")

    parser.add_argument("--dump-after-each", action="store_true",
                       help="每个 Pass 后都输出 IR")
    parser.add_argument("--print-input", action="store_true",
                       help="打印输入 IR")

    args = parser.parse_args()

    # Initialize explorer
    try:
        explorer = TritonAdapterExplorer()
    except Exception as e:
        print(f"Failed to initialize: {e}", file=sys.stderr)
        sys.exit(1)

    # Load input
    try:
        module = explorer.load_mlir(args.input)
    except Exception as e:
        print(f"Failed to load input: {e}", file=sys.stderr)
        sys.exit(1)

    # Print input if requested
    if args.print_input:
        explorer.print_module(module, "Input")

    # Process passes
    if args.passes:
        pipeline = []

        for pass_name in args.passes:
            if pass_name == "full-pipeline":
                # Replace with full pipeline
                pipeline.extend(create_full_pipeline(
                    compile_on_910_95=args.compile_on_910_95,
                    force_simt_template=args.force_simt_template
                ))
            else:
                options = {}
                if pass_name in ["discrete-mask-access-conversion", "triton-to-unstructure"]:
                    options["compile_on_910_95"] = args.compile_on_910_95
                    options["force_simt_template"] = args.force_simt_template
                if pass_name == "triton-to-linalg":
                    options["compile_on_910_95"] = args.compile_on_910_95

                pipeline.append({"name": pass_name, "options": options})

        try:
            explorer.run_pipeline(module, pipeline, args.dump_after_each)
        except Exception as e:
            print(f"Pipeline failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Just print the input if no passes specified
        explorer.print_module(module, "Input (no passes specified)")

    # Output result
    try:
        output = str(module)
        if args.output == "-":
            print(output)
        else:
            Path(args.output).write_text(output)
            print(f"✓ Output written to: {args.output}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
