/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_TO_CFG_PASSES
#define TRITON_TO_CFG_PASSES

include "mlir/Pass/PassBase.td"

def BuildCFG : Pass<"build-cfg", "mlir::ModuleOp"> {
  let summary = "Build Control Flow Graph from TTIR";
  let description = [{
    This pass analyzes the TTIR and builds a Control Flow Graph (CFG)
    representation. It can export the CFG to a file in text or DOT format.

    Options:
    - output-file: Specify output file (default: print to stdout)
    - export-dot: Export in DOT format (default: false)
  }];
  let constructor = "mlir::triton::createBuildCFGPass()";
}

#endif // TRITON_TO_CFG_PASSES
