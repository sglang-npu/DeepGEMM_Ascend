/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_DYNAMIC_MATMUL_ACLNN_TILING_H
#define CATLASS_DYNAMIC_MATMUL_ACLNN_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/tiling_context.h"

#include "../op_kernel/catlass_dynamic_matmul_tiling_data.h"
#include "../op_kernel/catlass_dynamic_matmul_tiling_key.h"

namespace optiling {
ge::graphStatus CatlassDynamicMatmulTilingFunc(gert::TilingContext *context);
}

#endif // CATLASS_DYNAMIC_MATMUL_ACLNN_TILING_H