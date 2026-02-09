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

#include "catlass_dynamic_matmul_tiling.h"
#include "register/op_def_registry.h"

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape* selfShape = context->GetInputShape(0);
    const gert::Shape* mat2Shape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);

    auto selfDimNum = selfShape->GetDimNum();
    auto mat2DimNum = mat2Shape->GetDimNum();
    if (selfDimNum != 2 || mat2DimNum != 2) {
        return ge::GRAPH_FAILED;
    }

    auto m = selfShape->GetDim(0);
    auto n = selfShape->GetDim(0);

    outShape->SetDimNum(selfDimNum);
    outShape->SetDim(0, m);
    outShape->SetDim(1, n);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto selfDataType = context->GetInputDataType(0);
    const auto mat2DataType = context->GetInputDataType(1);
    if (selfDataType != mat2DataType) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, selfDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class CatlassDynamicMatmul : public OpDef {
public:
    explicit CatlassDynamicMatmul(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("mat2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::CatlassDynamicMatmulTilingFunc);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(CatlassDynamicMatmul);
} // namespace ops