namespace optiling {

struct PFAShapeInfo {
    uint32_t b = 0;
    uint32_t n = 0;
    uint32_t s = 0;
    uint32_t d = 0;
    uint32_t h = 0;
    uint32_t t = 0;
};

BEGIN_TILING_DATA_DEF(PromptAttentionBaseParams)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, headNumSize);
    TILING_DATA_FIELD_DEF(uint32_t, seqSize);
    TILING_DATA_FIELD_DEF(uint32_t, headSize);
    TILING_DATA_FIELD_DEF(float, scaleValue);
    TILING_DATA_FIELD_DEF(int32_t, preTokens);
    TILING_DATA_FIELD_DEF(int32_t, nextTokens);
    TILING_DATA_FIELD_DEF(int32_t, blockSize);
    TILING_DATA_FIELD_DEF(int32_t, blockTableDim2);
    TILING_DATA_FIELD_DEF(int32_t, PABlockNumSum);
    TILING_DATA_FIELD_DEF(uint32_t, dimNumOfseq);
    TILING_DATA_FIELD_DEF(uint32_t, typeByteNum);
    TILING_DATA_FIELD_DEF(uint32_t, seqInnerSize);
    TILING_DATA_FIELD_DEF(uint32_t, prefixSeqInnerSize);
    TILING_DATA_FIELD_DEF(uint32_t, usePseShift);
    TILING_DATA_FIELD_DEF(uint32_t, useMask);
    TILING_DATA_FIELD_DEF(uint32_t, headNumRatio);
    TILING_DATA_FIELD_DEF(uint32_t, attenMaskElemType);
    TILING_DATA_FIELD_DEF(uint32_t, pseShiftTypeByteNum);
    TILING_DATA_FIELD_DEF(uint32_t, pseMaskMaxSize);
    TILING_DATA_FIELD_DEF(uint32_t, maskTypeByteNum);
    TILING_DATA_FIELD_DEF(uint32_t, outputTypeByteNum);
    TILING_DATA_FIELD_DEF(uint32_t, softmaxTypeByteNum);
    TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
    TILING_DATA_FIELD_DEF(uint32_t, alignedHeadSize);
    TILING_DATA_FIELD_DEF(uint32_t, splitS2);
    TILING_DATA_FIELD_DEF(uint32_t, splitD);
    TILING_DATA_FIELD_DEF(uint32_t, layoutType);
    TILING_DATA_FIELD_DEF(uint32_t, PAlayoutType);
    TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1Size);
    TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2Size);
    TILING_DATA_FIELD_DEF(uint32_t, maskKVsSize);
    TILING_DATA_FIELD_DEF(uint32_t, maskQsSize);
    TILING_DATA_FIELD_DEF(uint32_t, isLayoutSH);
    TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsNull);
    TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsKVNull);
    TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);
    TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);
    TILING_DATA_FIELD_DEF(uint32_t, deqScaleFlag);
    TILING_DATA_FIELD_DEF(uint32_t, deqScale2Flag);
    TILING_DATA_FIELD_DEF(uint32_t, isAntiPerchannel);
    TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid);
    TILING_DATA_FIELD_DEF(uint32_t, softmaxOuterSize);
    TILING_DATA_FIELD_DEF(uint32_t, isQuant2Perchannel);
    TILING_DATA_FIELD_DEF(uint32_t, isQuant2BF16);
    TILING_DATA_FIELD_DEF(uint32_t, isKvContinuous);
    TILING_DATA_FIELD_DEF(uint32_t, fromFused);
    TILING_DATA_FIELD_DEF(uint32_t, isBSNDOut);
    TILING_DATA_FIELD_DEF(uint32_t, isIFA);
    TILING_DATA_FIELD_DEF(uint32_t, isSoftMaxLseEnable);
    TILING_DATA_FIELD_DEF(uint32_t, isActualSharedPrefixLenNull);
    TILING_DATA_FIELD_DEF(uint32_t, isQHasLeftPadding);
    TILING_DATA_FIELD_DEF(uint32_t, isKVHasLeftPadding);
    TILING_DATA_FIELD_DEF(int64_t, keyAntiquantMode);
    TILING_DATA_FIELD_DEF(int64_t, valueAntiquantMode);
    TILING_DATA_FIELD_DEF(uint32_t, hasKeyAntiquantOffset);
    TILING_DATA_FIELD_DEF(uint32_t, isMsd);
    TILING_DATA_FIELD_DEF(uint32_t, isQuant2FP16);
    TILING_DATA_FIELD_DEF(uint32_t, ropeHeadSize);
    TILING_DATA_FIELD_DEF(uint32_t, qkHeadSize);
    TILING_DATA_FIELD_DEF(uint32_t, vHeadSize);
    TILING_DATA_FIELD_DEF(uint32_t, gOfMla);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionBaseParamsOp, PromptAttentionBaseParams)

BEGIN_TILING_DATA_DEF(PromptAttentionBaseApiBaseParams)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, headNumSize);
    TILING_DATA_FIELD_DEF(uint32_t, headSize);
    TILING_DATA_FIELD_DEF(uint32_t, maskTypeByteNum);
    
    TILING_DATA_FIELD_DEF(uint32_t, inputLayoutType);
    TILING_DATA_FIELD_DEF(uint32_t, kvHeadNumSize);
    TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, totalQBlkNum);
    TILING_DATA_FIELD_DEF(uint32_t, embeddingSizeV);
    TILING_DATA_FIELD_DEF(uint32_t, quantType);
    TILING_DATA_FIELD_DEF(uint32_t, dataShapeType);
    TILING_DATA_FIELD_DEF(uint32_t, scaleType);
    TILING_DATA_FIELD_DEF(uint64_t, workSize);
    TILING_DATA_FIELD_DEF(float, tor);
    TILING_DATA_FIELD_DEF(uint32_t, headStride);
    TILING_DATA_FIELD_DEF(uint32_t, maskStride);
    TILING_DATA_FIELD_DEF(uint32_t, isTriuMask);
    TILING_DATA_FIELD_DEF(uint32_t, isClamp);
    TILING_DATA_FIELD_DEF(uint32_t, clampMin);
    TILING_DATA_FIELD_DEF(uint32_t, clampMax);
    TILING_DATA_FIELD_DEF(uint32_t, tilingHeadSize);
    TILING_DATA_FIELD_DEF(uint32_t, tilingParaSize);
    TILING_DATA_FIELD_DEF(uint32_t, isLongSeq);
    TILING_DATA_FIELD_DEF(uint32_t, isAlibiMaskSqrt);
    TILING_DATA_FIELD_DEF(uint32_t, maskType);
    TILING_DATA_FIELD_DEF(uint32_t, alibiCompressOffset);
    TILING_DATA_FIELD_DEF(uint32_t, alibiLeftAlign);
    TILING_DATA_FIELD_DEF(uint32_t, ppMScalar);
    TILING_DATA_FIELD_DEF(uint32_t, ppNScalar);
    TILING_DATA_FIELD_DEF(uint32_t, totalQBlkNumFirst);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionBaseApiBaseParamsOp, PromptAttentionBaseApiBaseParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSeqParams)
    // Temporary reuse
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, CoreHeadNumTail);       // coreNStart
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, actualS1);              // coreNEnd
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, actualCoreNums);        // coreSidStart
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, singleCoreHeadNumSize); // coreSidEnd
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, coreSeqPosStart);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, coreSeqPosEnd);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSeqParamsOp, PromptAttentionSeqParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSplitCoreParams)
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, startBlkArray);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, endBlkArray);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSplitCoreParamsOp, PromptAttentionSplitCoreParams);

BEGIN_TILING_DATA_DEF(PromptAttentionSingleCoreParams)
    TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
    TILING_DATA_FIELD_DEF(uint32_t, singleProcessSOuterSize);
    TILING_DATA_FIELD_DEF(uint32_t, multiSmaxsInnerLoopTimes);
    TILING_DATA_FIELD_DEF(uint32_t, actualCoreNums);
    TILING_DATA_FIELD_DEF(uint32_t, pseShiftBatch);
    TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatch);
    TILING_DATA_FIELD_DEF(uint32_t, kvAntiquantSInnerSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSingleCoreParamsOp, PromptAttentionSingleCoreParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSingleCoreTensorSize)
    TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, pseShiftUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, attenMaskUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, maskSize);
    TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxSize);
    TILING_DATA_FIELD_DEF(uint32_t, softmaxSumSize);
    TILING_DATA_FIELD_DEF(uint32_t, softmaxExpSize);
    TILING_DATA_FIELD_DEF(uint32_t, softmaxValueSize);
    TILING_DATA_FIELD_DEF(uint32_t, spmTmpSize);
    TILING_DATA_FIELD_DEF(uint32_t, scmTmpSize);
    TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, tmpMMResBmm2PreUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, tmpSoftmaxBmm2UbSize);
    TILING_DATA_FIELD_DEF(uint32_t, selectSpaceUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, tmpSoftMaxV2Size);
    TILING_DATA_FIELD_DEF(uint32_t, mm1TmpUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, mm2TmpUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, kvAntiquantUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbMsdSize);
    TILING_DATA_FIELD_DEF(uint32_t, tempBmm2QueueMsdSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdInQueueSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdQRowSumBuffSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdAMaxTmpBuffSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdAMaxResBuffSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxResAmaxBuffSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxRowSumScaleBuffSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdScaleBuffSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdOffsetBuffSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm1BuffSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm2BuffSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdOutQueueSize);
    TILING_DATA_FIELD_DEF(uint32_t, msdComputeLines);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSingleCoreTensorSizeOp, PromptAttentionSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(PromptAttentionInitOutputParams)
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);
    TILING_DATA_FIELD_DEF(int64_t, totalOutputSize);
    TILING_DATA_FIELD_DEF(int64_t, totalSoftMaxLseOutputSize);
    TILING_DATA_FIELD_DEF(uint32_t, needInit);
    TILING_DATA_FIELD_DEF(uint32_t, isOneN);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionInitOutputParamsOp, PromptAttentionInitOutputParams)

BEGIN_TILING_DATA_DEF(PromptFlashAttentionTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingDataRect);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingDataRect);

    TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionBaseParams, promptAttentionBaseParams);
    TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSeqParams, promptAttentionSeqParams);
    TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSingleCoreParams, promptAttentionSingleCoreParams);
    TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSingleCoreTensorSize, promptAttentionTensorSizeRect);
    TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionInitOutputParams, promptAttentionInitOutputParams);

    TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingDataRect);
    TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingDataRect);
    TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataRect);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptFlashAttention, PromptFlashAttentionTilingData)

BEGIN_TILING_DATA_DEF(PFAInputParams)
    TILING_DATA_FIELD_DEF(int64_t, bSize);
    TILING_DATA_FIELD_DEF(int64_t, n2Size);
    TILING_DATA_FIELD_DEF(int64_t, gSize);
    TILING_DATA_FIELD_DEF(int64_t, s1Size);
    TILING_DATA_FIELD_DEF(int64_t, s2Size);
    TILING_DATA_FIELD_DEF(int64_t, alignedS2);
    TILING_DATA_FIELD_DEF(int64_t, dSize);
    TILING_DATA_FIELD_DEF(int64_t, valueDSize);
    TILING_DATA_FIELD_DEF(float, keepProb);
    TILING_DATA_FIELD_DEF(float, scaleValue);
    TILING_DATA_FIELD_DEF(int64_t, preTokens);
    TILING_DATA_FIELD_DEF(int64_t, nextTokens);
    // in pse encoding scenes, s1 and s2 might not equal with s1, s2 in Q, K
    TILING_DATA_FIELD_DEF(int64_t, pseS1Size);
    TILING_DATA_FIELD_DEF(int64_t, pseS2Size);
    TILING_DATA_FIELD_DEF(uint32_t, pseBSize);
    TILING_DATA_FIELD_DEF(uint32_t, bandIndex);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, blockTableDim2);

    // 1: BSH/BSND, 2: SBH, 3: BNSD
    TILING_DATA_FIELD_DEF(uint8_t, layoutType);
    // Paged Attention kvcache layout 0: BBH, 1: BNBD, 2: NZ
    TILING_DATA_FIELD_DEF(uint32_t, paCacheLayoutType);
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    TILING_DATA_FIELD_DEF(uint8_t, pseShapeType);
    // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType);
    // 0: fp16, 1: bool(uint8)
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType);
    // ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);
    // 0: high precise, 1: high performance, 2: invalid line high precise
    TILING_DATA_FIELD_DEF(uint8_t, implMode);
    TILING_DATA_FIELD_DEF(uint8_t, sparseType);
    TILING_DATA_FIELD_DEF(uint8_t, fromFused);
    TILING_DATA_FIELD_DEF(uint8_t, pseEncodeType);
    TILING_DATA_FIELD_DEF(uint8_t, isSoftMaxLseEnable);
    TILING_DATA_FIELD_DEF(uint16_t, remain);
    TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
    TILING_DATA_FIELD_DEF(uint32_t, pseType);
    TILING_DATA_FIELD_DEF(uint32_t, rsv1);
    TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
    TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
    TILING_DATA_FIELD_DEF(uint32_t, hasLearnableSink);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFAInputParamsOp, PFAInputParams)

BEGIN_TILING_DATA_DEF(PFAMultiCoreParams)
    TILING_DATA_FIELD_DEF(int32_t, coreNum);
    TILING_DATA_FIELD_DEF(int32_t, reserve);
    // BN2GS1.o
    TILING_DATA_FIELD_DEF(int64_t, totalSize);
    // BN2GS1.o / core_num
    TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
    TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 48, sparseStartIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFAMultiCoreParamsOp, PFAMultiCoreParams)

BEGIN_TILING_DATA_DEF(PFACoreParams)
    TILING_DATA_FIELD_DEF(int32_t, s1BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, s1BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, s1Vec2OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, s2BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, s2BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, s2OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, dBaseSize);
    TILING_DATA_FIELD_DEF(int32_t, dBaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, dOuterSize);
    TILING_DATA_FIELD_DEF(int32_t, bBaseSize);
    TILING_DATA_FIELD_DEF(int32_t, bBaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, bOuterSize);
    TILING_DATA_FIELD_DEF(int32_t, n2BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, n2BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, n2OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, gBaseSize);
    TILING_DATA_FIELD_DEF(int32_t, gBaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, gOuterSize);
    TILING_DATA_FIELD_DEF(int32_t, nRatio);
    TILING_DATA_FIELD_DEF(int32_t, rsvd);
    TILING_DATA_FIELD_DEF(int64_t, s1SparseValidSize);
    TILING_DATA_FIELD_DEF(int64_t, s2SparseValidSize);
    TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
    TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFACoreParamsOp, PFACoreParams)

BEGIN_TILING_DATA_DEF(PFATensorSizeParams)
    TILING_DATA_FIELD_DEF(int32_t, bmm1ResUbSize);
    TILING_DATA_FIELD_DEF(int32_t, attenMaskUbSize);
    TILING_DATA_FIELD_DEF(int32_t, pseUbSize);
    TILING_DATA_FIELD_DEF(int32_t, dropMaskUbSize);
    TILING_DATA_FIELD_DEF(int32_t, castUbSize);
    TILING_DATA_FIELD_DEF(int32_t, softmaxMaxUbSize);
    TILING_DATA_FIELD_DEF(int32_t, softmaxSumUbSize);
    TILING_DATA_FIELD_DEF(int32_t, softmaxExpUbSize);
    TILING_DATA_FIELD_DEF(int32_t, apiTmpBufferBytes);
    TILING_DATA_FIELD_DEF(int32_t, bmm2ResUbSize);
    TILING_DATA_FIELD_DEF(int32_t, inputQueBytes);
    TILING_DATA_FIELD_DEF(int32_t, outputQueBytes);
    // API buffer use remain space of ub
    TILING_DATA_FIELD_DEF(int32_t, tmpBufBytes);
    TILING_DATA_FIELD_DEF(int32_t, softmaxMaxOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, softmaxSumOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, maxSumApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, pseTbufOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, dropoutApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, maxSumApiSize);
    TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiSize);
    TILING_DATA_FIELD_DEF(int32_t, dropoutApiSize);
    TILING_DATA_FIELD_DEF(int32_t, attenMaskApiSize);
    TILING_DATA_FIELD_DEF(int32_t, attenMaskApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2Size);
    TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2OffsetBytes);
    // workspace
    TILING_DATA_FIELD_DEF(int32_t, wkspSection1OffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, wkspSection2OffsetBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFATensorSizeParamsOp, PFATensorSizeParams)

BEGIN_TILING_DATA_DEF(MLAGeneralTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(PFAInputParams, PFAinputParams);
    TILING_DATA_FIELD_DEF_STRUCT(PFAMultiCoreParams, PFAmultiCoreParams);
    TILING_DATA_FIELD_DEF_STRUCT(PFACoreParams, PFAcoreParams);
    TILING_DATA_FIELD_DEF_STRUCT(PFATensorSizeParams, PFAtensorSizeParams);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
    TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataTailCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000000000000, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000000000001, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000000000002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000000000003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000000100002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000000100003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000010000002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000010000003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000010100002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000010100003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000020000002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000020000003, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000020100002, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_4000000000020100003, MLAGeneralTilingData)

BEGIN_TILING_DATA_DEF(PromptFlashAttentionBaseApiTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionBaseApiBaseParams, promptAttentionBaseApiBaseParams);
    TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSplitCoreParams, promptAttentionSplitCoreParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_1000000000000112288, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_1000000000000122288, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_2000000002004000012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_2000000000004001012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_2000000010004001012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_2000000000004000012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_2000000010004000012, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_2000000002004010112, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_2000000000004010112, PromptFlashAttentionBaseApiTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_2000000010004010112, PromptFlashAttentionBaseApiTilingData)

BEGIN_TILING_DATA_DEF(InputParamsRegbase)
    TILING_DATA_FIELD_DEF(int64_t, bSize);
    TILING_DATA_FIELD_DEF(int64_t, n2Size);
    TILING_DATA_FIELD_DEF(int64_t, gSize);
    TILING_DATA_FIELD_DEF(int64_t, s1Size);
    TILING_DATA_FIELD_DEF(int64_t, s2Size);
    TILING_DATA_FIELD_DEF(int64_t, alignedS2);
    TILING_DATA_FIELD_DEF(int64_t, dSize);
    TILING_DATA_FIELD_DEF(int64_t, dSizeV);
    TILING_DATA_FIELD_DEF(int64_t, dSizeRope);
    TILING_DATA_FIELD_DEF(float, keepProb);
    TILING_DATA_FIELD_DEF(float, scaleValue);
    TILING_DATA_FIELD_DEF(int64_t, preTokens);
    TILING_DATA_FIELD_DEF(int64_t, nextTokens);
    // in pse encoding scenes, s1 and s2 might not equal with s1, s2 in Q, K
    TILING_DATA_FIELD_DEF(int64_t, pseS1Size);
    TILING_DATA_FIELD_DEF(int64_t, pseS2Size);
    TILING_DATA_FIELD_DEF(uint32_t, pseBSize);
    TILING_DATA_FIELD_DEF(uint32_t, bandIndex);

    // 1: BSH/BSND, 2: SBH, 3: BNSD
    TILING_DATA_FIELD_DEF(uint8_t, layoutType);
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    TILING_DATA_FIELD_DEF(uint8_t, pseShapeType);
    // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType);
    // 0: fp16, 1: bool(uint8)
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType);
    // ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);
    // 0: high precise, 1: high performance, 2: invalid line high precise
    TILING_DATA_FIELD_DEF(uint8_t, implMode);
    TILING_DATA_FIELD_DEF(uint8_t, sparseType);
    TILING_DATA_FIELD_DEF(uint8_t, needDropMaskOp);
    TILING_DATA_FIELD_DEF(uint8_t, dropMaskOuter);
    TILING_DATA_FIELD_DEF(uint8_t, pseEncodeType);
    TILING_DATA_FIELD_DEF(uint16_t, remain);
    TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
    TILING_DATA_FIELD_DEF(uint32_t, pseType);
    TILING_DATA_FIELD_DEF(uint32_t, rsv1);
    TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
    TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
    TILING_DATA_FIELD_DEF(int64_t, s1SparseValidSize);
    TILING_DATA_FIELD_DEF(int64_t, s2SparseValidSize);
    TILING_DATA_FIELD_DEF(int64_t, seed);
    TILING_DATA_FIELD_DEF(int64_t, offset);
    TILING_DATA_FIELD_DEF(int64_t, keepProbUint8);
    TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
    TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);

    // PFA
    TILING_DATA_FIELD_DEF(uint8_t, deqScaleFlag);
    TILING_DATA_FIELD_DEF(uint8_t, deqScale2Flag);
    TILING_DATA_FIELD_DEF(uint8_t, isActualSeqLengthsNull);
    TILING_DATA_FIELD_DEF(uint8_t, isActualSeqLengthsKVNull);
    TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);
    TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);
    TILING_DATA_FIELD_DEF(uint8_t, isKvContinuous);
    TILING_DATA_FIELD_DEF(uint8_t, fromFused);
    TILING_DATA_FIELD_DEF(uint8_t, isBSNDOut);
    TILING_DATA_FIELD_DEF(uint8_t, isGqa);
    TILING_DATA_FIELD_DEF(uint8_t, isSoftMaxLseEnable);
    TILING_DATA_FIELD_DEF(uint8_t, isActualSharedPrefixLenNull);
    TILING_DATA_FIELD_DEF(uint8_t, isQHasLeftPadding);
    TILING_DATA_FIELD_DEF(uint8_t, isKVHasLeftPadding);
    TILING_DATA_FIELD_DEF(uint32_t, ropeHeadSize);
    TILING_DATA_FIELD_DEF(uint32_t, prefixSeqInnerSize);
    TILING_DATA_FIELD_DEF(uint32_t, headNumRatio);
    TILING_DATA_FIELD_DEF(int32_t, blockSize);
    TILING_DATA_FIELD_DEF(int32_t, blockTableDim2);
    TILING_DATA_FIELD_DEF(int32_t, paBlockNumSum);
    TILING_DATA_FIELD_DEF(uint32_t, attenMaskS1Size);
    TILING_DATA_FIELD_DEF(uint32_t, kvSplitPart);
    TILING_DATA_FIELD_DEF(uint32_t, accumOutSize);
    TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize);

    TILING_DATA_FIELD_DEF(uint8_t, paLayoutType);
    TILING_DATA_FIELD_DEF(uint8_t, isRowInvalid);
    TILING_DATA_FIELD_DEF(uint8_t, isPostQuantPerChnl);
    TILING_DATA_FIELD_DEF(uint8_t, isPostQuantBF16);
    TILING_DATA_FIELD_DEF(uint16_t, antiquantPerTensorFlag);
    TILING_DATA_FIELD_DEF(uint16_t, antiquantPerHeadFlag);
    TILING_DATA_FIELD_DEF(uint32_t, antiquantParaSeqSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InputParamsRegbaseOp, InputParamsRegbase)

BEGIN_TILING_DATA_DEF(MultiCoreParamsRegbase)
    TILING_DATA_FIELD_DEF(int32_t, coreNum);
    TILING_DATA_FIELD_DEF(int64_t, totalSize);
    TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
    TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
    TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, bnStartIdx);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 48, sparseStartIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MultiCoreParamsRegbaseOp, MultiCoreParamsRegbase)

BEGIN_TILING_DATA_DEF(DropmaskParamsRegbase)
    TILING_DATA_FIELD_DEF(int32_t, multiCoreFactorSize);
    TILING_DATA_FIELD_DEF(int32_t, baseUbCalSize);
    TILING_DATA_FIELD_DEF(int64_t, multiCoreTotalSize);
    TILING_DATA_FIELD_DEF(int64_t, shapeTotalSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DropmaskParamsRegbaseOp, DropmaskParamsRegbase)

BEGIN_TILING_DATA_DEF(InitOutputParams)
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);
    TILING_DATA_FIELD_DEF(uint8_t, needInit);
    TILING_DATA_FIELD_DEF(uint8_t, isOneN);
    TILING_DATA_FIELD_DEF_ARR(uint8_t, 2, rsvd);
    TILING_DATA_FIELD_DEF(int64_t, totalOutputSize);
    TILING_DATA_FIELD_DEF(int64_t, totalSoftMaxLseOutputSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InitOutputParamsOp, InitOutputParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreSimplifiedTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(InputParamsRegbase, inputParamsRegbase);
    TILING_DATA_FIELD_DEF_STRUCT(MultiCoreParamsRegbase, multiCoreParamsRegbase);
    TILING_DATA_FIELD_DEF_STRUCT(DropmaskParamsRegbase, dropmaskParamsRegbase);
    TILING_DATA_FIELD_DEF_STRUCT(InitOutputParams, initOutputParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_1000000000000000090, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_10000000000000090, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_1002312000040001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_1002312000040021212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_1001311000000001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(PromptFlashAttention_1001311000000021212, FlashAttentionScoreSimplifiedTilingData)

class BufferNum {
public:
    // sum and max always use fp32, shape is (S1, 1), inner axis align 32B.
    size_t bufferS1S2Num; // unit: input dtype
    size_t bufferS1DNum;
    size_t bufferExpNum; // unit: input dtype, shape: [S1, 1], inner axis align 32B.
};
} // namespace optiling


