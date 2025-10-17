#include "kernel_operator.h"

#define CUBE_BLOCK          16   // block的行列数，单位字节
#define CUBE_BLOCK_SIZE     256  // block的总大小，单位字节

__aicore__ inline uint32_t CeilCubeBlock(uint32_t len)
{
    return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
}

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t BlockLen(uint32_t blocks)
{
    return blocks * CUBE_BLOCK;
}

__aicore__ inline uint32_t BlockSize(uint32_t blocks)
{
    return blocks * CUBE_BLOCK_SIZE;
}

__aicore__ inline uint32_t CalcAOffset(uint32_t mi, uint32_t k, uint32_t msec_blocks,
    uint32_t ki, uint32_t ksec_blocks)
{
    return (mi * BlockLen(msec_blocks) * k + ki * BlockLen(ksec_blocks));
}

__aicore__ inline uint32_t CalcBOffset(uint32_t ni, uint32_t n, uint32_t ksec_blocks,
    uint32_t ki, uint32_t nsec_blocks)
{
    return (ki * BlockLen(ksec_blocks) * n + ni * BlockLen(nsec_blocks));
}

extern "C" __global__ __aicore__ void mmad_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;

    bool init_zero = true;

    AscendC::GlobalTensor<uint32_t> paramGM;
    paramGM.SetGlobalBuffer((__gm__ uint32_t *)params);

    uint32_t m = paramGM.GetValue(0);
    uint32_t n = paramGM.GetValue(1);
    uint32_t k = paramGM.GetValue(2);
    uint32_t m_sections = paramGM.GetValue(3); // 核间切分，m维度切分数量
    uint32_t n_sections = paramGM.GetValue(4); // 核间切分，n维度切分数量
    uint32_t m_sec_o_blocks = paramGM.GetValue(5); // 核内切分，m维度每块的大小，单位block
    uint32_t n_sec_o_blocks = paramGM.GetValue(6); // 核内切分，n维度每块的大小，单位block
    uint32_t k_o_iter_blocks = paramGM.GetValue(7); // 核内切分，k维度每块的大小，单位block
    uint32_t db_o_blocks = paramGM.GetValue(8); // 核内切分，double buffer每块的大小，单位block
    uint32_t batch = paramGM.GetValue(9);
    uint32_t k_iters = paramGM.GetValue(10);
    uint32_t m_blocks = paramGM.GetValue(11);
    uint32_t n_blocks = paramGM.GetValue(12);
    uint32_t k_blocks = paramGM.GetValue(13);
    uint32_t m_sc_blocks = paramGM.GetValue(14);
    uint32_t n_sc_blocks = paramGM.GetValue(15);

    uint32_t m_o_fix = paramGM.GetValue(16);
    uint32_t n_o_fix = paramGM.GetValue(17);
    uint32_t k_o_fix = paramGM.GetValue(18);
    uint32_t db_o_num = paramGM.GetValue(19);

    uint32_t m_parts = paramGM.GetValue(20);
    uint32_t n_parts = paramGM.GetValue(21);
    uint32_t r_m_parts = paramGM.GetValue(22);
    uint32_t r_n_parts = paramGM.GetValue(23);

    uint32_t r_m_blocks = paramGM.GetValue(24);
    uint32_t r_n_blocks = paramGM.GetValue(25);
    uint32_t r_k_blocks = paramGM.GetValue(26);
    uint32_t r_db_num = paramGM.GetValue(27);

    int a_buffer_size = BlockSize(m_sec_o_blocks * k_o_iter_blocks);
    int b_buffer_size = BlockSize(n_sec_o_blocks * k_o_iter_blocks);
    int c_buffer_size = BlockSize(m_sec_o_blocks * n_sec_o_blocks);

    AscendC::GlobalTensor<half> aGM;
    aGM.SetGlobalBuffer((__gm__ half *)a);
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    pipe.InitBuffer(inQueueA1, 2, a_buffer_size * sizeof(half)); // 每个A1分配出两块内存，每块大小为a_buffer_size * 2字节（half）

    AscendC::GlobalTensor<half> bGM;
    bGM.SetGlobalBuffer((__gm__ half *)b);
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    pipe.InitBuffer(inQueueB1, 2, b_buffer_size * sizeof(half));

    AscendC::GlobalTensor<float> cGM;
    cGM.SetGlobalBuffer((__gm__ float *)c);

    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
    pipe.InitBuffer(outQueueCO1, 1, c_buffer_size * sizeof(float)); // 每个A1分配出两块内存，每块大小为c_buffer_size * 4字节（float）

    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    pipe.InitBuffer(inQueueA2, 1, BlockSize(m_sec_o_blocks * db_o_blocks) * sizeof(half));
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    pipe.InitBuffer(inQueueB2, 1, BlockSize(db_o_blocks * n_sec_o_blocks) * sizeof(half));

    for(uint32_t bi = 0; bi < batch; bi++)
    {
        uint32_t offsetA = bi * m * k;
        uint32_t offsetB = bi * n * k;
        uint32_t offsetC = bi * m * n;
        uint32_t a_offset, b_offset;
        uint32_t msec_blocks, nsec_blocks, k_iter_blocks, db_blocks;
        uint32_t m_fix, n_fix, k_fix;
        uint32_t db_num;


        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t mCoreIndx = blockIdx % m_sections;
        uint32_t nCoreIndx = blockIdx / m_sections;
        bool is_last_m = mCoreIndx == (m_sections - 1);
        bool is_last_n = nCoreIndx == (n_sections - 1);

        if (is_last_m)
        {
            m_parts = r_m_parts;
        }
        if (is_last_n)
        {
            n_parts = r_n_parts;
        }

        offsetA += mCoreIndx * k * BlockLen(m_sc_blocks);
        offsetB += nCoreIndx * BlockLen(n_sc_blocks);
        offsetC += mCoreIndx * n * BlockLen(m_sc_blocks) + nCoreIndx * BlockLen(n_sc_blocks);

        AscendC::LocalTensor<half> a1Local, b1Local, a1, b1;
        AscendC::LocalTensor<half> a2Local, b2Local, a2, b2;

        AscendC::Nd2NzParams nd2nzParams;
        AscendC::LoadData2DParams loadDataParams;

        uint32_t dstOffset, srcOffset;
        AscendC::MmadParams mmadParams;
        AscendC::FixpipeParamsV220 fixpipeParams;

        for (uint32_t mi = 0; mi < m_parts; mi++)
        {
            if (is_last_m && (mi == m_parts - 1))
            {
                msec_blocks = r_m_blocks;
                m_fix = m_o_fix;
            }
            else
            {
                msec_blocks = m_sec_o_blocks;
                m_fix = 0;
            }

            for (uint32_t ni = 0; ni < n_parts; ni++)
            {
                if (is_last_n && (ni == n_parts - 1))
                {
                    nsec_blocks = r_n_blocks;
                    n_fix = n_o_fix;
                }
                else
                {
                    nsec_blocks = n_sec_o_blocks;
                    n_fix = 0;
                }

                init_zero = true;
                AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

                for (uint32_t ki = 0; ki < k_iters; ki ++)
                {
                    if (ki == (k_iters - 1))
                    {
                        k_iter_blocks = (r_db_num - 1) * db_o_blocks + r_k_blocks;
                        db_num = r_db_num;
                    }
                    else
                    {
                        k_iter_blocks = k_o_iter_blocks;
                        db_num = db_o_num;
                    }

                    a1Local = inQueueA1.AllocTensor<half>();
                    a_offset = CalcAOffset(mi, k, m_sec_o_blocks, ki, k_o_iter_blocks);
                    if (BlockLen(msec_blocks) - m_fix == 1) {
                        AscendC::DataCopy(a1Local, aGM[offsetA + a_offset], CeilCubeBlock(k) * CUBE_BLOCK);
                    } else {
                        nd2nzParams.ndNum = 1; // 每次只需要搬一个矩阵
                        nd2nzParams.nValue = BlockLen(msec_blocks); // 矩阵大小为nValue * dValue
                        nd2nzParams.dValue = BlockLen(k_iter_blocks);
                        nd2nzParams.srcNdMatrixStride = 0; // 只有1个矩阵，不需要偏移
                        nd2nzParams.srcDValue = k; // 每行偏移为k
                        nd2nzParams.dstNzC0Stride = BlockLen(msec_blocks); // 转换后，每行为BlockLen(msec_blocks) * CUBE_BLOCK个数据
                        nd2nzParams.dstNzNStride = 1;
                        nd2nzParams.dstNzMatrixStride = 0;
                        AscendC::DataCopy(a1Local, aGM[offsetA + a_offset], nd2nzParams);
                    }

                    b1Local = inQueueB1.AllocTensor<half>();
                    b_offset = CalcBOffset(ni, n, k_o_iter_blocks, ki, n_sec_o_blocks);
                    nd2nzParams.ndNum = 1;
                    nd2nzParams.nValue = BlockLen(k_iter_blocks);
                    nd2nzParams.dValue = BlockLen(nsec_blocks);
                    nd2nzParams.srcNdMatrixStride = 0;
                    nd2nzParams.srcDValue = n;
                    nd2nzParams.dstNzC0Stride = BlockLen(k_iter_blocks);
                    nd2nzParams.dstNzNStride = 1;
                    nd2nzParams.dstNzMatrixStride = 0;
                    AscendC::DataCopy(b1Local, bGM[offsetB + b_offset], nd2nzParams);

                    inQueueA1.EnQue(a1Local);
                    inQueueB1.EnQue(b1Local);

                    a1 = inQueueA1.DeQue<half>();
                    b1 = inQueueB1.DeQue<half>();

                    for (int kii = 0; kii < db_num; kii++)
                    {
                        if ((ki == (k_iters - 1)) && kii == (db_num - 1))
                        {
                            db_blocks = r_k_blocks;
                            k_fix = k_o_fix;
                        }
                        else
                        {
                            db_blocks = db_o_blocks;
                            k_fix = 0;
                        }

                        a2Local = inQueueA2.AllocTensor<half>();
                        b2Local = inQueueB2.AllocTensor<half>();

                        dstOffset = BlockSize(db_blocks);
                        if (BlockLen(msec_blocks) - m_fix == 1) {
                            srcOffset = BlockLen(kii * db_o_blocks * msec_blocks);
                        } else {
                            srcOffset = BlockSize(kii * db_o_blocks * (mi == (m_parts - 1) ? msec_blocks : m_sec_o_blocks));
                        }

                        // Nz -> Zz
                        loadDataParams.repeatTimes = db_blocks;
                        loadDataParams.srcStride = msec_blocks;
                        loadDataParams.dstGap = 0;
                        loadDataParams.ifTranspose = false;
                        for(int j = 0; j < msec_blocks; ++j)
                        {
                            AscendC::LoadData(a2Local[j * dstOffset], a1[BlockSize(j) + srcOffset], loadDataParams);
                        }
                        inQueueA2.EnQue<half>(a2Local);

                        dstOffset = BlockSize(nsec_blocks);
                        srcOffset = BlockSize(kii * db_o_blocks);

                        // Nz -> Zn
                        loadDataParams.repeatTimes = nsec_blocks;
                        loadDataParams.srcStride = k_iter_blocks;
                        loadDataParams.dstGap = 0;
                        loadDataParams.ifTranspose = true;
                        for(int j = 0; j < db_blocks; j++)
                        {
                            AscendC::LoadData(b2Local[j * dstOffset], b1[BlockSize(j) + srcOffset], loadDataParams);
                        }
                        inQueueB2.EnQue<half>(b2Local);

                        a2 = inQueueA2.DeQue<half>();
                        b2 = inQueueB2.DeQue<half>();

                        mmadParams.m = BlockLen(msec_blocks) - m_fix;
                        mmadParams.n = BlockLen(nsec_blocks) - n_fix;
                        mmadParams.k = BlockLen(db_blocks) - k_fix;

                        if(!init_zero)
                        {
                            mmadParams.cmatrixInitVal = false;
                            mmadParams.cmatrixSource = false;
                        }
                        else
                        {
                            mmadParams.cmatrixInitVal = true;
                            mmadParams.cmatrixSource = true;
                        }

                        AscendC::Mmad(c1Local, a2, b2, mmadParams);
                        if(init_zero)
                        {
                            init_zero = false;
                        }

                        inQueueA2.FreeTensor(a2);
                        inQueueB2.FreeTensor(b2);
                    }

                    inQueueA1.FreeTensor(a1);
                    inQueueB1.FreeTensor(b1);
                }

                outQueueCO1.EnQue<float>(c1Local);
                c1Local = outQueueCO1.DeQue<float>();
                fixpipeParams.nSize = BlockLen(nsec_blocks) - n_fix;
                fixpipeParams.mSize = BlockLen(msec_blocks) - m_fix;
                fixpipeParams.srcStride = BlockLen(msec_blocks);
                fixpipeParams.dstStride = n;

                fixpipeParams.ndNum = 1;
                fixpipeParams.srcNdStride = 0;
                fixpipeParams.dstNdStride = 0;

                AscendC::Fixpipe(
                    cGM[offsetC + mi * BlockLen(m_sec_o_blocks) * n + ni * BlockLen(n_sec_o_blocks)],
                    c1Local,
                    fixpipeParams);
                outQueueCO1.FreeTensor(c1Local);
            }
        }
    }
}